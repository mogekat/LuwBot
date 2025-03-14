import asyncio
import time
from typing import Dict, List, Optional, Set, Union

from loguru import logger

from .config import global_config
from .chat_stream import ChatStream
from .llm_generator import ResponseGenerator
from .message import MessageRecv
from .willing_manager import willing_manager
from ..models.utils_model import LLM_request

# 2023-12-10 修复follow-up系统只能运行一次的问题：
# 1. 修改了evaluate_and_respond方法，只在需要回复或没有消息时才停用跟踪
# 2. 修改了_tracking_task方法，只在跟踪器不活动时才清理
# 3. 修改了should_continue_tracking方法，触发评估但不终止跟踪
# 4. 添加了restart_tracking方法，在评估后重新启动跟踪
# 5. 增加了更多日志记录点，便于调试

class FollowUpTracker:
    """
    跟踪一个特定对话，收集后续消息并判断是否需要回复
    """
    def __init__(self, chat_stream: ChatStream, message_id: str, llm_model: LLM_request):
        self.chat_stream = chat_stream
        self.initial_message_id = message_id  # 机器人发送的消息ID
        self.follow_up_messages: List[MessageRecv] = []  # 后续的消息列表
        self.start_time = time.time()  # 开始跟踪的时间
        self.active = True  # 跟踪是否激活
        self.task: Optional[asyncio.Task] = None  # 异步任务引用
        self.llm_model = llm_model  # 用于判断是否需要回复的LLM模型
    
    def add_message(self, message: MessageRecv) -> None:
        """添加一条后续消息"""
        if self.active:
            self.follow_up_messages.append(message)
    
    def should_continue_tracking(self) -> bool:
        """判断是否应该继续跟踪"""
        # 检查跟踪是否已被手动停用
        if not self.active:
            return False
        
        # 判断是否应该触发评估（超时或达到消息数量上限）
        if (time.time() - self.start_time >= global_config.follow_up_timeout or 
            len(self.follow_up_messages) >= global_config.follow_up_max_messages):
            # 达到评估条件，但不终止跟踪
            return False
            
        # 跟踪仍然活跃
        return True
    
    def deactivate(self) -> None:
        """停止跟踪"""
        self.active = False
        if self.task:
            self.task.cancel()
    
    async def evaluate_and_respond(self) -> None:
        """评估后续消息并决定是否需要回复"""
        if not self.follow_up_messages:
            self.deactivate()
            return
        
        # 构建对话上下文
        context = self._build_context()
        
        # 使用LLM判断是否需要回复
        need_response = await self._check_need_response(context)
        
        if need_response:
            logger.info(f"[跟踪回复] 决定对 {self.chat_stream.stream_id} 的后续消息进行回复")
            # 设置意愿为高值，确保会回复
            willing_manager.set_willing(self.chat_stream.stream_id, 2.0)
            # 最后一条消息会通过正常的消息处理流程处理
            # 停用当前跟踪器，因为已经决定回复
            self.deactivate()
        else:
            # 如果不需要回复但时间超时或消息数量达到上限，也停用跟踪器
            if (time.time() - self.start_time >= global_config.follow_up_timeout or 
                len(self.follow_up_messages) >= global_config.follow_up_max_messages):
                logger.debug(f"[跟踪回复] 不需要回复，且超时或消息数量达到上限，停用跟踪器")
                self.deactivate()
            # 否则继续跟踪
            else:
                logger.debug(f"[跟踪回复] 不需要回复，继续跟踪 {self.chat_stream.stream_id} 的后续消息")
                # 不停用跟踪器，继续收集消息
    
    def _build_context(self) -> str:
        """构建对话上下文"""
        context = []
        for msg in self.follow_up_messages:
            if hasattr(msg, 'processed_plain_text'):
                if msg.message_info and msg.message_info.user_info:
                    sender = f"{msg.message_info.user_info.user_nickname}"
                    content = msg.processed_plain_text
                    context.append(f"{sender}: {content}")
        
        return "\n".join(context)
    
    async def _check_need_response(self, context: str) -> bool:
        """使用LLM判断是否需要回复"""
        prompt = global_config.follow_up_llm_prompt + "\n\n对话内容：\n" + context
        
        try:
            response, _ = await self.llm_model.generate_response(prompt)
            # 简单判断回复中是否包含"是"
            return "是" in response
        except Exception as e:
            logger.error(f"[跟踪回复] LLM判断失败: {e}")
            return False


class FollowUpManager:
    """
    管理所有正在跟踪的对话
    """
    def __init__(self):
        self.trackers: Dict[str, FollowUpTracker] = {}  # stream_id -> tracker
        self.active_message_ids: Set[str] = set()  # 正在跟踪的消息ID
        self.llm_model: LLM_request = self._initialize_llm_model()  # 确保不为None
    
    def _initialize_llm_model(self) -> LLM_request:
        """初始化LLM模型并返回"""
        # 检查是否配置了专用的follow_up_llm模型
        if global_config.follow_up_llm and global_config.follow_up_llm.get("name"):
            # 使用指定的follow_up_llm模型
            model = LLM_request(
                model=global_config.follow_up_llm,
                temperature=0.7,
                max_tokens=200
            )
            logger.info(f"[跟踪回复] 使用指定的follow_up_llm模型: {global_config.follow_up_llm.get('name')}")
        else:
            # 使用默认的小型模型
            model = LLM_request(
                model=global_config.llm_normal_minor,
                temperature=0.7,
                max_tokens=200
            )
            logger.info("[跟踪回复] 使用默认的llm_normal_minor模型")
        return model
    
    def start_tracking(self, chat_stream: ChatStream, message_id: str) -> None:
        """开始跟踪一个新的对话"""
        if not global_config.follow_up_enabled:
            return
        
        # 如果已经在跟踪这个对话，先停止旧的跟踪
        if chat_stream.stream_id in self.trackers:
            logger.info(f"[跟踪回复] 停止旧的跟踪任务 {chat_stream.stream_id}")
            self.stop_tracking(chat_stream.stream_id)
        
        # 创建新的跟踪器
        tracker = FollowUpTracker(chat_stream, message_id, self.llm_model)
        self.trackers[chat_stream.stream_id] = tracker
        self.active_message_ids.add(message_id)
        
        # 创建异步任务
        tracker.task = asyncio.create_task(self._tracking_task(chat_stream.stream_id))
        logger.info(f"[跟踪回复] 开始跟踪 {chat_stream.stream_id} 的对话，消息ID: {message_id}，超时时间: {global_config.follow_up_timeout}秒")
    
    def stop_tracking(self, stream_id: str) -> None:
        """停止跟踪一个对话"""
        if stream_id in self.trackers:
            tracker = self.trackers[stream_id]
            tracker.deactivate()
            self.active_message_ids.discard(tracker.initial_message_id)
            del self.trackers[stream_id]
            logger.debug(f"[跟踪回复] 停止跟踪 {stream_id} 的对话")
    
    def add_message(self, chat_stream: ChatStream, message: MessageRecv) -> None:
        """添加一条后续消息到跟踪器"""
        if not global_config.follow_up_enabled:
            return
            
        stream_id = chat_stream.stream_id
        if stream_id in self.trackers and self.trackers[stream_id].active:
            # 添加日志显示消息内容和追踪器状态
            self.trackers[stream_id].add_message(message)
            logger.debug(f"[跟踪回复] 添加消息到 {stream_id} 的跟踪器，"
                        f"当前消息数: {len(self.trackers[stream_id].follow_up_messages)}，" 
                        f"内容: {message.processed_plain_text[:30]}..."
                        f"追踪开始时间: {time.strftime('%H:%M:%S', time.localtime(self.trackers[stream_id].start_time))}")
    
    async def _tracking_task(self, stream_id: str) -> None:
        """跟踪任务，在超时或达到消息数量上限时评估并可能回复"""
        if stream_id not in self.trackers:
            return
            
        tracker = self.trackers[stream_id]
        
        while tracker.should_continue_tracking():
            await asyncio.sleep(1)  # 检查间隔
        
        # 评估并可能回复
        await tracker.evaluate_and_respond()
        
        # 只有在跟踪器已经不活动时才清理
        if stream_id in self.trackers and self.trackers[stream_id] == tracker and not tracker.active:
            logger.debug(f"[跟踪回复] 清理不活动的跟踪器 {stream_id}")
            self.stop_tracking(stream_id)
        else:
            # 如果跟踪器仍然活动，重置时间和评估状态
            if stream_id in self.trackers and self.trackers[stream_id] == tracker:
                logger.debug(f"[跟踪回复] 重启跟踪器 {stream_id}")
                await self.restart_tracking(stream_id)
    
    async def restart_tracking(self, stream_id: str) -> None:
        """重启跟踪，清空消息并重置开始时间"""
        if stream_id in self.trackers:
            tracker = self.trackers[stream_id]
            if tracker.active:
                # 保留跟踪器但重置状态
                tracker.start_time = time.time()  # 重置开始时间
                tracker.follow_up_messages = []  # 清空已收集的消息
                
                # 创建新的异步任务
                if tracker.task:
                    tracker.task.cancel()  # 取消旧任务
                tracker.task = asyncio.create_task(self._tracking_task(stream_id))
                
                logger.debug(f"[跟踪回复] 重新开始跟踪 {stream_id} 的对话")

# 创建全局实例
follow_up_manager = FollowUpManager() 
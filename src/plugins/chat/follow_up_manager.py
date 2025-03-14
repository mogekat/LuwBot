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
        # 判断时间是否超时
        if time.time() - self.start_time >= global_config.follow_up_timeout:
            return False
        
        # 判断消息数量是否达到上限
        if len(self.follow_up_messages) >= global_config.follow_up_max_messages:
            return False
            
        return self.active
    
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
        
        self.deactivate()
    
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
            self.stop_tracking(chat_stream.stream_id)
        
        # 创建新的跟踪器
        tracker = FollowUpTracker(chat_stream, message_id, self.llm_model)
        self.trackers[chat_stream.stream_id] = tracker
        self.active_message_ids.add(message_id)
        
        # 创建异步任务
        tracker.task = asyncio.create_task(self._tracking_task(chat_stream.stream_id))
        logger.debug(f"[跟踪回复] 开始跟踪 {chat_stream.stream_id} 的对话")
    
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
        if stream_id in self.trackers:
            self.trackers[stream_id].add_message(message)
            logger.debug(f"[跟踪回复] 添加消息到 {stream_id} 的跟踪器")
    
    async def _tracking_task(self, stream_id: str) -> None:
        """跟踪任务，在超时或达到消息数量上限时评估并可能回复"""
        if stream_id not in self.trackers:
            return
            
        tracker = self.trackers[stream_id]
        
        while tracker.should_continue_tracking():
            await asyncio.sleep(1)  # 检查间隔
        
        # 评估并可能回复
        await tracker.evaluate_and_respond()
        
        # 清理
        if stream_id in self.trackers and self.trackers[stream_id] == tracker:
            self.stop_tracking(stream_id)

# 创建全局实例
follow_up_manager = FollowUpManager() 
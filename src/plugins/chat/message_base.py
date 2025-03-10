from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Any, Dict

@dataclass
class Seg:
    """消息片段类，用于表示消息的不同部分
    
    Attributes:
        type: 片段类型，可以是 'text'、'image'、'seglist' 等
        data: 片段的具体内容
            - 对于 text 类型，data 是字符串
            - 对于 image 类型，data 是 base64 字符串
            - 对于 seglist 类型，data 是 Seg 列表
        translated_data: 经过翻译处理的数据（可选）
    """
    type: str
    data: Union[str, List['Seg']]
    

    # def __init__(self, type: str, data: Union[str, List['Seg']],):
    #     """初始化实例，确保字典和属性同步"""
    #     # 先初始化字典
    #     self.type = type
    #     self.data = data
        
    @classmethod  
    def from_dict(cls, data: Dict) -> 'Seg':
        """从字典创建Seg实例"""
        type=data.get('type')
        data=data.get('data')
        if type == 'seglist':
            data = [Seg.from_dict(seg) for seg in data]
        return cls(
            type=type,
            data=data
        )

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        result = {'type': self.type}
        if self.type == 'seglist':
            result['data'] = [seg.to_dict() for seg in self.data]
        else:
            result['data'] = self.data
        return result

@dataclass
class GroupInfo:
    """群组信息类"""
    platform: Optional[str] = None
    group_id: Optional[int] = None
    group_name: Optional[str] = None  # 群名称

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GroupInfo':
        """从字典创建GroupInfo实例
        
        Args:
            data: 包含必要字段的字典
            
        Returns:
            GroupInfo: 新的实例
        """
        return cls(
            platform=data.get('platform'),
            group_id=data.get('group_id'),
            group_name=data.get('group_name',None)
        )

@dataclass
class UserInfo:
    """用户信息类"""
    platform: Optional[str] = None
    user_id: Optional[int] = None
    user_nickname: Optional[str] = None  # 用户昵称
    user_cardname: Optional[str] = None  # 用户群昵称

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserInfo':
        """从字典创建UserInfo实例
        
        Args:
            data: 包含必要字段的字典
            
        Returns:
            UserInfo: 新的实例
        """
        return cls(
            platform=data.get('platform'),
            user_id=data.get('user_id'),
            user_nickname=data.get('user_nickname',None),
            user_cardname=data.get('user_cardname',None)
        )

@dataclass
class BaseMessageInfo:
    """消息信息类"""
    platform: Optional[str] = None
    message_id: Union[str,int,None] = None
    time: Optional[int] = None
    group_info: Optional[GroupInfo] = None
    user_info: Optional[UserInfo] = None

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        result = {}
        for field, value in asdict(self).items():
            if value is not None:
                if isinstance(value, (GroupInfo, UserInfo)):
                    result[field] = value.to_dict()
                else:
                    result[field] = value
        return result
    @classmethod
    def from_dict(cls, data: Dict) -> 'BaseMessageInfo':
        """从字典创建BaseMessageInfo实例
        
        Args:
            data: 包含必要字段的字典
            
        Returns:
            BaseMessageInfo: 新的实例
        """
        group_info = GroupInfo(**data.get('group_info', {}))
        user_info = UserInfo(**data.get('user_info', {}))
        return cls(
            platform=data.get('platform'),
            message_id=data.get('message_id'),
            time=data.get('time'),
            group_info=group_info,
            user_info=user_info
        )

@dataclass
class MessageBase:
    """消息类"""
    message_info: BaseMessageInfo
    message_segment: Seg
    raw_message: Optional[str] = None  # 原始消息，包含未解析的cq码

    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 包含所有非None字段的字典，其中：
                - message_info: 转换为字典格式
                - message_segment: 转换为字典格式
                - raw_message: 如果存在则包含
        """
        result = {
            'message_info': self.message_info.to_dict(),
            'message_segment': self.message_segment.to_dict()
        }
        if self.raw_message is not None:
            result['raw_message'] = self.raw_message
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'MessageBase':
        """从字典创建MessageBase实例
        
        Args:
            data: 包含必要字段的字典
            
        Returns:
            MessageBase: 新的实例
        """
        message_info = BaseMessageInfo(**data.get('message_info', {}))
        message_segment = Seg(**data.get('message_segment', {}))
        raw_message = data.get('raw_message',None)
        return cls(
            message_info=message_info,
            message_segment=message_segment,
            raw_message=raw_message
        )

    
    

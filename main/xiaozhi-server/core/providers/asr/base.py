from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class ASRProviderBase(ABC):
    @abstractmethod
    def save_audio_to_file(self, opus_data: List[bytes], session_id: str) -> str:
        """解码Opus数据并保存为WAV文件"""
        pass

    @abstractmethod
    async def speech_to_text(self, opus_data: List[bytes], session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """将语音数据转换为文本"""
        pass

    # --- 新增实时处理相关方法 ---
    @abstractmethod
    async def process_audio_chunk(self, opus_packet: bytes):
        """
        处理单个实时音频块 (Opus格式)。
        内部进行解码、缓冲、调用模型生成中间结果，并维护状态。
        此方法通常不直接返回识别文本给调用者。
        """
        pass

    @abstractmethod
    async def finalize_recognition(self) -> Optional[str]:
        """
        结束当前的实时识别流程，处理缓冲区中剩余的音频，
        获取最终识别结果，并重置内部状态。
        返回最终识别的文本。
        """
        pass

    # --- 新增属性或方法以获取配置 ---
    @property
    @abstractmethod
    def use_realtime_asr(self) -> bool:
        """获取是否配置为使用实时ASR"""
        pass
# --- END OF FILE core/providers/asr/base.py ---

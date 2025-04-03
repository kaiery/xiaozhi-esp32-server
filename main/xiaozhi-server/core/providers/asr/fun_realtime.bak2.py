import time
import wave
import os
import sys
import io

import numpy as np

from config.logger import setup_logging
from typing import Optional, Tuple, List, Dict
import uuid
import opuslib_next
from core.providers.asr.base import ASRProviderBase

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

TAG = __name__
logger = setup_logging()


# 捕获标准输出
class CaptureOutput:
    def __enter__(self):
        self._output = io.StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = self._output

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        self.output = self._output.getvalue()
        self._output.close()

        # 将捕获到的内容通过 logger 输出
        if self.output:
            logger.bind(tag=TAG).info(self.output.strip())


class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_dir")
        self.delete_audio_file = delete_audio_file
        # 从配置读取 use_realtime_asr
        self._use_realtime_asr = config.get("use_realtime_asr", False)  # 使用下划线表示内部状态

        # 实时语音识别相关参数 (从你的示例代码获取)
        self.chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
        self.encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
        self.decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化模型 (不需要 streaming=True 参数, 根据你的示例代码)
        logger.bind(tag=TAG).info(
            f"Initializing FunASR AutoModel for {'Realtime' if self._use_realtime_asr else 'Non-Realtime'} ASR...")
        with CaptureOutput():
            self.model = AutoModel(
                model=self.model_dir,
                disable_update=True,
                hub="hf",
                device='cpu'  # 或者根据你的环境配置 'cuda:0'
            )
        logger.bind(tag=TAG).info("FunASR AutoModel initialized.")

        # --- 实时处理所需的状态 ---
        self.decoder = opuslib_next.Decoder(16000, 1)  # 初始化解码器 (每个实例一个)
        self.pcm_chunk_buffer = b''  # 内部PCM缓冲区，用于凑齐chunk_stride
        self.cache = {}  # FunASR流式识别的缓存
        # chunk_stride = chunk_size[1] * 960 (sample rate 16000, chunk_size[1]*60ms = chunk_size[1]*0.06*16000 = chunk_size[1]*960 samples)
        self.chunk_stride_samples = self.chunk_size[1] * 960  # 一个处理块包含的采样点数
        self.chunk_stride_bytes = self.chunk_stride_samples * 2  # 每个采样点2字节 (16-bit PCM)
        self.last_intermediate_result = ""  # 记录上一次的中间结果，避免重复打印日志
        # --------------------------
        # 新增属性
        self.final_text_buffer = ""  # 用于累积最终文本
        self.min_silence_duration = 1.0  # 静音持续时间阈值(秒)
        self.last_audio_time = 0  # 最后收到音频的时间戳
        # 新增关键属性
        self.full_result_cache = ""  # 累积完整识别结果
        self.last_partial_result = ""  # 记录上次部分结果
        self.result_history = []  # 记录历史结果用于调试

    # --- 实现基类新增的属性 ---
    @property
    def use_realtime_asr(self) -> bool:
        return self._use_realtime_asr  # 返回内部状态

    def save_audio_to_file(self, opus_data: List[bytes], session_id: str) -> str:
        """将Opus音频数据解码并保存为WAV文件（非实时模式使用）"""
        file_name = f"asr_{session_id}_{uuid.uuid4()}.wav"
        file_path = os.path.join(self.output_dir, file_name)

        decoder = opuslib_next.Decoder(16000, 1)  # 16kHz, 单声道
        pcm_data = []

        for opus_packet in opus_data:
            try:
                pcm_frame = decoder.decode(opus_packet, 960)  # 960 samples = 60ms
                pcm_data.append(pcm_frame)
            except opuslib_next.OpusError as e:
                logger.bind(tag=TAG).error(f"Opus解码错误: {e}", exc_info=True)

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"".join(pcm_data))

        return file_path

    async def speech_to_text(self, opus_data: List[bytes], session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        语音转文本主处理逻辑。
        如果配置为实时模式，此方法理论上不应该被 receiveAudioHandle 调用，
        但保留它作为处理完整音频列表的【非流式】后备方法。
        """
        if self._use_realtime_asr:
            logger.bind(tag=TAG).warning(
                "speech_to_text called in realtime mode. This indicates potential logic flaw in caller. Performing non-streaming ASR on the full audio list as fallback.")
            # 注意：这里执行的是非流式识别，即使配置了实时模式。
            # 真正的实时处理是通过 process_audio_chunk 和 finalize_recognition 实现的。

        file_path = None
        try:
            # 保存音频文件
            start_time = time.time()
            file_path = self.save_audio_to_file(opus_data, session_id)
            logger.bind(tag=TAG).debug(
                f"[Non-Realtime Fallback] 音频文件保存耗时: {time.time() - start_time:.3f}s | 路径: {file_path}")

            # 使用FunASR模型进行【非流式】语音识别
            start_time = time.time()
            # 非流式调用不需要 cache 和 chunk 参数
            result = self.model.generate(
                input=file_path,
                cache={},  # 非流式调用 cache 为空
                language="auto",
                use_itn=True,
                batch_size_s=60,  # 根据需要调整
            )
            text = rich_transcription_postprocess(result[0]["text"]) if result and result[0].get("text") else ""
            logger.bind(tag=TAG).debug(
                f"[Non-Realtime Fallback] 语音识别耗时: {time.time() - start_time:.3f}s | 结果: {text}")

            return text, file_path

        except Exception as e:
            logger.bind(tag=TAG).error(f"[Non-Realtime Fallback] 语音识别失败: {e}", exc_info=True)
            return "", None

        finally:
            # 文件清理逻辑
            if self.delete_audio_file and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.bind(tag=TAG).debug(f"[Non-Realtime Fallback] 已删除临时音频文件: {file_path}")
                except Exception as e:
                    logger.bind(tag=TAG).error(f"[Non-Realtime Fallback] 文件删除失败: {file_path} | 错误: {e}")

    # --- 核心修改区域 ---
    async def process_audio_chunk(self, opus_packet: bytes):
        """处理音频块(终极版)"""
        try:
            # 解码音频
            pcm_frame = self.decoder.decode(opus_packet, 960)
            self.pcm_chunk_buffer += pcm_frame
            self.last_audio_time = time.time()
            # 处理完整块
            while len(self.pcm_chunk_buffer) >= self.chunk_stride_bytes:
                chunk_data = self.pcm_chunk_buffer[:self.chunk_stride_bytes]
                self.pcm_chunk_buffer = self.pcm_chunk_buffer[self.chunk_stride_bytes:]
                audio_int16 = np.frombuffer(chunk_data, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                # 关键修改：强制启用增量模式
                res = self.model.generate(
                    input=audio_float32,
                    cache=self.cache,
                    is_final=False,
                    chunk_size=self.chunk_size,
                    encoder_chunk_look_back=self.encoder_chunk_look_back,
                    decoder_chunk_look_back=self.decoder_chunk_look_back,
                    disable_pbar=True,
                    hotword='',  # 确保不影响基础识别
                    enable_timestamp=False  # 简化输出
                )
                # 解析结果
                current_text = self._parse_asr_result(res)
                if not current_text:
                    continue
                # 智能合并策略（增强版）
                self._update_results(current_text)
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理音频块出错: {e}", exc_info=True)

    def _parse_asr_result(self, res):
        """统一解析ASR结果"""
        result_list = res[0] if isinstance(res, tuple) else res
        if not result_list or not result_list[0].get("text"):
            return ""

        text = result_list[0]["text"].strip()
        # 记录原始结果用于调试
        self.result_history.append(text)
        logger.bind(tag=TAG).debug(f"模型原始输出: {text}")
        return rich_transcription_postprocess(text)

    def _update_results(self, current_text):
        """智能更新结果(终极版)"""
        # 情况1: 全新开始
        if not self.full_result_cache:
            self.full_result_cache = current_text
            self.last_partial_result = current_text
            logger.bind(tag=TAG).info(f"初始结果: {self.full_result_cache}")
            return
        # 情况2: 当前结果是上次结果的延续
        if current_text.startswith(self.last_partial_result):
            new_part = current_text[len(self.last_partial_result):]
            self.full_result_cache += new_part
            self.last_partial_result = current_text
            logger.bind(tag=TAG).info(f"追加结果: +{new_part} → {self.full_result_cache}")
            return
        # 情况3: 完全不同的结果（可能是修正）
        overlap = self._find_max_overlap(self.full_result_cache, current_text)
        if overlap:
            self.full_result_cache = self.full_result_cache[:overlap] + current_text
            logger.bind(tag=TAG).info(f"修正结果: {overlap}字重叠 → {self.full_result_cache}")
        else:
            self.full_result_cache += " " + current_text
            logger.bind(tag=TAG).info(f"新增结果: {current_text} → {self.full_result_cache}")

        self.last_partial_result = current_text

    def _find_max_overlap(self, base_text, new_text):
        """查找最大重叠部分"""
        max_len = min(len(base_text), len(new_text), 10)  # 限制最大检查长度
        for i in range(max_len, 0, -1):
            if base_text.endswith(new_text[:i]):
                return i
        return 0

    async def finalize_recognition(self) -> str:
        """最终处理(终极版)"""
        try:
            # 处理剩余音频
            if len(self.pcm_chunk_buffer) > 0:
                audio_int16 = np.frombuffer(self.pcm_chunk_buffer, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                res = self.model.generate(
                    input=audio_float32,
                    cache=self.cache,
                    is_final=True,
                    chunk_size=self.chunk_size,
                    encoder_chunk_look_back=self.encoder_chunk_look_back,
                    decoder_chunk_look_back=self.decoder_chunk_look_back,
                )
                final_part = self._parse_asr_result(res)
                if final_part:
                    self._update_results(final_part)
            # 返回累积结果
            final_text = self.full_result_cache.strip()
            logger.bind(tag=TAG).info(f"最终结果: {final_text} (历史: {self.result_history})")
            return final_text if final_text else ""
        except Exception as e:
            logger.bind(tag=TAG).error(f"最终处理出错: {e}", exc_info=True)
            return self.full_result_cache if self.full_result_cache else ""
        finally:
            self._reset_realtime_state()

    def _reset_realtime_state(self):
        """重置状态"""
        self.cache = {}
        self.pcm_chunk_buffer = b''
        self.full_result_cache = ""
        self.last_partial_result = ""
        self.result_history = []

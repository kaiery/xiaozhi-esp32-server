import time
import wave
import os
import sys
import io

import numpy as np

from config.logger import setup_logging
from typing import Optional, Tuple, List, Any, Coroutine
import uuid
import opuslib_next
from core.providers.asr.base import ASRProviderBase

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

TAG = __name__
logger = setup_logging()


# 捕获标准输出 (用于抑制 FunASR 初始化时的打印信息)
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
        if self.output and self.output.strip():  # 优化：仅在有实际输出时记录
            logger.bind(tag=TAG).info(f"FunASR 输出: {self.output.strip()}")


class ASRProvider(ASRProviderBase):
    """
    使用 FunASR Paraformer Online 模型实现实时语音识别的提供者类。
    核心逻辑是累积处理音频块，并拼接模型返回的中间文本片段。
    """

    def __init__(self, config: dict, delete_audio_file: bool):
        """
        初始化 ASR 提供者。

        Args:
            config (dict): 包含模型配置的字典。
            delete_audio_file (bool): 是否删除非实时模式下生成的临时 WAV 文件。
        """
        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_dir")
        self.delete_audio_file = delete_audio_file
        # 从配置读取 use_realtime_asr
        self._use_realtime_asr = config.get("use_realtime_asr", False)

        # 实时语音识别相关参数 (从你的示例代码获取)
        # 这些参数影响模型处理音频块的方式和上下文依赖长度
        self.chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
        self.encoder_chunk_look_back = 4  # 编码器回看块数
        self.decoder_chunk_look_back = 1  # 解码器回看块数
        # logger.bind(tag=TAG).info(f"实时 ASR 参数: chunk_size={self.chunk_size}, encoder_look_back={self.encoder_chunk_look_back}, decoder_look_back={self.decoder_chunk_look_back}")

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # logger.bind(tag=TAG).info(f"Initializing FunASR AutoModel for {'Realtime' if self._use_realtime_asr else 'Non-Realtime'} ASR...")
        # 初始化模型 (不需要 streaming=True 参数, 根据你的示例代码)
        with CaptureOutput():
            self.model = AutoModel(
                model=self.model_dir,  # 模型目录或 HuggingFace ID
                disable_update=True,  # 禁止自动更新模型
                hub="hf",  # 指定模型来源 (HuggingFace)
                device='cpu'  # 或者根据你的环境配置 'cuda:0'
            )
        logger.bind(tag=TAG).info(f"FunASR AutoModel 初始化完成。")

        # --- 实时处理所需的状态 ---
        self.decoder = opuslib_next.Decoder(16000, 1)  # Opus 解码器 (16kHz, 单声道)
        self.pcm_chunk_buffer = b''  # 存储解码后的 PCM 数据，用于凑齐一个处理块
        self.cache = {}  # FunASR 流式识别的核心状态缓存 (原地修改)
        # 计算一个处理块对应的采样点数和字节数 (16kHz, 16-bit PCM)
        self.chunk_stride_samples = self.chunk_size[1] * 960  # 10ms * 16kHz = 160 samples per frame
        self.chunk_stride_bytes = self.chunk_stride_samples * 2  # 16-bit PCM = 2 bytes/sample

        # --- 结果累积和调试变量 ---
        self.full_result_cache = ""  # **核心变量**: 用于累积拼接识别结果
        self.result_history = []  # 记录模型每次返回的原始文本，用于调试

    # --- 实现基类新增的属性 ---
    @property
    def use_realtime_asr(self) -> bool:
        return self._use_realtime_asr  # 返回内部状态

    def save_audio_to_file(self, opus_data: List[bytes], session_id: str) -> str:
        """将 Opus 音频数据解码并保存为 WAV 文件（主要用于非实时模式或调试）。"""
        file_name = f"asr_{session_id}_{uuid.uuid4()}.wav"
        file_path = os.path.join(self.output_dir, file_name)

        # 使用独立的解码器，避免影响流式解码器的状态
        decoder = opuslib_next.Decoder(16000, 1)  # 16kHz, 单声道
        pcm_data = []

        for i, opus_packet in enumerate(opus_data):
            try:
                # 假设每个 Opus 包是 60ms (960 samples)
                pcm_frame = decoder.decode(opus_packet, 960)
                pcm_data.append(pcm_frame)
            except opuslib_next.OpusError as e:
                logger.bind(tag=TAG).error(f"Opus解码错误: {e}", exc_info=True)
            except Exception as e:
                logger.bind(tag=TAG).error(f"解码时发生未知错误 (包 {i + 1}/{len(opus_data)}): {e}")

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 2 bytes = 16位 PCM
            wf.setframerate(16000)  # 16kHz 采样率
            wf.writeframes(b"".join(pcm_data))

        return file_path

    async def speech_to_text(self, opus_data: List[bytes], session_id: str):
        """
        【非实时】语音转文本处理逻辑。
        处理完整的 Opus 数据列表，将其保存为 WAV 文件后进行识别。
        """
        if self._use_realtime_asr:
            logger.bind(tag=TAG).warning("在实时模式下调用了 speech_to_text。将执行非流式识别作为后备。")
            # 注意：这里执行的是非流式识别，即使配置了实时模式。
            # 真正的实时处理是通过 process_audio_chunk 和 finalize_recognition 实现的。

        file_path = None
        try:
            # 保存音频文件
            file_path = self.save_audio_to_file(opus_data, session_id)
            if not file_path:  # 如果保存文件失败
                logger.bind(tag=TAG).error("[非实时后备] 音频文件保存失败，无法进行识别。")
                return "", None

            # 使用FunASR模型进行【非流式】语音识别
            start_time = time.time()
            # 非流式调用不需要 cache 和 chunk 参数
            result = self.model.generate(
                input=file_path,
                cache={},  # 非流式调用 cache 为空
                language="auto",  # 自动语言检测 (如果模型支持)
                use_itn=True,  # 启用逆文本标准化 (数字转阿拉伯，标点恢复)
                batch_size_s=60,  # 处理批次大小 (秒)
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
                    logger.bind(tag=TAG).debug(f"[非实时后备] 已删除临时音频文件: {file_path}")
                except Exception as e:
                    logger.bind(tag=TAG).error(f"[非实时后备] 删除临时文件失败: {file_path} | 错误: {e}")

    # --- 实时处理核心方法 ---
    async def process_audio_chunk(self, opus_packet: bytes):
        """
        处理单个实时 Opus 音频包。
        解码 -> 缓冲 -> 调用模型处理完整块 -> 累积结果。
        """
        if not self._use_realtime_asr:  # 如果不是实时模式，则忽略
            # logger.bind(tag=TAG).warning("在非实时模式下调用了 process_audio_chunk。")
            return
        try:
            # 1. 解码 Opus 包到 PCM 数据
            pcm_frame = self.decoder.decode(opus_packet, 960)  # 假设 60ms 包
            self.pcm_chunk_buffer += pcm_frame  # 追加到缓冲区

            # 2. 循环处理缓冲区中所有完整的块
            while len(self.pcm_chunk_buffer) >= self.chunk_stride_bytes:
                # 提取一个处理块的数据
                chunk_data = self.pcm_chunk_buffer[:self.chunk_stride_bytes]
                # 更新缓冲区，移除已提取的数据
                self.pcm_chunk_buffer = self.pcm_chunk_buffer[self.chunk_stride_bytes:]

                # 将 bytes 转换为 float32 NumPy 数组
                audio_int16 = np.frombuffer(chunk_data, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0  # 标准化到 [-1, 1]

                # 3. 调用模型进行流式推理 (is_final=False)
                # 假设 cache 会被原地修改
                try:
                    res = self.model.generate(
                        input=audio_float32,
                        cache=self.cache,  # 传入当前 cache
                        is_final=False,  # 明确告知模型这是中间块
                        chunk_size=self.chunk_size,  # 传递流式参数
                        encoder_chunk_look_back=self.encoder_chunk_look_back,
                        decoder_chunk_look_back=self.decoder_chunk_look_back,
                        disable_pbar=True,  # 禁用进度条打印
                        hotword='',  # 禁用热词 (除非你需要)
                        enable_timestamp=False  # 禁用时间戳 (简化输出)
                    )
                except Exception as model_e:
                    logger.bind(tag=TAG).error(f"模型推理错误 (中间块): {model_e}", exc_info=True)
                    continue  # 跳过这个块的处理，继续下一个循环

                # 4. 解析模型返回的文本片段
                current_text = self._parse_asr_result(res)
                if not current_text:  # 如果模型未返回有效文本，跳过
                    continue

                # 5. 更新累积的识别结果
                self._update_results(current_text)

        except opuslib_next.OpusError as decode_e:
            logger.bind(tag=TAG).error(f"Opus 解码错误 in process_audio_chunk: {decode_e}", exc_info=True)
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理音频块时发生未知错误: {e}", exc_info=True)

    def _parse_asr_result(self, res) -> str:
        """
        从模型响应中解析出文本结果。

        Args:
            res: model.generate 的返回值。

        Returns:
            处理后的文本字符串，如果无效则返回空字符串。
        """
        result_list = []
        # 兼容可能的返回格式：元组 (result,) 或列表 [result]
        if isinstance(res, tuple) and len(res) >= 1:
            result_list = res[0] if isinstance(res[0], list) else []
        elif isinstance(res, list):
            result_list = res

        # 安全地提取文本
        raw_text = ""
        if result_list and isinstance(result_list[0], dict) and result_list[0].get("text"):
            raw_text = result_list[0]["text"]
        else:
            logger.bind(tag=TAG).debug("模型未返回文本或格式无效。")
            return ""  # 无有效文本

        # 记录原始结果（去除首尾空格）用于调试
        raw_text_stripped = raw_text.strip()
        self.result_history.append(raw_text_stripped)
        logger.bind(tag=TAG).debug(f"模型原始输出 (处理前): '{raw_text_stripped}'")

        if not raw_text_stripped:  # 如果去除空格后为空，也视为无效
            return ""

        # 应用 ITN 和标点恢复等后处理
        processed_text = rich_transcription_postprocess(raw_text_stripped)
        logger.bind(tag=TAG).debug(f"模型输出 (处理后): '{processed_text}'")

        return processed_text

    def _update_results(self, current_text):
        """智能更新结果(终极优化版)"""
        if not current_text:
            return
        # 记录原始输出
        self.result_history.append(current_text)

        # 首次识别
        if not self.full_result_cache:
            self.full_result_cache = current_text
            logger.bind(tag=TAG).info(f"初始结果: {self.full_result_cache}")
            return

        # 极简合并：直接追加（不添加任何标点）
        self.full_result_cache += current_text
        logger.bind(tag=TAG).info(f"直接合并: {current_text} → {self.full_result_cache}")

    async def finalize_recognition(self) -> str:
        """最终处理(极简版)"""
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
                    self.full_result_cache += final_part
            # 返回原始合并结果
            final_text = self.full_result_cache
            # logger.bind(tag=TAG).info(f"最终结果: {final_text} (历史: {self.result_history})")
            return final_text if final_text else ""
        except Exception as e:
            logger.bind(tag=TAG).error(f"最终处理出错: {e}")
            return self.full_result_cache if self.full_result_cache else ""
        finally:
            self._reset_realtime_state()

    def _reset_realtime_state(self):
        """重置状态"""
        self.cache = {}
        self.pcm_chunk_buffer = b''
        self.full_result_cache = ""
        self.result_history = []

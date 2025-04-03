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
        """处理单个实时音频块"""
        if not self._use_realtime_asr:
            logger.bind(tag=TAG).warning("process_audio_chunk called when not in realtime mode. Ignoring.")
            return

        try:
            pcm_frame = self.decoder.decode(opus_packet, 960)  # 960 samples = 60ms
            self.pcm_chunk_buffer += pcm_frame
            # logger.bind(tag=TAG).debug(f"Received chunk, buffer size: {len(self.pcm_chunk_buffer)} bytes")

            # 检查缓冲区是否有足够数据进行一次模型推理
            while len(self.pcm_chunk_buffer) >= self.chunk_stride_bytes:
                # 提取一个 chunk 的数据
                chunk_data = self.pcm_chunk_buffer[:self.chunk_stride_bytes]
                # 更新缓冲区，移除已提取的数据
                self.pcm_chunk_buffer = self.pcm_chunk_buffer[self.chunk_stride_bytes:]

                # 将 bytes 转换为 numpy array (int16 -> float32)
                # FunASR 模型期望输入是 float32 NumPy array
                audio_int16 = np.frombuffer(chunk_data, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0  # 标准化到 [-1, 1]

                # 调用模型进行流式推理
                try:
                    start_time = time.time()
                    res = self.model.generate(
                        input=audio_float32,
                        cache=self.cache,  # 传入当前的 cache
                        is_final=False,  # 重要：告知模型这是中间块
                        chunk_size=self.chunk_size,
                        encoder_chunk_look_back=self.encoder_chunk_look_back,
                        decoder_chunk_look_back=self.decoder_chunk_look_back,
                        # language="auto" # online 模型通常不需要指定 language
                    )
                    # logger.bind(tag=TAG).debug(f"Model generate (intermediate) took: {time.time() - start_time:.4f}s")

                    # 更新 cache, generate 会返回更新后的 cache
                    if isinstance(res, tuple) and len(res) == 2:  # 老版本 funasr 可能返回 (result, cache)
                        result_list = res[0]
                        self.cache = res[1]
                    elif isinstance(res, list) and len(res) > 0 and 'cache' in res[0]:  # 新版本 funasr 可能在结果字典中包含 cache
                        result_list = res
                        self.cache = res[0].get('cache', self.cache)  # 更新缓存
                    else:  # 兼容其他可能的返回格式
                        result_list = res if isinstance(res, list) else []
                        # Cache 可能在generate调用中被内部更新了，但这里无法显式获取，依赖模型内部实现

                    # 处理中间结果 (可选，主要用于调试或显示实时字幕)
                    if result_list and result_list[0].get("text"):
                        intermediate_text = rich_transcription_postprocess(result_list[0]["text"])
                        if intermediate_text != self.last_intermediate_result:
                            logger.bind(tag=TAG).debug(f"实时识别中间结果: {intermediate_text}")
                            self.last_intermediate_result = intermediate_text  # 更新最后结果，避免日志刷屏
                        # 注意：这里获取的可能是片段，或者是基于当前片段更新后的完整句预测
                        # 不将中间结果传递给 startToChat

                except Exception as model_e:
                    logger.bind(tag=TAG).error(f"实时模型推理错误: {model_e}", exc_info=True)
                    # 这里可以选择重置 cache 或者继续尝试
                    # self.cache = {} # 如果模型出错，可能需要重置cache

        except opuslib_next.OpusError as decode_e:
            logger.bind(tag=TAG).error(f"Opus解码错误 in process_audio_chunk: {decode_e}", exc_info=True)
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理音频块时发生未知错误: {e}", exc_info=True)

    async def finalize_recognition(self) -> Optional[str]:
        """结束识别，获取最终结果并重置状态"""
        if not self._use_realtime_asr:
            logger.bind(tag=TAG).warning("finalize_recognition called when not in realtime mode. Returning None.")
            return None

        final_text = ""
        try:
            logger.bind(tag=TAG).debug(
                f"Finalizing recognition. Remaining buffer size: {len(self.pcm_chunk_buffer)} bytes")
            # 处理缓冲区中剩余的不足一个 chunk_stride 的数据
            if len(self.pcm_chunk_buffer) > 0:
                # 将剩余 bytes 转换为 numpy array
                audio_int16 = np.frombuffer(self.pcm_chunk_buffer, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0

                # 调用模型，标记为 is_final=True
                start_time = time.time()
                res = self.model.generate(
                    input=audio_float32,
                    cache=self.cache,
                    is_final=True,  # 重要：告知模型这是最后一块
                    chunk_size=self.chunk_size,
                    encoder_chunk_look_back=self.encoder_chunk_look_back,
                    decoder_chunk_look_back=self.decoder_chunk_look_back,
                )
                # logger.bind(tag=TAG).debug(f"Model generate (final) took: {time.time() - start_time:.4f}s")

                if isinstance(res, tuple) and len(res) == 2:
                    result_list = res[0]
                elif isinstance(res, list):
                    result_list = res
                else:
                    result_list = []

                if result_list and result_list[0].get("text"):
                    final_text = rich_transcription_postprocess(result_list[0]["text"])
                    logger.bind(tag=TAG).info(f"实时识别最终结果: {final_text}")
                else:
                    # 如果最后一块没有识别结果，可能需要结合之前的中间结果，
                    # 但通常 is_final=True 会给出完整的句子。
                    # 如果最后确实没识别出东西，就用之前的记录（如果需要的话）
                    # final_text = self.last_intermediate_result # 这是一个选择
                    logger.bind(tag=TAG).info("实时识别最终块未返回文本。")
                    # 尝试使用 self.last_intermediate_result 作为最终结果，如果它有意义
                    if self.last_intermediate_result:
                        final_text = self.last_intermediate_result
                        logger.bind(tag=TAG).info(f"使用最后的中间结果作为最终结果: {final_text}")


            else:
                # 如果缓冲区为空，说明最后一次 process_audio_chunk 可能已经得到了接近最终的结果
                # 但仍然需要调用 is_final=True 来确保模型状态正确结束并获取最终确认的文本
                logger.bind(tag=TAG).debug(
                    "Buffer is empty, calling generate with is_final=True on empty input to finalize.")
                start_time = time.time()
                # FunASR 可能需要一个空的输入来触发最后的处理
                empty_input = np.zeros(0, dtype=np.float32)
                res = self.model.generate(
                    input=empty_input,
                    cache=self.cache,
                    is_final=True,
                    chunk_size=self.chunk_size,
                    encoder_chunk_look_back=self.encoder_chunk_look_back,
                    decoder_chunk_look_back=self.decoder_chunk_look_back,
                )
                # logger.bind(tag=TAG).debug(f"Model generate (final, empty input) took: {time.time() - start_time:.4f}s")

                if isinstance(res, tuple) and len(res) == 2:
                    result_list = res[0]
                elif isinstance(res, list):
                    result_list = res
                else:
                    result_list = []

                if result_list and result_list[0].get("text"):
                    final_text = rich_transcription_postprocess(result_list[0]["text"])
                    logger.bind(tag=TAG).info(f"实时识别最终结果 (from empty final call): {final_text}")
                else:
                    logger.bind(tag=TAG).info("实时识别最终确认调用未返回文本。")
                    # 同样，考虑使用最后的中间结果
                    if self.last_intermediate_result:
                        final_text = self.last_intermediate_result
                        logger.bind(tag=TAG).info(f"使用最后的中间结果作为最终结果: {final_text}")


        except Exception as e:
            logger.bind(tag=TAG).error(f"结束实时识别时发生错误: {e}", exc_info=True)
            final_text = ""  # 出错则返回空

        finally:
            # --- 重置实时状态，为下一句话做准备 ---
            logger.bind(tag=TAG).debug("Resetting realtime ASR state (cache, buffer).")
            self.cache = {}
            self.pcm_chunk_buffer = b''
            self.last_intermediate_result = ""
            # self.decoder 不需要重置，除非opus流格式变化

        return final_text if final_text else ""  # 确保至少返回空字符串

# --- END OF FILE core/providers/asr/fun_realtime.py ---

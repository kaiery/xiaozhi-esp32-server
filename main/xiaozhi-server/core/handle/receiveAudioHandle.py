from config.logger import setup_logging
import time
from core.utils.util import remove_punctuation_and_length
from core.handle.sendAudioHandle import send_stt_message
from core.handle.intentHandler import handle_user_intent

TAG = __name__
logger = setup_logging()


async def handleAudioMessage(conn, audio):
    if not conn.asr_server_receive:
        logger.bind(tag=TAG).debug(f"前期数据处理中，暂停接收")
        return
    if conn.client_listen_mode == "auto":
        have_voice = conn.vad.is_vad(conn, audio)
    else:
        have_voice = conn.client_have_voice

    # 检查是否启用了实时ASR
    is_realtime_asr = conn.asr.use_realtime_asr
    # --- 实时ASR处理逻辑 ---
    if is_realtime_asr:
        # 只要有声音就持续处理音频块
        if have_voice or len(conn.asr_audio) > 0:  # 如果VAD刚检测到无声音，但缓冲区还有数据，也需要处理
            conn.client_no_voice_last_time = 0.0  # 重置静默计时器
            conn.asr_audio.append(audio)  # 仍然暂存原始opus包，可能用于调试或特殊情况
            try:
                # 将音频块传递给ASR进行实时处理
                await conn.asr.process_audio_chunk(audio)
            except Exception as e:
                logger.bind(tag=TAG).error(f"实时处理音频块失败: {e}", exc_info=True)

        # 当检测到语音停止时，触发最终识别
        if conn.client_voice_stop:
            conn.client_abort = False  # 重置打断标记
            conn.asr_server_receive = False  # 暂停接收新音频，直到LLM响应回来
            # logger.bind(tag=TAG).info("检测到语音停止，开始获取最终识别结果...")
            try:
                # 调用 finalize 获取最终文本
                final_text = await conn.asr.finalize_recognition()
                logger.bind(tag=TAG).info(f"最终识别文本: {final_text}")
            except Exception as e:
                logger.bind(tag=TAG).error(f"获取最终识别结果失败: {e}", exc_info=True)
                final_text = ""  # 出错则为空

            # 清理本次对话的原始opus包缓存
            conn.asr_audio.clear()
            conn.reset_vad_states()  # 重置VAD状态

            # 检查识别结果是否有效
            text_len, _ = remove_punctuation_and_length(final_text)
            if text_len > 0:
                await startToChat(conn, final_text)  # 发起聊天
            else:
                # logger.bind(tag=TAG).info("最终识别结果为空或无效，不触发聊天。")
                conn.asr_server_receive = True  # 没有有效文本，重新允许接收音频
        # 如果没有声音，并且没有积攒的音频需要处理（asr_audio为空），并且不是刚停止说话，则检查超时
        elif not have_voice and len(conn.asr_audio) == 0:
            await no_voice_close_connect(conn)

    else:
        # 如果本次没有声音，本段也没声音，就把声音丢弃了
        if have_voice == False and conn.client_have_voice == False:
            await no_voice_close_connect(conn)
            conn.asr_audio.append(audio)
            conn.asr_audio = conn.asr_audio[
                -10:
            ]  # 保留最新的10帧音频内容，解决ASR句首丢字问题
            return
        conn.client_no_voice_last_time = 0.0
        conn.asr_audio.append(audio)
        # 如果本段有声音，且已经停止了
        if conn.client_voice_stop:
            conn.client_abort = False
            conn.asr_server_receive = False
            # 音频太短了，无法识别
            if len(conn.asr_audio) < 15:
                logger.bind(tag=TAG).info(f"非实时模式：音频长度过短 ({len(conn.asr_audio)} frames)，忽略。")
                conn.asr_server_receive = True
            else:
                logger.bind(tag=TAG).info(f"非实时模式：检测到语音停止，处理累积的 {len(conn.asr_audio)} 个音频包...")
                # 调用原来的 speech_to_text 处理整个列表
                text, file_path = await conn.asr.speech_to_text(
                    conn.asr_audio, conn.session_id
                )
                logger.bind(tag=TAG).info(f"识别文本: {text}")
                text_len, _ = remove_punctuation_and_length(text)
                if text_len > 0:
                    await startToChat(conn, text)
                else:
                    logger.bind(tag=TAG).info("非实时识别结果为空或无效，不触发聊天。")
                    conn.asr_server_receive = True  # 没有有效文本，重新允许接收音频
            # 清理累积的音频和VAD状态
            conn.asr_audio.clear()
            conn.reset_vad_states()


async def startToChat(conn, text):
    # 首先进行意图分析
    intent_handled = await handle_user_intent(conn, text)

    if intent_handled:
        # 如果意图已被处理，不再进行聊天
        conn.asr_server_receive = True
        return

    # 意图未被处理，继续常规聊天流程
    await send_stt_message(conn, text)
    if conn.use_function_call_mode:
        # 使用支持function calling的聊天方法
        conn.executor.submit(conn.chat_with_function_calling, text)
    else:
        conn.executor.submit(conn.chat, text)


async def no_voice_close_connect(conn):
    if conn.client_no_voice_last_time == 0.0:
        conn.client_no_voice_last_time = time.time() * 1000
    else:
        no_voice_time = time.time() * 1000 - conn.client_no_voice_last_time
        close_connection_no_voice_time = conn.config.get(
            "close_connection_no_voice_time", 120
        )
        if (
            not conn.close_after_chat
            and no_voice_time > 1000 * close_connection_no_voice_time
        ):
            conn.close_after_chat = True
            conn.client_abort = False
            conn.asr_server_receive = False
            prompt = (
                "请你以“时间过得真快”为开头，用富有感情、依依不舍的话来结束这场对话吧。"
            )
            await startToChat(conn, prompt)

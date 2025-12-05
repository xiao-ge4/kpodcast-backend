from typing import List, Dict, Any, Tuple, Optional
import os
from utils.config_loader import load_ini
from clients.bocha_client import BochaClient
from clients.hunyuan_client import HunyuanClient
from clients.hunyuan_api_client import HunyuanAPIClient
from clients.tencent_tts import synthesize_tencent_tts
from utils.doc_loader import fetch_url
from utils.audio import ensure_dir, mix_intro_with_voice, export_with_intro
import re
from io import BytesIO
from pydub import AudioSegment
import base64


def retrieve_sources(cfg: Dict[str, Any], mode: str, query: str = "", url: str = "", doc_text: str = "") -> List[Dict[str, Any]]:
    if mode == "query":
        client = BochaClient(cfg["bocha_base_url"], cfg["bocha_api_id"], cfg["bocha_api_key"], cfg["bocha_search_path"])
        items = client.search(query, count=8)
        return items
    elif mode == "url":
        r = fetch_url(url)
        if r.get("success"):
            return [{"title": url, "url": url, "snippet": r.get("text", "")}]
        return []
    elif mode == "doc":
        return [{"title": "上传文档", "url": "", "snippet": doc_text}]
    return []


def build_outline_and_script(cfg: Dict[str, Any], topic: str, sources: List[Dict[str, Any]], style: str = "news") -> Dict[str, str]:
    # 使用云端混元（hunyuan-turbos-latest）
    api = HunyuanAPIClient(
        secret_id=cfg["hunyuan_api_secret_id"],
        secret_key=cfg["hunyuan_api_secret_key"],
        region=cfg["hunyuan_api_region"],
        model=cfg["hunyuan_api_model"],
        temperature=cfg["hunyuan_api_temperature"],
        top_p=cfg["hunyuan_api_top_p"],
        max_tokens=cfg["hunyuan_api_max_tokens"],
    )
    # 证据拼接 - 每条保留2000字符，平衡信息完整性和处理速度
    evidence = "\n\n".join([
        f"[{i+1}] 标题：{s.get('title','')}\n来源：{s.get('url','')}\n内容：{(s.get('snippet') or '')[:2000]}"
        for i, s in enumerate(sources)
    ])
    prompt = (
        f"你是资深中文播客编剧。任务：为话题《{topic}》创作高质量两人对话播客脚本。\n\n"
        
        f"【时长要求】\n"
        f"- 目标时长：8-15分钟（约2400-4500字）\n"
        f"- 语速参考：每分钟300字左右\n\n"
        
        f"【角色设定】\n"
        f"- 主播A（专家型）：知识渊博、逻辑清晰、善于深度分析，偶有专业术语但会解释，语气沉稳但不失亲和力\n"
        f"- 主播B（引导型）：好奇心强、善于提问、代表听众视角，会适时总结要点、提出疑问、调节气氛，语气活泼自然\n\n"
        
        f"【内容结构】（严格遵循）\n"
        f"1. 开场白（2-3轮）：热情欢迎+话题引入+为什么重要\n"
        f"2. 核心内容（6-10轮）：\n"
        f"   - 按逻辑层层递进（背景→现状→分析→影响）\n"
        f"   - 每个要点配合具体案例/数据[引用编号]\n"
        f"   - B适时提问、总结、过渡\n"
        f"3. 深度讨论（3-5轮）：争议点/多角度思考/未来展望\n"
        f"4. 结尾（2-3轮）：核心观点回顾+行动建议+互动号召\n\n"
        
        f"【对话风格】\n"
        f"- 自然口语化：使用'嗯'、'确实'、'你看'、'比如说'等口语词\n"
        f"- 情感节奏：适度停顿（用'...'表示）、语气词（'啊'、'呢'、'吧'）、情绪变化（惊讶/赞同/质疑）\n"
        f"- 互动真实：有打断、追问、玩笑、共鸣、不同观点的碰撞\n"
        f"- 避免说教：用故事化、场景化方式呈现，而非枯燥陈述\n\n"
        
        f"【事实依据】\n"
        f"- 所有关键事实、数据、观点必须来自下方证据，并标注[编号]\n"
        f"- 证据不足时，明确说明'目前研究显示...'、'有观点认为...'等限定表达\n"
        f"- 禁止编造数据和事实\n\n"
        
        f"【证据材料】\n{evidence}\n\n"
        
        f"【输出规范】\n"
        f"- 纯对话格式，每行一句，按行交替（A→B→A→B...）\n"
        f"- 不要任何标签（如'主播A：'、'旁白'等）\n"
        f"- 不要Markdown格式\n"
        f"- 总字数控制在2400-4500字\n"
        f"- 确保对话完整，有明确的开头和结尾\n"
    )
    messages = [
        {"Role": "system", "Content": "你是一个资深中文播客编剧，擅长创作自然流畅、信息丰富、情感真实的对话脚本"},
        {"Role": "user", "Content": prompt},
    ]
    resp = api.chat(messages, stream=False)
    # 兼容返回结构：
    content = ""
    try:
        choices = resp.get("Choices") or resp.get("choices") or []
        if choices:
            msg = choices[0].get("Message") or choices[0].get("message") or {}
            content = msg.get("Content") or msg.get("content") or ""
    except Exception:
        content = ""
    script = content or ""
    return {"script": script}


def _split_for_tts(text: str, limit: int = 240) -> List[str]:
    """将长文本切分为腾讯TTS可接受的短片段。limit 取一个安全阈值（中文字符数）。"""
    if not text:
        return []
    parts: List[str] = []
    # 先按段落
    for para in [p.strip() for p in text.splitlines() if p.strip()]:
        if len(para) <= limit:
            parts.append(para)
            continue
        # 再按句号等切分
        segs = re.split(r"([。！？；.!?])", para)
        buf = ""
        for i in range(0, len(segs), 2):
            s = segs[i]
            tail = segs[i+1] if i+1 < len(segs) else ""
            sent = (s + tail).strip()
            if not sent:
                continue
            if len(buf) + len(sent) <= limit:
                buf += sent
            else:
                if buf:
                    parts.append(buf)
                if len(sent) <= limit:
                    buf = sent
                else:
                    # 极长句，硬切
                    for j in range(0, len(sent), limit):
                        parts.append(sent[j:j+limit])
                    buf = ""
        if buf:
            parts.append(buf)
    return parts


def _sanitize_for_tts(text: str, aggressive: bool = False) -> str:
    """清洗文本以通过腾讯TTS校验：
    - 移除引用标记 [n]
    - 去除URL/邮箱
    - 去除不可见控制符与emoji
    - 规范标点与空白
    - aggressive=True 时，仅保留中英数字与常见标点
    """
    if not text:
        return ""
    t = text
    # 删除 [123] 引用
    t = re.sub(r"\s*\[[0-9]+\]\s*", "", t)
    # 删除URL/邮箱
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"\S+@\S+", "", t)
    # 移除控制字符
    t = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]", "", t)
    # 移除 emoji（代理项范围）
    t = re.sub(r"[\U00010000-\U0010FFFF]", "", t)
    # 统一破折号/省略号
    t = t.replace("——", "—").replace("…", "...")
    # 归一空白
    t = re.sub(r"\s+", " ", t).strip()
    if aggressive:
        t = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff，。！？；、：“”‘’\(\)\[\]\-~…\.!,?;: ]", "", t)
        t = re.sub(r"\s+", " ", t).strip()
    return t or "。"


def _parse_voice(v: Optional[str], default_num: str) -> str:
    # 输入形如 "501006:千嶂" -> 取冒号前的数字
    if not v:
        return default_num
    if ":" in v:
        return v.split(":",1)[0].strip()
    return v.strip()


def tts_and_mix(cfg: Dict[str, Any], script: str, intro_style: str = "serious", speed: int = 0,
                voice_a: Optional[str] = None, voice_b: Optional[str] = None) -> Tuple[str, str]:
    ensure_dir(cfg["output_dir"])
    # 分段合成，规避 TextTooLong
    chunks = _split_for_tts(script, limit=220)
    if not chunks:
        raise RuntimeError("脚本为空，无法合成TTS")
    segments: List[AudioSegment] = []
    fillers = ["嗯，我们继续。", "好的，接着说。", "下面进入下一段。"]
    vnum_a = _parse_voice(voice_a, cfg.get("voice_role_a", "501006"))
    vnum_b = _parse_voice(voice_b, cfg.get("voice_role_b", "601007"))
    for idx, ch in enumerate(chunks):
        text1 = _sanitize_for_tts(ch, aggressive=False)
        # 交替角色（简单规则：奇偶行切换），实际可按标注段落区分 A/B
        use_voice = vnum_a if (idx % 2 == 0) else vnum_b
        sec = synthesize_tencent_tts(
            text1,
            secret_id=cfg["tencent_secret_id"],
            secret_key=cfg["tencent_secret_key"],
            region=cfg["tencent_region"],
            voice=use_voice,
            speed=speed,
            codec="mp3",
        )
        if (not sec.get("success") or not sec.get("bytes")) and "InvalidText" in str(sec.get("error", "")):
            # 尝试更激进清洗后重试
            text2 = _sanitize_for_tts(text1, aggressive=True)
            sec = synthesize_tencent_tts(
                text2,
                secret_id=cfg["tencent_secret_id"],
                secret_key=cfg["tencent_secret_key"],
                region=cfg["tencent_region"],
                voice=use_voice,
                speed=speed,
                codec="mp3",
            )
        if (not sec.get("success") or not sec.get("bytes")) and "InvalidText" in str(sec.get("error", "")):
            # 再失败则用兜底占位短句
            safe = fillers[idx % len(fillers)]
            sec = synthesize_tencent_tts(
                safe,
                secret_id=cfg["tencent_secret_id"],
                secret_key=cfg["tencent_secret_key"],
                region=cfg["tencent_region"],
                voice=use_voice,
                speed=speed,
                codec="mp3",
            )
        if not sec.get("success") or not sec.get("bytes"):
            raise RuntimeError(f"TTS失败: {sec.get('error')}")
        seg = AudioSegment.from_file(BytesIO(sec["bytes"]), format="mp3")
        segments.append(seg)
    # 拼接音频，段间留短暂停顿
    final_audio = AudioSegment.silent(duration=100)
    pause = AudioSegment.silent(duration=200)
    for seg in segments:
        final_audio = final_audio.append(seg, crossfade=50).append(pause, crossfade=0)
    voice_path = os.path.join(cfg["output_dir"], "podcast_voice.mp3")
    final_audio.export(voice_path, format="mp3", bitrate="192k")

    bgm_map = {
        "history": cfg["bgm_history"],
        "entertainment": cfg["bgm_entertainment"],
        "serious": cfg["bgm_serious"],
    }
    intro_file = os.path.join(cfg["assets_bgm_dir"], bgm_map.get(intro_style, cfg["bgm_serious"]))
    out_mp3 = os.path.join(cfg["output_dir"], "podcast_final.mp3")
    # 合成片头
    try:
        export_with_intro(final_audio, out_mp3, intro_path=intro_file if os.path.exists(intro_file) else None)
    except Exception:
        mix_intro_with_voice(intro_file if os.path.exists(intro_file) else None, voice_path, out_mp3)
    transcript_path = os.path.join(cfg["output_dir"], "podcast_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(script)
    return out_mp3, transcript_path


def run_end_to_end(mode: str, topic_or_url_or_text: str, style: str = "news", intro_style: str = "serious", speed: int = 0,
                   voice_a: Optional[str] = None, voice_b: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_ini()
    if cfg.get("tts_provider") != "tencent":
        raise RuntimeError("当前仅启用腾讯云 TTS，请在配置中设置 [tts] provider = tencent")
    if not (cfg.get("tencent_secret_id") and cfg.get("tencent_secret_key")):
        raise RuntimeError("请在 Source_config_podcast.ini 配置腾讯云 TTS 密钥")
    if mode == "query":
        topic = topic_or_url_or_text
        sources = retrieve_sources(cfg, "query", query=topic)
    elif mode == "url":
        topic = topic_or_url_or_text
        sources = retrieve_sources(cfg, "url", url=topic)
    else:
        topic = "用户上传文档"
        sources = retrieve_sources(cfg, "doc", doc_text=topic_or_url_or_text)
    script_res = build_outline_and_script(cfg, topic, sources, style=style)
    audio_path, transcript_path = tts_and_mix(cfg, script_res["script"], intro_style=intro_style, speed=speed,
                                              voice_a=voice_a, voice_b=voice_b)
    return {
        "audio_path": audio_path,
        "transcript_path": transcript_path,
        "sources": sources,
        "script": script_res["script"],
    }


def generate_stream(mode: str, topic_or_url_or_text: str, style: str = "news", intro_style: str = "serious", speed: int = 0,
                    voice_a: Optional[str] = None, voice_b: Optional[str] = None):
    """快方案：逐段TTS边生成边返回。
    Yields:
        {"type":"chunk", "index": i, "path": seg_path, "text": chunk_text, "transcript": so_far}
        最后：{"type":"done", "final_audio": final_path, "transcript": full_text}
    """
    cfg = load_ini()
    if cfg.get("tts_provider") != "tencent":
        yield {"type": "error", "error": "当前仅启用腾讯云 TTS"}
        return
    # 规范化输入类型，避免大小写导致走到文档分支
    mode_norm = (mode or "").strip().lower()
    # 1) 取来源
    if mode_norm == "query":
        topic = topic_or_url_or_text
        sources = retrieve_sources(cfg, "query", query=topic)
    elif mode_norm == "url":
        topic = topic_or_url_or_text
        sources = retrieve_sources(cfg, "url", url=topic)
    else:
        topic = "用户上传文档"
        sources = retrieve_sources(cfg, "doc", doc_text=topic_or_url_or_text)
    # 2) 生成脚本
    script_res = build_outline_and_script(cfg, topic, sources, style=style)
    script = script_res.get("script") or ""
    ensure_dir(cfg["output_dir"])
    chunks_dir = os.path.join(cfg["output_dir"], "chunks")
    ensure_dir(chunks_dir)
    # 3) 切分并合成
    vnum_a = _parse_voice(voice_a, cfg.get("voice_role_a", "501006"))
    vnum_b = _parse_voice(voice_b, cfg.get("voice_role_b", "601007"))
    pairs: List[Tuple[str, str]] = []
    # 尝试按行交替，否则退回句切
    lines = [ln.strip() for ln in (script or "").splitlines()]
    if any(lines):
        idx_line = 0
        for ln in lines:
            if not ln:
                continue
            clean_ln = _sanitize_for_tts(re.sub(r"^[*#\\s]+", "", re.sub(r"^主播[AB]\s*[：:]\s*", "", ln)))
            raw_ln = ln  # 原始带引用行
            if not clean_ln:
                continue
            voice = vnum_a if (idx_line % 2 == 0) else vnum_b
            if len(clean_ln) <= 220:
                pairs.append((clean_ln, voice, raw_ln))
            else:
                for s in _split_for_tts(clean_ln, limit=220):
                    if s:
                        pairs.append((_sanitize_for_tts(s), voice, raw_ln))  # raw_ln 重复无妨
            idx_line += 1
    if not pairs:
        # 回退：整段句切并交替音色
        chunks = _split_for_tts(script, limit=220)
        for i, ch in enumerate(chunks):
            voice = vnum_a if (i % 2 == 0) else vnum_b
            pairs.append((_sanitize_for_tts(ch), voice))
    transcript_so_far = ""
    final_segments: List[AudioSegment] = []
    fillers = ["嗯，我们继续。", "好的，接着说。", "下面进入下一段。"]
    for idx, (text, use_voice) in enumerate(pairs):
        # 合成
        sec = synthesize_tencent_tts(
            text,
            secret_id=cfg["tencent_secret_id"],
            secret_key=cfg["tencent_secret_key"],
            region=cfg["tencent_region"],
            voice=use_voice,
            speed=speed,
            codec="mp3",
        )
        if (not sec.get("success") or not sec.get("bytes")) and "InvalidText" in str(sec.get("error", "")):
            safe = fillers[idx % len(fillers)]
            sec = synthesize_tencent_tts(
                safe,
                secret_id=cfg["tencent_secret_id"],
                secret_key=cfg["tencent_secret_key"],
                region=cfg["tencent_region"],
                voice=use_voice,
                speed=speed,
                codec="mp3",
            )
        if not sec.get("success") or not sec.get("bytes"):
            yield {"type": "error", "error": f"TTS失败: {sec.get('error')}"}
            return
        audio_bytes = sec["bytes"]
        seg = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
        final_segments.append(seg)
        # 保存分段文件
        seg_path = os.path.join(chunks_dir, f"seg_{idx+1:04d}.mp3")
        seg.export(seg_path, format="mp3", bitrate="192k")
        data_url = "data:audio/mpeg;base64," + base64.b64encode(audio_bytes).decode("ascii")
        transcript_so_far += (text + "\n")
        yield {"type": "chunk", "index": idx+1, "path": seg_path, "data_url": data_url, "text": text, "transcript": transcript_so_far, "sources": sources}
    # 结束后导出最终文件
    final_audio = AudioSegment.silent(duration=100)
    pause = AudioSegment.silent(duration=200)
    for seg in final_segments:
        final_audio = final_audio.append(seg, crossfade=50).append(pause, crossfade=0)
    voice_path = os.path.join(cfg["output_dir"], "podcast_voice.mp3")
    final_audio.export(voice_path, format="mp3", bitrate="192k")
    bgm_map = {
        "history": cfg["bgm_history"],
        "entertainment": cfg["bgm_entertainment"],
        "serious": cfg["bgm_serious"],
    }
    intro_file = os.path.join(cfg["assets_bgm_dir"], bgm_map.get(intro_style, cfg["bgm_serious"]))
    out_mp3 = os.path.join(cfg["output_dir"], "podcast_final.mp3")
    try:
        export_with_intro(final_audio, out_mp3, intro_path=intro_file if os.path.exists(intro_file) else None)
    except Exception:
        mix_intro_with_voice(intro_file if os.path.exists(intro_file) else None, voice_path, out_mp3)
    yield {"type": "done", "final_audio": out_mp3, "transcript": transcript_so_far, "sources": sources}


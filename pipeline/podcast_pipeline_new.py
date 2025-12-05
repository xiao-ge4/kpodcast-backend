from typing import List, Dict, Any, Tuple, Optional
import os
import logging
from utils.config_loader import load_ini
from clients.bocha_client import BochaClient
# 移除对hunyuan_client的导入，因为我们只使用hunyuan_api_client
from clients.hunyuan_api_client import HunyuanAPIClient
from clients.tencent_tts import synthesize_tencent_tts
from utils.doc_loader import fetch_url
from utils.audio import ensure_dir, mix_intro_with_voice, export_with_intro
from clients.search_agent import SearchAgent
from clients.prompt_adjuster import PromptAdjuster
from clients.instruction_analyzer import InstructionAnalyzer
import re
from io import BytesIO
from pydub import AudioSegment
import base64

# 配置日志
logger = logging.getLogger(__name__)


def retrieve_sources(cfg: Dict[str, Any], mode: str, query: str = "", url: str = "", doc_text: str = "", instruction: Optional[str] = None, instruction_analysis: Optional[Dict[str, Any]] = None, pdf_documents: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    # 查询模式
    if mode == "query":
        # 使用搜索代理优化查询
        try:
            search_agent = SearchAgent(cfg)
            
            # 如果有指令分析结果，使用分析结果中的搜索重点
            search_focus = []
            if instruction_analysis:
                search_focus = instruction_analysis.get("search_focus", [])
                logger.info(f"使用指令分析结果中的搜索重点: {search_focus}")
            
            # 生成优化查询
            optimized_query = search_agent.generate_search_query(
                query, 
                instruction=instruction,
                search_focus=search_focus
            )
            
            logger.info(f"原始查询: {query}")
            logger.info(f"优化查询: {optimized_query}")
            query = optimized_query
        except Exception as e:
            logger.error(f"搜索代理异常: {e}")
        
        client = BochaClient(cfg["bocha_base_url"], cfg["bocha_api_id"], cfg["bocha_api_key"], cfg["bocha_search_path"])
        items = client.search(query, count=8)
        return items
    
    # URL模式
    elif mode == "url":
        sources = []
        
        # 获取URL内容作为主要资料
        r = fetch_url(url)
        if r.get("success"):
            sources.append({"title": url, "url": url, "snippet": r.get("text", ""), "is_primary": True})
        
        # 根据 URL 内容生成搜索查询，获取补充资料
        supplementary_count = cfg.get("supplementary_search_count", 4)
        if supplementary_count > 0 and r.get("success") and r.get("text"):
            try:
                # 使用搜索代理生成相关查询
                search_agent = SearchAgent(cfg)
                
                # 如果有指令分析结果，使用分析结果中的搜索重点
                search_focus = []
                if instruction_analysis:
                    search_focus = instruction_analysis.get("search_focus", [])
                    logger.info(f"使用指令分析结果中的搜索重点: {search_focus}")
                
                # 取URL内容的前1000个字符作为生成查询的基础
                content_sample = r.get("text", "")[:1000]
                supplementary_query = search_agent.generate_search_query(
                    content_sample, 
                    instruction=instruction,
                    search_focus=search_focus
                )
                logger.info(f"补充查询: {supplementary_query}")
                
                # 执行补充搜索
                client = BochaClient(cfg["bocha_base_url"], cfg["bocha_api_id"], cfg["bocha_api_key"], cfg["bocha_search_path"])
                supplementary_items = client.search(supplementary_query, count=supplementary_count)
                
                # 添加补充资料，标记为非主要资料
                for item in supplementary_items:
                    item["is_primary"] = False
                    sources.append(item)
            except Exception as e:
                logger.error(f"补充搜索异常: {e}")
        
        return sources
    
    # 文档模式
    elif mode == "doc":
        sources = []
        
        # 如果有多个PDF文档，将每个PDF作为独立的主要资料
        if pdf_documents and len(pdf_documents) > 0:
            logger.info(f"处理 {len(pdf_documents)} 个PDF文档作为独立主要资料")
            
            # 根据文档数量动态调整每个文档的最大长度
            # 总上下文预算约90000字符，预留30000给补充资料和提示词
            total_budget = 60000
            max_per_doc = max(10000, total_budget // len(pdf_documents))
            logger.info(f"每个文档最大长度限制: {max_per_doc} 字符")
            
            for i, doc_info in enumerate(pdf_documents):
                title = doc_info.get("title", f"文档{i+1}")
                content = doc_info.get("content", "")
                
                # 清理不可打印字符
                clean_content = ''.join(char for char in content if char.isprintable() or char.isspace())
                if not clean_content.strip():
                    clean_content = content
                
                # 限制每个文档的内容长度
                if len(clean_content) > max_per_doc:
                    clean_content = clean_content[:max_per_doc] + f"\n...[内容已截断，原文共{len(content)}字符]"
                
                sources.append({
                    "title": title,
                    "url": "",
                    "snippet": clean_content,
                    "is_primary": True
                })
                logger.info(f"添加主要资料 [{i+1}]: {title}, 内容长度: {len(clean_content)}")
        else:
            # 单文档模式：将文档内容直接作为主要资料
            # 从指令中提取主题（如果有）
            doc_title = ""
            if instruction:
                # 尝试从指令中提取主题信息
                topic_match = re.search(r"主题：(.+)(?:\n|$)", instruction)
                if topic_match:
                    doc_title = topic_match.group(1).strip()
            
            # 如果没有从指令中提取到主题，尝试从文档内容提取一个标题
            if not doc_title:
                # 提取文档内容的前100个字符作为标题预览
                doc_preview = doc_text[:100].replace("\n", " ").strip()
                doc_title = doc_preview + "..." if len(doc_text) > 100 else doc_preview
            
            # 将文档内容作为主要资料
            sources.append({"title": doc_title, "url": "", "snippet": doc_text, "is_primary": True})
        
        # 根据文档内容生成搜索查询，获取补充资料
        supplementary_count = cfg.get("supplementary_search_count", 4)
        if supplementary_count > 0 and doc_text:
            try:
                # 使用搜索代理生成相关查询
                search_agent = SearchAgent(cfg)
                
                # 如果有指令分析结果，使用分析结果中的搜索重点
                search_focus = []
                if instruction_analysis:
                    search_focus = instruction_analysis.get("search_focus", [])
                    logger.info(f"使用指令分析结果中的搜索重点: {search_focus}")
                
                # 取文档内容的前1000个字符作为生成查询的基础
                content_sample = doc_text[:1000]
                supplementary_query = search_agent.generate_search_query(
                    content_sample, 
                    instruction=instruction,
                    search_focus=search_focus
                )
                logger.info(f"补充查询: {supplementary_query}")
                
                # 执行补充搜索
                client = BochaClient(cfg["bocha_base_url"], cfg["bocha_api_id"], cfg["bocha_api_key"], cfg["bocha_search_path"])
                supplementary_items = client.search(supplementary_query, count=supplementary_count)
                
                # 添加补充资料，标记为非主要资料
                for item in supplementary_items:
                    item["is_primary"] = False
                    sources.append(item)
            except Exception as e:
                logger.error(f"补充搜索异常: {e}")
        
        return sources
    
    return []


def build_outline_and_script(cfg: Dict[str, Any], topic: str, sources: List[Dict[str, Any]], style: str = "news", instruction: Optional[str] = None, mode: str = "query", original_input: str = "", instruction_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
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
    # 分离主要资料和补充资料
    primary_sources = [s for s in sources if s.get("is_primary", True)]
    supplementary_sources = [s for s in sources if not s.get("is_primary", True)]
    
    # 确保在脚本生成中使用所有主要资料的引用
    primary_evidence_parts = []
    for i, s in enumerate(primary_sources):
        # 获取内容并清理
        snippet = s.get('snippet') or ''
        clean_snippet = ''.join(char for char in snippet[:30000] if char.isprintable() or char.isspace())
        
        # 如果是PDF文档，特别标记
        if s.get('title', '').lower().endswith('.pdf'):
            primary_evidence_parts.append(
                f"[{i+1}] 文档：{s.get('title','')}\
内容：{clean_snippet}"
            )
        else:
            primary_evidence_parts.append(
                f"[{i+1}] 标题：{s.get('title','')}\
来源：{s.get('url','')}\
内容：{clean_snippet}"
            )

    primary_evidence = "\n\n".join(primary_evidence_parts)

    # 确保在脚本生成中使用所有补充资料的引用
    supplementary_evidence_parts = []
    for i, s in enumerate(supplementary_sources):
        # 获取内容并清理
        snippet = s.get('snippet') or ''
        clean_snippet = ''.join(char for char in snippet[:1000] if char.isprintable() or char.isspace())
        
        # 如果是PDF文档，特别标记
        if s.get('title', '').lower().endswith('.pdf'):
            supplementary_evidence_parts.append(
                f"[S{i+1}] 文档：{s.get('title','')}\
内容：{clean_snippet}"
            )
        else:
            supplementary_evidence_parts.append(
                f"[S{i+1}] 标题：{s.get('title','')}\
来源：{s.get('url','')}\
内容：{clean_snippet}"
            )

    supplementary_evidence = "\n\n".join(supplementary_evidence_parts)

    # 合并证据，主要资料在前，补充资料在后
    evidence = primary_evidence
    if supplementary_evidence:
        evidence += "\n\n【补充资料】\n" + supplementary_evidence

    # 处理特殊指令
    special_instruction = ""
    if instruction and instruction.strip():
        special_instruction = f"【特殊指令】\n{instruction.strip()}\n\n"

    # 检测是否需要生成英文内容
    is_english_mode = False
    
    # 优先使用指令分析结果判断是否需要英文
    if instruction_analysis and "is_english" in instruction_analysis:
        is_english_mode = instruction_analysis["is_english"]
        if is_english_mode:
            logger.info("根据指令分析结果使用英文模式")
    # 如果没有指令分析结果或结果中没有is_english，使用简单字符串匹配作为后备
    elif instruction:
        english_keywords = ["english", "in english", "generate in english", "use english", "英文", "用英文", "英语", "使用英文", "使用英语"]
        instruction_lower = instruction.lower()
        for keyword in english_keywords:
            if keyword.lower() in instruction_lower:
                is_english_mode = True
                logger.info(f"使用字符串匹配检测到英文生成指令: '{keyword}'")
                break
    
    # 初始提示词
    if is_english_mode:
        # 英文提示词模板
        base_prompt = (
            f"You are an experienced English podcast scriptwriter. Task: Create a high-quality two-person dialogue podcast script for the topic '{topic}'.\n\n"
            f"{special_instruction}"
            
            f"[Duration Requirements]\n"
            f"- Target duration: 8-15 minutes (approx. 1200-2250 words)\n"
            f"- Speech rate reference: about 150 words per minute\n\n"
            
            f"[Character Setup]\n"
            f"- Host A (Expert): Knowledgeable, logical, good at in-depth analysis, occasionally uses technical terms but explains them, tone is steady yet approachable\n"
            f"- Host B (Guide): Curious, good at asking questions, represents the audience's perspective, summarizes key points, raises questions, and maintains a natural, lively tone\n\n"
        )
        logger.info("使用英文提示词模板")
    else:
        # 中文提示词模板
        base_prompt = (
            f"你是资深中文播客编剧。任务：为话题《{topic}》创作高质量两人对话播客脚本。\n\n"
            f"{special_instruction}"
            
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
            f"4. 结尾（2-3轮）：核心观点回顾+行动建议+互动召唤\n\n"
            
            f"【对话风格】\n"
            f"- 自然口语化：使用'嗯'、'确实'、'你看'、'比如说'等口语词\n"
            f"- 情感节奏：适度停顿（用'...'表示）、语气词（'啊'、'呢'、'吧'）、情绪变化（惊讶/赞同/质疑）\n"
            f"- 互动真实：有打断、追问、玩笑、共鸣、不同观点的碰撞\n"
            f"- 避免说教：用故事化、场景化方式呈现，而非枯燥陈述\n\n"
            
            f"【事实依据】\n"
            f"- 所有关键事实、数据、观点必须来自下方证据，并标注[编号]\n"
            f"- 主要资料（标记为[1]、[2]等）是最重要的内容来源，应作为主要参考\n"
            f"- 必须覆盖所有主要资料：每个主要资料至少引用一次（出现[1]、[2]、…），若某份资料与主题弱相关，请简要说明并做最小引用\n"
            f"- 补充资料（标记为[S1]、[S2]等）只用于补充主要资料中缺失的信息，不应作为主要内容来源\n"
            f"- 证据不足时，明确说明'目前研究显示...'、'有观点认为...'等限定表达\n"
            f"- 禁止编造数据和事实\n\n"
            
            f"【证据材料】\n{evidence}\n\n"
            
            f"【输出规范】\n"
            f"- 纯对话格式，每行一句，按行交替（A→B→A→B...）\n"
            f"- 严格禁止使用任何角色标签，如'主播A：'、'A：'、'主持人：'、'旁白'等\n"
            f"- 严格禁止使用任何结构提示，如'（开场白部分）'、'（核心内容）'、'（结尾部分）'等\n"
            f"- 不要使用Markdown格式或其他标记语言\n"
            f"- 总字数控制在2400-4500字\n"
            f"- 确保对话完整，有明确的开头和结尾\n"
        )
    
    # 使用提示词自适应调整器分析内容并调整提示词
    try:
        # 初始化提示词调整器
        prompt_adjuster = PromptAdjuster(cfg)
        
        # 分析内容，确定适合的播客长度和结构
        analysis_result = prompt_adjuster.analyze_content(
            mode=mode,
            content=original_input,
            sources=sources,
            instruction=instruction
        )
        
        print(f"\n内容分析结果: {analysis_result}")
        
        # 根据分析结果调整提示词
        adjusted_prompt = prompt_adjuster.adjust_prompt(base_prompt, analysis_result)
        base_prompt = adjusted_prompt
    except Exception as e:
        print(f"提示词自适应调整失败: {e}")
        # 如果调整失败，使用原始提示词
    
    # 使用最终的提示词
    prompt = base_prompt
    
    messages = [
        {"Role": "system", "Content": "你是一个资深中文播客编剧，擅长创作自然流畅、信息丰富、情感真实的对话脚本。你生成的脚本必须是纯对话形式，不包含任何角色标签（如'主播A：'）或结构提示（如'（开场白部分）'）。每行一句对话，按A→B→A→B的顺序交替。"},
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
    
    # 后处理：清理角色标签和结构提示
    # 清理角色标签，如"主播A："、"A："等
    script = re.sub(r"^\s*主播[A-Za-z]\s*[：:]\s*", "", script, flags=re.MULTILINE)
    script = re.sub(r"^\s*[A-Za-z]\s*[：:]\s*", "", script, flags=re.MULTILINE)
    script = re.sub(r"^\s*主持人\s*[：:]\s*", "", script, flags=re.MULTILINE)
    script = re.sub(r"^\s*旁白\s*[：:]\s*", "", script, flags=re.MULTILINE)
    
    # 清理结构提示，如"（开场白部分）"、"(核心内容)"等
    script = re.sub(r"[（(][^）)]*[）)]\s*", "", script)
    
    return {"script": script}


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
        t = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff，。！？；、：“”‘’()[]\-~…….!,?;: ]", "", t)
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

    # 更新片头音乐映射，支持新的风格
    bgm_map = {
        # 原有风格
        "history": cfg["bgm_history"],
        "entertainment": cfg["bgm_entertainment"],
        "serious": cfg["bgm_serious"],
        # 新增风格
        "chengzhang": "chengzhang.mp3",
        "kejigan": "kejigan.mp3",
        "shangye": "shangye.mp3",
        "yingshi": "yingshi.mp3",
        "zhichang": "zhichang.mp3",
        "tongyong": "tongyong.MP3"
    }
    
    # 获取片头音乐文件路径
    intro_file = os.path.join(cfg["assets_bgm_dir"], bgm_map.get(intro_style, "tongyong.MP3"))
    
    # 如果指定风格的文件不存在，尝试使用通用风格
    if not os.path.exists(intro_file):
        intro_file = os.path.join(cfg["assets_bgm_dir"], "tongyong.MP3")
    
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


def _split_for_tts(text: str, limit: int = 240) -> List[str]:
    """将长文本切分为腾讯TTS可接受的短片段。limit 取一个安全阈值（中文字符数）。"""
    if not text:
        return []
    parts: List[str] = []
    # 先按段落
    paragraphs = re.split(r"\n+", text)
    for p in paragraphs:
        if not p.strip():
            continue
        # 如果段落较短，直接添加
        if len(p) <= limit:
            parts.append(p)
            continue
        # 否则按句子切分
        sentences = re.split(r"([。！？.!?])", p)
        current = ""
        for i in range(0, len(sentences), 2):
            s = sentences[i]
            # 添加标点（如果有）
            if i + 1 < len(sentences):
                s += sentences[i + 1]
            # 如果当前句子加上已有内容超过限制，先保存已有内容
            if len(current) + len(s) > limit:
                if current:
                    parts.append(current)
                    current = ""
                # 如果单个句子超过限制，需要进一步切分
                if len(s) > limit:
                    # 按逗号等次级标点切分
                    sub_sentences = re.split(r"([，、,;；])", s)
                    sub_current = ""
                    for j in range(0, len(sub_sentences), 2):
                        ss = sub_sentences[j]
                        # 添加标点（如果有）
                        if j + 1 < len(sub_sentences):
                            ss += sub_sentences[j + 1]
                        # 如果当前子句加上已有内容超过限制，先保存已有内容
                        if len(sub_current) + len(ss) > limit:
                            if sub_current:
                                parts.append(sub_current)
                                sub_current = ""
                            # 如果单个子句仍然超过限制，按字符硬切分
                            while len(ss) > limit:
                                parts.append(ss[:limit])
                                ss = ss[limit:]
                            sub_current = ss
                        else:
                            sub_current += ss
                    # 保存最后的子句内容
                    if sub_current:
                        parts.append(sub_current)
                else:
                    current = s
            else:
                current += s
        # 保存最后的句子内容
        if current:
            parts.append(current)
    return parts


def run_end_to_end(mode: str, topic_or_url_or_text: str, style: str = "news", intro_style: str = "serious", speed: int = 0,
                   voice_a: Optional[str] = None, voice_b: Optional[str] = None, instruction: Optional[str] = None,
                   file_titles: Optional[List[str]] = None, pdf_documents: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    cfg = load_ini()
    
    # 分析指令
    instruction_analysis = None
    if instruction:
        try:
            analyzer = InstructionAnalyzer(cfg)
            instruction_analysis = analyzer.analyze_instruction(
                instruction=instruction, 
                mode=mode, 
                content=topic_or_url_or_text,
                file_titles=file_titles  # 传递文件标题列表
            )
            logger.info(f"指令分析结果: {instruction_analysis}")
        except Exception as e:
            logger.error(f"指令分析失败: {e}")
    
    # 根据分析结果设置英文模式
    is_english_mode = False
    if instruction_analysis and "is_english" in instruction_analysis:
        is_english_mode = instruction_analysis["is_english"]
    # 如果没有指令分析结果或结果中没有is_english，使用简单字符串匹配作为后备
    elif instruction:
        english_keywords = ["english", "in english", "generate in english", "use english", "英文", "用英文", "英语", "使用英文", "使用英语"]
        instruction_lower = instruction.lower()
        for keyword in english_keywords:
            if keyword.lower() in instruction_lower:
                is_english_mode = True
                logger.info(f"使用字符串匹配检测到英文生成指令: '{keyword}'")
                break
    
    # 英文模式的特殊处理
    if is_english_mode:
        # 英文模式自动选择英文音色
        if not voice_a or not voice_b:
            logger.info("英文模式自动选择英文音色")
            # 使用WeJames和WeWinny音色
            voice_a = "501008:WeJames"  # 英文男声
            voice_b = "501009:WeWinny"  # 英文女声
        
        # 英文模式使用通用片头
        if intro_style != "tongyong":
            logger.info("英文模式使用通用片头")
            intro_style = "tongyong"
    if cfg.get("tts_provider") != "tencent":
        raise RuntimeError("当前仅启用腾讯云 TTS，请在配置中设置 [tts] provider = tencent")
    if not (cfg.get("tencent_secret_id") and cfg.get("tencent_secret_key")):
        raise RuntimeError("请在 Source_config_podcast.ini 配置腾讯云 TTS 密钥")
    if mode == "query":
        topic = topic_or_url_or_text
        sources = retrieve_sources(cfg, "query", query=topic, instruction=instruction, instruction_analysis=instruction_analysis)
    elif mode == "url":
        topic = topic_or_url_or_text
        sources = retrieve_sources(cfg, "url", url=topic, instruction=instruction, instruction_analysis=instruction_analysis)
    else:
        # 尝试从指令中提取主题
        if instruction and "主题：" in instruction:
            # 提取主题信息
            topic_match = re.search(r"主题：(.+)(?:\n|$)", instruction)
            if topic_match:
                topic = topic_match.group(1).strip()
                logger.info(f"从指令中提取的主题: {topic}")
            else:
                # 从文档内容提取主题
                content_preview = topic_or_url_or_text[:50].replace("\n", " ").strip()
                topic = content_preview + "..." if len(topic_or_url_or_text) > 50 else content_preview
                logger.info(f"从文档内容提取的主题: {topic}")
        else:
            # 从文档内容提取主题
            content_preview = topic_or_url_or_text[:50].replace("\n", " ").strip()
            topic = content_preview + "..." if len(topic_or_url_or_text) > 50 else content_preview
            logger.info(f"从文档内容提取的主题: {topic}")
        
        sources = retrieve_sources(cfg, "doc", doc_text=topic_or_url_or_text, instruction=instruction, instruction_analysis=instruction_analysis, pdf_documents=pdf_documents)
    script_res = build_outline_and_script(cfg, topic, sources, style=style, instruction=instruction,
                                   mode=mode, original_input=topic_or_url_or_text, instruction_analysis=instruction_analysis)
    audio_path, transcript_path = tts_and_mix(cfg, script_res["script"], intro_style=intro_style, speed=speed,
                                              voice_a=voice_a, voice_b=voice_b)
    return {
        "audio_path": audio_path,
        "transcript_path": transcript_path,
        "sources": sources,
        "script": script_res["script"],
    }


def generate_stream(mode: str, topic_or_url_or_text: str, style: str = "news", intro_style: str = "serious", speed: int = 0,
                    voice_a: Optional[str] = None, voice_b: Optional[str] = None, instruction: Optional[str] = None):
    cfg = load_ini()
    
    # 分析指令
    instruction_analysis = None
    if instruction:
        try:
            analyzer = InstructionAnalyzer(cfg)
            instruction_analysis = analyzer.analyze_instruction(
                instruction=instruction, 
                mode=mode, 
                content=topic_or_url_or_text
            )
            logger.info(f"指令分析结果: {instruction_analysis}")
        except Exception as e:
            logger.error(f"指令分析失败: {e}")
    
    # 根据分析结果设置英文模式
    is_english_mode = False
    if instruction_analysis and "is_english" in instruction_analysis:
        is_english_mode = instruction_analysis["is_english"]
    # 如果没有指令分析结果或结果中没有is_english，使用简单字符串匹配作为后备
    elif instruction:
        english_keywords = ["english", "in english", "generate in english", "use english", "英文", "用英文", "英语", "使用英文", "使用英语"]
        instruction_lower = instruction.lower()
        for keyword in english_keywords:
            if keyword.lower() in instruction_lower:
                is_english_mode = True
                logger.info(f"使用字符串匹配检测到英文生成指令: '{keyword}'")
                break
    
    # 英文模式的特殊处理
    if is_english_mode:
        # 英文模式自动选择英文音色
        if not voice_a or not voice_b:
            logger.info("英文模式自动选择英文音色")
            # 使用WeJames和WeWinny音色
            voice_a = "501008:WeJames"  # 英文男声
            voice_b = "501009:WeWinny"  # 英文女声
        
        # 英文模式使用通用片头
        if intro_style != "tongyong":
            logger.info("英文模式使用通用片头")
            intro_style = "tongyong"
    """快方案：逐段TTS边生成边返回。
    Yields:
        {"type":"chunk", "index": i, "path": seg_path, "text": chunk_text, "transcript": so_far}
        最后：{"type":"done", "final_audio": final_path, "transcript": full_text}
    """
    if cfg.get("tts_provider") != "tencent":
        yield {"type": "error", "error": "当前仅启用腾讯云 TTS"}
        return
    # 规范化输入类型，避免大小写导致走到文档分支
    mode_norm = (mode or "").strip().lower()
    # 1) 取来源
    if mode_norm == "query":
        topic = topic_or_url_or_text
        sources = retrieve_sources(cfg, "query", query=topic, instruction=instruction, instruction_analysis=instruction_analysis)
    elif mode_norm == "url":
        topic = topic_or_url_or_text
        sources = retrieve_sources(cfg, "url", url=topic, instruction=instruction, instruction_analysis=instruction_analysis)
    else:
        # 尝试从指令中提取主题
        if instruction and "主题：" in instruction:
            # 提取主题信息
            topic_match = re.search(r"主题：(.+)(?:\n|$)", instruction)
            if topic_match:
                topic = topic_match.group(1).strip()
                logger.info(f"从指令中提取的主题: {topic}")
            else:
                # 从文档内容提取主题
                content_preview = topic_or_url_or_text[:50].replace("\n", " ").strip()
                topic = content_preview + "..." if len(topic_or_url_or_text) > 50 else content_preview
                logger.info(f"从文档内容提取的主题: {topic}")
        else:
            # 从文档内容提取主题
            content_preview = topic_or_url_or_text[:50].replace("\n", " ").strip()
            topic = content_preview + "..." if len(topic_or_url_or_text) > 50 else content_preview
            logger.info(f"从文档内容提取的主题: {topic}")
        
        sources = retrieve_sources(cfg, "doc", doc_text=topic_or_url_or_text, instruction=instruction, instruction_analysis=instruction_analysis)
    # 2) 生成脚本
    script_res = build_outline_and_script(cfg, topic, sources, style=style, instruction=instruction,
                                   mode=mode_norm, original_input=topic_or_url_or_text, instruction_analysis=instruction_analysis)
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
        # 保存片段
        seg_path = os.path.join(chunks_dir, f"chunk_{idx:03d}.mp3")
        with open(seg_path, "wb") as f:
            f.write(sec["bytes"])
        # 更新转写
        try:
            raw_text = pairs[idx][2] if len(pairs[idx]) > 2 else text
            transcript_so_far += raw_text + "\n"
        except Exception:
            transcript_so_far += text + "\n"
        # 返回片段
        yield {"type": "chunk", "index": idx, "path": seg_path, "text": text, "transcript": transcript_so_far}
        # 累积片段
        seg = AudioSegment.from_file(BytesIO(sec["bytes"]), format="mp3")
        final_segments.append(seg)
    # 拼接最终音频
    final_audio = AudioSegment.silent(duration=100)
    pause = AudioSegment.silent(duration=200)
    for seg in final_segments:
        final_audio = final_audio.append(seg, crossfade=50).append(pause, crossfade=0)
    voice_path = os.path.join(cfg["output_dir"], "podcast_voice.mp3")
    final_audio.export(voice_path, format="mp3", bitrate="192k")
    
    # 更新片头音乐映射，支持新的风格
    bgm_map = {
        # 原有风格
        "history": cfg["bgm_history"],
        "entertainment": cfg["bgm_entertainment"],
        "serious": cfg["bgm_serious"],
        # 新增风格
        "chengzhang": "chengzhang.mp3",
        "kejigan": "kejigan.mp3",
        "shangye": "shangye.mp3",
        "yingshi": "yingshi.mp3",
        "zhichang": "zhichang.mp3",
        "tongyong": "tongyong.MP3"
    }
    
    # 获取片头音乐文件路径
    intro_file = os.path.join(cfg["assets_bgm_dir"], bgm_map.get(intro_style, "tongyong.MP3"))
    
    # 如果指定风格的文件不存在，尝试使用通用风格
    if not os.path.exists(intro_file):
        intro_file = os.path.join(cfg["assets_bgm_dir"], "tongyong.MP3")
    
    out_mp3 = os.path.join(cfg["output_dir"], "podcast_final.mp3")
    # 合成片头
    try:
        export_with_intro(final_audio, out_mp3, intro_path=intro_file if os.path.exists(intro_file) else None)
    except Exception:
        mix_intro_with_voice(intro_file if os.path.exists(intro_file) else None, voice_path, out_mp3)
    # 保存完整转写
    transcript_path = os.path.join(cfg["output_dir"], "podcast_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_so_far)
    # 返回最终结果
    yield {"type": "done", "final_audio": out_mp3, "transcript": transcript_so_far}

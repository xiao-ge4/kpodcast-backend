from typing import List, Dict, Any, Tuple, Optional
import os
import logging
from utils.config_loader import load_ini
from clients.bocha_client import BochaClient
# 移除对hunyuan_client的导入，因为我们只使用hunyuan_api_client
from clients.hunyuan_api_client import HunyuanAPIClient
from clients.tencent_tts import synthesize_tencent_tts
from utils.doc_loader import fetch_url
from utils.audio import ensure_dir, mix_intro_with_voice, export_with_intro, export_with_dynamic_intro
from clients.search_agent import SearchAgent
from utils.intro_config import get_bgm_length_strategy, get_loop_crossfade_ms
from clients.prompt_adjuster import PromptAdjuster
from clients.instruction_analyzer import InstructionAnalyzer
from utils.intro_config import get_intro_script, get_intro_bgm_filename, INTRO_BGM_FILES
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


def build_outline_and_script(cfg: Dict[str, Any], topic: str, sources: List[Dict[str, Any]], style: str = "chat", custom_style: Optional[str] = None, instruction: Optional[str] = None, mode: str = "query", original_input: str = "", instruction_analysis: Optional[Dict[str, Any]] = None, host_mode: str = "dual") -> Dict[str, str]:
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

    # 处理播客风格
    style_descriptions = {
        "chat": {
            "cn_single": "轻松闲聊风格：像朋友间的随意聊天，语气轻松自然，可以有适当的口语化表达、感叹词，偶尔开个小玩笑，让听众感觉像在听朋友分享有趣的事情。",
            "cn_dual": "轻松闲聊风格：两位主播像老朋友聊天一样自然，语气轻松随意，可以互相调侃、开玩笑，有真实的情感互动，让听众感觉像在旁听两个朋友的有趣对话。",
            "en_single": "Casual chat style: Like a friendly conversation, relaxed and natural tone, with appropriate colloquialisms and exclamations, occasional light humor, making listeners feel like they're hearing a friend share interesting things.",
            "en_dual": "Casual chat style: Two hosts chat like old friends, relaxed and natural, with playful banter and genuine emotional interaction, making listeners feel like they're overhearing an interesting conversation between friends."
        },
        "professional": {
            "cn_single": "专业深度风格：语气沉稳专业，逻辑严谨，善于深度分析和解读，使用准确的专业术语但会适当解释，给听众带来权威可信的感觉。",
            "cn_dual": "专业深度风格：两位主播都展现专业素养，讨论深入透彻，逻辑严谨，善于从多角度分析问题，使用准确的专业术语并互相补充解释，给听众带来权威可信的感觉。",
            "en_single": "Professional in-depth style: Steady and professional tone, rigorous logic, good at deep analysis and interpretation, using accurate terminology with appropriate explanations, giving listeners a sense of authority and credibility.",
            "en_dual": "Professional in-depth style: Both hosts demonstrate professional expertise, with thorough and rigorous discussion, analyzing issues from multiple angles, using accurate terminology and complementing each other's explanations."
        },
        "story": {
            "cn_single": "故事叙述风格：像讲故事一样娓娓道来，注重情节铺垫和悬念设置，语言生动形象，善用比喻和场景描写，让听众沉浸在故事中。",
            "cn_dual": "故事叙述风格：两位主播像在讲述一个精彩的故事，一人主讲一人配合，注重情节铺垫和悬念设置，语言生动形象，让听众沉浸在故事中。",
            "en_single": "Storytelling style: Narrate like telling a story, with attention to plot development and suspense, vivid and imaginative language, good use of metaphors and scene descriptions, immersing listeners in the narrative.",
            "en_dual": "Storytelling style: Two hosts tell an engaging story together, one leading and one supporting, with attention to plot development and suspense, vivid language that immerses listeners in the narrative."
        },
        "debate": {
            "cn_single": "观点碰撞风格：主播会呈现不同观点和立场，进行自我辩论式的分析，展示问题的多面性，引导听众独立思考，语气时而坚定时而质疑。",
            "cn_dual": "观点碰撞风格：两位主播持有不同观点或立场，进行友好但有深度的辩论，互相质疑和反驳，展示问题的多面性，激发听众思考。",
            "en_single": "Debate style: The host presents different viewpoints and positions, conducting self-debate analysis, showing multiple sides of issues, guiding listeners to think independently, with tones alternating between firm and questioning.",
            "en_dual": "Debate style: Two hosts hold different viewpoints or positions, engaging in friendly but deep debate, questioning and rebutting each other, showing multiple sides of issues and stimulating listener thought."
        },
        "educational": {
            "cn_single": "科普教学风格：像一位耐心的老师，循序渐进地讲解，善于用通俗易懂的语言解释复杂概念，多用类比和实例，确保听众能够理解和学到知识。",
            "cn_dual": "科普教学风格：一位主播像老师讲解，另一位像学生提问，循序渐进，善于用通俗易懂的语言解释复杂概念，多用类比和实例，确保听众能够理解和学到知识。",
            "en_single": "Educational style: Like a patient teacher, explaining step by step, good at using accessible language to explain complex concepts, with many analogies and examples, ensuring listeners can understand and learn.",
            "en_dual": "Educational style: One host explains like a teacher, the other asks questions like a student, progressing step by step, using accessible language to explain complex concepts with analogies and examples."
        }
    }
    
    # 生成风格指令
    style_instruction = ""
    if style == "custom" and custom_style and custom_style.strip():
        # 用户自定义风格
        style_instruction = f"【播客风格要求】\n用户希望的风格：{custom_style.strip()}\n请根据用户描述的风格特点来调整播客的语气、表达方式和整体氛围。\n\n"
        logger.info(f"使用自定义风格: {custom_style}")
    elif style in style_descriptions:
        # 预设风格
        style_info = style_descriptions[style]
        style_instruction = f"【播客风格要求】\n"
        logger.info(f"使用预设风格: {style}")
    else:
        # 默认使用轻松闲聊风格
        style = "chat"
        style_info = style_descriptions["chat"]
        logger.info("使用默认风格: chat")

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
    
    # 根据风格和语言生成风格指令文本
    def get_style_text(style_key: str, is_english: bool, is_single: bool) -> str:
        if style_key == "custom":
            return ""  # 自定义风格已在 style_instruction 中处理
        if style_key not in style_descriptions:
            style_key = "chat"
        style_info = style_descriptions[style_key]
        if is_english:
            return style_info["en_single"] if is_single else style_info["en_dual"]
        else:
            return style_info["cn_single"] if is_single else style_info["cn_dual"]

    # 初始提示词 - 根据 host_mode 选择单人或双人模板
    if host_mode == "single":
        # 单人播客模板
        if is_english_mode:
            # 生成风格文本
            if style == "custom" and custom_style:
                style_text = f"[Style Requirements]\nUser requested style: {custom_style}\nPlease adjust the podcast's tone, expression, and overall atmosphere according to the user's style description.\n\n"
            else:
                style_text = f"[Style Requirements]\n{get_style_text(style, True, True)}\n\n"
            
            # 英文单人播客提示词模板
            base_prompt = (
                f"You are an experienced English podcast scriptwriter. Task: Create a high-quality single-host monologue podcast script for the topic '{topic}'.\n\n"
                f"{special_instruction}"
                f"{style_text}"
                
                f"[Duration Requirements]\n"
                f"- Target duration: 5-12 minutes (approx. 750-1800 words)\n"
                f"- Speech rate reference: about 150 words per minute\n\n"
                
                f"[Host Character]\n"
                f"- Single Host: Knowledgeable yet approachable, speaks directly to the audience, uses conversational tone\n"
                f"- Engages listeners through rhetorical questions, personal insights, and vivid examples\n"
                f"- Maintains energy through varied pacing and emotional expression\n\n"
                
                f"[Content Structure]\n"
                f"1. Opening (1-2 paragraphs): Hook the audience + introduce the topic + why it matters\n"
                f"2. Main Content (4-6 sections):\n"
                f"   - Progress logically (background → current state → analysis → implications)\n"
                f"   - Use transitions like 'Now let's look at...', 'Speaking of which...', 'Here's where it gets interesting...'\n"
                f"   - Include specific examples and data [citation numbers]\n"
                f"3. Deep Dive (2-3 sections): Controversies / multiple perspectives / future outlook\n"
                f"4. Conclusion (1-2 paragraphs): Key takeaways + call to action + sign off\n\n"
                
                f"[Speaking Style]\n"
                f"- Conversational and engaging: Use 'you', 'we', 'imagine this...'\n"
                f"- Natural rhythm: Include pauses (use '...'), emphasis, and emotional variation\n"
                f"- Rhetorical questions to maintain engagement: 'But here's the thing...', 'So what does this mean?'\n"
                f"- Storytelling approach: Use anecdotes and scenarios rather than dry facts\n\n"
                
                f"[Evidence Requirements]\n"
                f"- All key facts, data, and opinions must come from the evidence below, marked with [citation number]\n"
                f"- Primary sources (marked as [1], [2], etc.) are the main content sources\n"
                f"- Must cover all primary sources: cite each at least once\n"
                f"- Supplementary sources (marked as [S1], [S2], etc.) only fill gaps in primary sources\n"
                f"- When evidence is insufficient, use limiting expressions like 'Research suggests...', 'Some experts believe...'\n"
                f"- Do not fabricate data or facts\n\n"
                
                f"[Evidence Materials]\n{evidence}\n\n"
                
                f"[Output Format]\n"
                f"- Pure monologue format, written as continuous paragraphs\n"
                f"- NO host labels like 'Host:', 'Speaker:', etc.\n"
                f"- NO structural hints like '(Opening)', '(Main Content)', '(Conclusion)'\n"
                f"- No Markdown formatting\n"
                f"- Total word count: 750-1800 words\n"
                f"- Ensure complete narrative with clear beginning and ending\n"
            )
            logger.info("使用英文单人播客提示词模板")
        else:
            # 生成风格文本
            if style == "custom" and custom_style:
                style_text = f"【播客风格要求】\n用户希望的风格：{custom_style}\n请根据用户描述的风格特点来调整播客的语气、表达方式和整体氛围。\n\n"
            else:
                style_text = f"【播客风格要求】\n{get_style_text(style, False, True)}\n\n"
            
            # 中文单人播客提示词模板
            base_prompt = (
                f"你是资深中文播客编剧。任务：为话题《{topic}》创作高质量单人独白播客脚本。\n\n"
                f"{special_instruction}"
                f"{style_text}"
                
                f"【时长要求】\n"
                f"- 目标时长：5-12分钟（约1500-3600字）\n"
                f"- 语速参考：每分钟300字左右\n\n"
                
                f"【主播设定】\n"
                f"- 单人主播：知识渊博但亲和力强，直接与听众对话，使用口语化表达\n"
                f"- 通过设问、个人见解、生动案例来吸引听众\n"
                f"- 通过节奏变化和情感表达保持活力\n\n"
                
                f"【内容结构】（严格遵循）\n"
                f"1. 开场（1-2段）：吸引注意力+引入话题+为什么重要\n"
                f"2. 核心内容（4-6个部分）：\n"
                f"   - 按逻辑层层递进（背景→现状→分析→影响）\n"
                f"   - 使用过渡语：'接下来我们来看...'、'说到这里...'、'有意思的是...'\n"
                f"   - 每个要点配合具体案例/数据[引用编号]\n"
                f"3. 深度探讨（2-3个部分）：争议点/多角度思考/未来展望\n"
                f"4. 结尾（1-2段）：核心观点回顾+行动建议+结束语\n\n"
                
                f"【讲述风格】\n"
                f"- 自然口语化：使用'你'、'我们'、'想象一下...'等拉近距离\n"
                f"- 情感节奏：适度停顿（用'...'表示）、语气词（'啊'、'呢'、'吧'）、情绪变化\n"
                f"- 设问互动：'但问题来了...'、'这意味着什么呢？'、'你可能会问...'\n"
                f"- 故事化呈现：用故事、场景、比喻来讲解，而非枯燥陈述\n\n"
                
                f"【事实依据】（非常重要，必须严格遵守）\n"
                f"- 所有关键事实、数据、观点必须来自下方证据，并在文中标注[编号]，如[1]、[2]等\n"
                f"- 主要资料（标记为[1]、[2]等）是最重要的内容来源，应作为主要参考\n"
                f"- 必须覆盖所有主要资料：每个主要资料至少引用一次（必须出现[1]、[2]、…），若某份资料与主题弱相关，请简要说明并做最小引用\n"
                f"- 补充资料（标记为[S1]、[S2]等）只用于补充主要资料中缺失的信息，不应作为主要内容来源\n"
                f"- 引用示例：'根据最新研究[1]，人工智能正在...'、'有专家指出[2]...'\n"
                f"- 证据不足时，明确说明'目前研究显示...'、'有观点认为...'等限定表达\n"
                f"- 禁止编造数据和事实\n\n"
                
                f"【证据材料】\n{evidence}\n\n"
                
                f"【输出规范】\n"
                f"- 纯独白格式，以连续段落形式书写\n"
                f"- 严格禁止使用任何角色标签，如'主播：'、'旁白：'等\n"
                f"- 严格禁止使用任何结构提示，如'（开场部分）'、'（核心内容）'等\n"
                f"- 不要使用Markdown格式或其他标记语言\n"
                f"- 总字数控制在1500-3600字\n"
                f"- 确保叙述完整，有明确的开头和结尾\n"
            )
            logger.info("使用中文单人播客提示词模板")
    elif is_english_mode:
        # 生成风格文本
        if style == "custom" and custom_style:
            style_text = f"[Style Requirements]\nUser requested style: {custom_style}\nPlease adjust the podcast's tone, expression, and overall atmosphere according to the user's style description.\n\n"
        else:
            style_text = f"[Style Requirements]\n{get_style_text(style, True, False)}\n\n"
        
        # 英文双人播客提示词模板
        base_prompt = (
            f"You are an experienced English podcast scriptwriter. Task: Create a high-quality two-person dialogue podcast script for the topic '{topic}'.\n\n"
            f"{special_instruction}"
            f"{style_text}"
            
            f"[Duration Requirements]\n"
            f"- Target duration: 8-15 minutes (approx. 1200-2250 words)\n"
            f"- Speech rate reference: about 150 words per minute\n\n"
            
            f"[Character Setup]\n"
            f"- Host A (Expert): Knowledgeable, logical, good at in-depth analysis, occasionally uses technical terms but explains them, tone is steady yet approachable\n"
            f"- Host B (Guide): Curious, good at asking questions, represents the audience's perspective, summarizes key points, raises questions, and maintains a natural, lively tone\n\n"
        )
        logger.info("使用英文双人播客提示词模板")
    else:
        # 生成风格文本
        if style == "custom" and custom_style:
            style_text = f"【播客风格要求】\n用户希望的风格：{custom_style}\n请根据用户描述的风格特点来调整播客的语气、表达方式和整体氛围。\n\n"
        else:
            style_text = f"【播客风格要求】\n{get_style_text(style, False, False)}\n\n"
        
        # 中文双人播客提示词模板
        base_prompt = (
            f"你是资深中文播客编剧。任务：为话题《{topic}》创作高质量两人对话播客脚本。\n\n"
            f"{special_instruction}"
            f"{style_text}"
            
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
    
    # 根据 host_mode 选择不同的 system message
    if host_mode == "single":
        system_content = "你是一个资深中文播客编剧，擅长创作自然流畅、信息丰富、情感真实的独白脚本。你生成的脚本必须是纯独白形式，以连续段落书写，不包含任何角色标签或结构提示。"
    else:
        system_content = "你是一个资深中文播客编剧，擅长创作自然流畅、信息丰富、情感真实的对话脚本。你生成的脚本必须是纯对话形式，不包含任何角色标签（如'主播A：'）或结构提示（如'（开场白部分）'）。每行一句对话，按A→B→A→B的顺序交替。"
    
    messages = [
        {"Role": "system", "Content": system_content},
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


def generate_intro_voice(cfg: Dict[str, Any], intro_style: str, speed: int = 0,
                         voice_a: Optional[str] = None, voice_b: Optional[str] = None,
                         host_mode: str = "dual", custom_script: Optional[str] = None) -> Optional[AudioSegment]:
    """
    根据风格动态生成片头语音
    
    Args:
        cfg: 配置信息
        intro_style: 片头风格代码 (tech, business, life, etc.)
        speed: 语速
        voice_a: 主播A音色
        voice_b: 主播B音色
        host_mode: 主持人模式 (dual/single)
        custom_script: 自定义片头文案（多行文本，每行一句）
    
    Returns:
        片头语音 AudioSegment，如果是通用风格则返回 None
    """
    # 获取片头文案（支持自定义文案）
    intro_scripts = get_intro_script(intro_style, host_mode, custom_script=custom_script)
    
    if not intro_scripts:
        # 通用风格，无语音
        logger.info(f"片头风格 {intro_style} 无语音文案，将只使用背景音乐")
        return None
    
    vnum_a = _parse_voice(voice_a, cfg.get("voice_role_a", "501006"))
    vnum_b = _parse_voice(voice_b, cfg.get("voice_role_b", "601007"))
    
    segments: List[AudioSegment] = []
    
    for idx, text in enumerate(intro_scripts):
        if not text.strip():
            continue
        
        # 根据模式选择音色
        if host_mode == "single":
            use_voice = vnum_a
        else:
            # 双人模式：A/B交替
            use_voice = vnum_a if (idx % 2 == 0) else vnum_b
        
        # 调用TTS
        sec = synthesize_tencent_tts(
            text,
            secret_id=cfg["tencent_secret_id"],
            secret_key=cfg["tencent_secret_key"],
            region=cfg["tencent_region"],
            voice=use_voice,
            speed=speed,
            codec="mp3",
        )
        
        if not sec.get("success") or not sec.get("bytes"):
            logger.warning(f"片头TTS失败: {sec.get('error')}, 文本: {text}")
            continue
        
        seg = AudioSegment.from_file(BytesIO(sec["bytes"]), format="mp3")
        segments.append(seg)
    
    if not segments:
        logger.warning("片头语音生成失败，所有TTS调用都失败了")
        return None
    
    # 拼接片头语音，段间留短暂停顿
    intro_audio = AudioSegment.silent(duration=50)
    pause = AudioSegment.silent(duration=150)  # 片头语音之间的停顿稍短
    
    for seg in segments:
        intro_audio = intro_audio.append(seg, crossfade=30).append(pause, crossfade=0)
    
    logger.info(f"片头语音生成成功，风格: {intro_style}, 时长: {len(intro_audio)}ms")
    return intro_audio


def tts_and_mix(cfg: Dict[str, Any], script: str, intro_style: str = "serious", speed: int = 0,
                voice_a: Optional[str] = None, voice_b: Optional[str] = None, host_mode: str = "dual",
                custom_intro_script: Optional[str] = None, custom_intro_bgm: Optional[str] = None) -> Tuple[str, str]:
    ensure_dir(cfg["output_dir"])
    # 分段合成，规避 TextTooLong
    # 单人播客使用更小的 limit，因为段落可能更长
    tts_limit = 120 if host_mode == "single" else 220
    chunks = _split_for_tts(script, limit=tts_limit)
    if not chunks:
        raise RuntimeError("脚本为空，无法合成TTS")
    segments: List[AudioSegment] = []
    fillers = ["嗯，我们继续。", "好的，接着说。", "下面进入下一段。"]
    vnum_a = _parse_voice(voice_a, cfg.get("voice_role_a", "501006"))
    vnum_b = _parse_voice(voice_b, cfg.get("voice_role_b", "601007"))
    for idx, ch in enumerate(chunks):
        text1 = _sanitize_for_tts(ch, aggressive=False)
        # 根据 host_mode 选择音色
        if host_mode == "single":
            # 单人模式：全程使用同一个音色 (voice_a)
            use_voice = vnum_a
        else:
            # 双人模式：交替角色（奇偶行切换）
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

    # 动态生成片头语音
    logger.info(f"开始生成片头，风格: {intro_style}, 主持人模式: {host_mode}, 自定义文案: {bool(custom_intro_script)}")
    intro_voice = generate_intro_voice(cfg, intro_style, speed=speed, 
                                        voice_a=voice_a, voice_b=voice_b, host_mode=host_mode,
                                        custom_script=custom_intro_script)
    
    # 获取片头背景音乐文件路径
    # 如果有自定义BGM，优先使用
    if custom_intro_bgm and os.path.exists(custom_intro_bgm):
        intro_bgm_file = custom_intro_bgm
        # 自定义BGM使用loop策略
        bgm_strategy = "loop"
        logger.info(f"使用自定义片头BGM: {intro_bgm_file}")
    else:
        # 优先使用新的 intro_bgm 目录
        intro_bgm_dir = os.path.join(os.path.dirname(cfg["assets_bgm_dir"]), "intro_bgm")
        bgm_filename = get_intro_bgm_filename(intro_style)
        intro_bgm_file = os.path.join(intro_bgm_dir, bgm_filename)
        
        # 如果新目录不存在或文件不存在，回退到旧目录
        if not os.path.exists(intro_bgm_file):
            # 尝试旧的映射
            old_bgm_map = {
                "tech": "kejigan.mp3",
                "business": "shangye.mp3",
                "growth": "chengzhang.mp3",
                "entertainment": "yingshi.mp3",
                "general": "tongyong.MP3",
            }
            old_filename = old_bgm_map.get(intro_style, "tongyong.MP3")
            intro_bgm_file = os.path.join(cfg["assets_bgm_dir"], old_filename)
            
            if not os.path.exists(intro_bgm_file):
                intro_bgm_file = os.path.join(cfg["assets_bgm_dir"], "tongyong.MP3")
        
        # 获取BGM长度调整策略
        bgm_strategy = get_bgm_length_strategy(intro_style)
    
    logger.info(f"片头背景音乐文件: {intro_bgm_file}, 存在: {os.path.exists(intro_bgm_file)}")
    
    loop_crossfade_ms = get_loop_crossfade_ms()
    logger.info(f"BGM长度调整策略: {bgm_strategy}, 循环交叉淡化: {loop_crossfade_ms}ms")
    
    out_mp3 = os.path.join(cfg["output_dir"], "podcast_final.mp3")
    
    # 使用动态片头合成
    try:
        export_with_dynamic_intro(
            final_audio, 
            intro_voice, 
            intro_bgm_file if os.path.exists(intro_bgm_file) else None, 
            out_mp3,
            bgm_strategy=bgm_strategy,
            loop_crossfade_ms=loop_crossfade_ms
        )
        logger.info(f"片头合成成功: {out_mp3}")
    except Exception as e:
        logger.error(f"动态片头合成失败: {e}, 回退到旧方式")
        # 回退到旧的方式
        try:
            export_with_intro(final_audio, out_mp3, intro_path=intro_bgm_file if os.path.exists(intro_bgm_file) else None)
        except Exception:
            mix_intro_with_voice(intro_bgm_file if os.path.exists(intro_bgm_file) else None, voice_path, out_mp3)
    
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


def run_end_to_end(mode: str, topic_or_url_or_text: str, style: str = "chat", custom_style: Optional[str] = None,
                   intro_style: str = "serious", speed: int = 0,
                   voice_a: Optional[str] = None, voice_b: Optional[str] = None, instruction: Optional[str] = None,
                   file_titles: Optional[List[str]] = None, pdf_documents: Optional[List[Dict[str, Any]]] = None,
                   host_mode: str = "dual", custom_intro_script: Optional[str] = None,
                   custom_intro_bgm: Optional[str] = None) -> Dict[str, Any]:
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
    script_res = build_outline_and_script(cfg, topic, sources, style=style, custom_style=custom_style,
                                   instruction=instruction, mode=mode, original_input=topic_or_url_or_text,
                                   instruction_analysis=instruction_analysis, host_mode=host_mode)
    audio_path, transcript_path = tts_and_mix(cfg, script_res["script"], intro_style=intro_style, speed=speed,
                                              voice_a=voice_a, voice_b=voice_b, host_mode=host_mode,
                                              custom_intro_script=custom_intro_script,
                                              custom_intro_bgm=custom_intro_bgm)
    return {
        "audio_path": audio_path,
        "transcript_path": transcript_path,
        "sources": sources,
        "script": script_res["script"],
        "host_mode": host_mode,
    }


def generate_script_only(mode: str, topic_or_url_or_text: str, style: str = "chat", custom_style: Optional[str] = None,
                         instruction: Optional[str] = None, host_mode: str = "dual",
                         file_titles: Optional[List[str]] = None, pdf_documents: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    第一阶段：只生成脚本，不合成语音
    双人播客脚本格式：每行以 A: 或 B: 开头
    """
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
                file_titles=file_titles
            )
            logger.info(f"指令分析结果: {instruction_analysis}")
        except Exception as e:
            logger.error(f"指令分析失败: {e}")
    
    # 获取资料
    if mode == "query":
        topic = topic_or_url_or_text
        sources = retrieve_sources(cfg, "query", query=topic, instruction=instruction, instruction_analysis=instruction_analysis)
    elif mode == "url":
        topic = topic_or_url_or_text
        sources = retrieve_sources(cfg, "url", url=topic, instruction=instruction, instruction_analysis=instruction_analysis)
    else:
        if instruction and "主题：" in instruction:
            topic_match = re.search(r"主题：(.+)(?:\n|$)", instruction)
            if topic_match:
                topic = topic_match.group(1).strip()
            else:
                content_preview = topic_or_url_or_text[:50].replace("\n", " ").strip()
                topic = content_preview + "..." if len(topic_or_url_or_text) > 50 else content_preview
        else:
            content_preview = topic_or_url_or_text[:50].replace("\n", " ").strip()
            topic = content_preview + "..." if len(topic_or_url_or_text) > 50 else content_preview
        
        sources = retrieve_sources(cfg, "doc", doc_text=topic_or_url_or_text, instruction=instruction, 
                                   instruction_analysis=instruction_analysis, pdf_documents=pdf_documents)
    
    # 生成脚本
    script_res = build_outline_and_script(cfg, topic, sources, style=style, custom_style=custom_style,
                                          instruction=instruction, mode=mode, original_input=topic_or_url_or_text,
                                          instruction_analysis=instruction_analysis, host_mode=host_mode)
    
    script = script_res.get("script", "")
    
    # 双人播客：为每行添加 A: 或 B: 标签
    if host_mode == "dual" and script:
        lines = script.strip().split('\n')
        labeled_lines = []
        for line in lines:
            line = line.strip()
            if line:
                # 检查是否已有标签
                if not line.startswith('A:') and not line.startswith('B:'):
                    # 用已添加的行数来判断奇偶，确保交替
                    label = 'A' if (len(labeled_lines) % 2 == 0) else 'B'
                    labeled_lines.append(f"{label}: {line}")
                else:
                    labeled_lines.append(line)
        script = '\n'.join(labeled_lines)
    
    return {
        "script": script,
        "sources": sources,
        "host_mode": host_mode,
    }


def synthesize_audio_only(script: str, intro_style: str = "general", speed: int = 0,
                          voice_a: Optional[str] = None, voice_b: Optional[str] = None,
                          host_mode: str = "dual", custom_intro_script: Optional[str] = None,
                          custom_intro_bgm: Optional[str] = None) -> Dict[str, Any]:
    """
    第二阶段：根据脚本合成语音
    双人播客：解析 A:/B: 标签分配音色
    """
    cfg = load_ini()
    
    if cfg.get("tts_provider") != "tencent":
        raise RuntimeError("当前仅启用腾讯云 TTS")
    if not (cfg.get("tencent_secret_id") and cfg.get("tencent_secret_key")):
        raise RuntimeError("请配置腾讯云 TTS 密钥")
    
    # 处理脚本：去除 A:/B: 标签，但记录每行的角色
    lines = script.strip().split('\n')
    clean_lines = []
    roles = []  # 记录每行的角色：'A', 'B', 或 None（单人模式）
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if host_mode == "dual":
            if line.startswith('A:'):
                clean_lines.append(line[2:].strip())
                roles.append('A')
            elif line.startswith('B:'):
                clean_lines.append(line[2:].strip())
                roles.append('B')
            else:
                # 没有标签的行，按顺序交替
                clean_lines.append(line)
                roles.append('A' if len(roles) % 2 == 0 else 'B')
        else:
            # 单人模式：去除可能存在的标签
            if line.startswith('A:') or line.startswith('B:'):
                clean_lines.append(line[2:].strip())
            else:
                clean_lines.append(line)
            roles.append(None)
    
    # 重新组合为纯文本脚本（用于 TTS）
    clean_script = '\n'.join(clean_lines)
    
    # 调用 TTS 合成，传入角色信息
    audio_path, transcript_path = tts_and_mix_with_roles(
        cfg, clean_lines, roles,
        intro_style=intro_style, speed=speed,
        voice_a=voice_a, voice_b=voice_b, host_mode=host_mode,
        custom_intro_script=custom_intro_script,
        custom_intro_bgm=custom_intro_bgm
    )
    
    return {
        "audio_path": audio_path,
        "transcript_path": transcript_path,
    }


def tts_and_mix_with_roles(cfg: Dict[str, Any], lines: List[str], roles: List[Optional[str]],
                           intro_style: str = "serious", speed: int = 0,
                           voice_a: Optional[str] = None, voice_b: Optional[str] = None,
                           host_mode: str = "dual", custom_intro_script: Optional[str] = None,
                           custom_intro_bgm: Optional[str] = None) -> Tuple[str, str]:
    """
    根据角色信息合成语音（支持 A/B 角色标签）
    """
    ensure_dir(cfg["output_dir"])
    
    if not lines:
        raise RuntimeError("脚本为空，无法合成TTS")
    
    segments: List[AudioSegment] = []
    fillers = ["嗯，我们继续。", "好的，接着说。", "下面进入下一段。"]
    vnum_a = _parse_voice(voice_a, cfg.get("voice_role_a", "501006"))
    vnum_b = _parse_voice(voice_b, cfg.get("voice_role_b", "601007"))
    
    # 单人播客使用更小的 limit
    tts_limit = 120 if host_mode == "single" else 220
    
    # 对每行进行 TTS
    for idx, (line, role) in enumerate(zip(lines, roles)):
        # 分段处理长文本
        chunks = _split_for_tts(line, limit=tts_limit)
        
        for chunk in chunks:
            text1 = _sanitize_for_tts(chunk, aggressive=False)
            if not text1.strip():
                continue
            
            # 根据角色选择音色
            if host_mode == "single" or role is None:
                use_voice = vnum_a
            else:
                use_voice = vnum_a if role == 'A' else vnum_b
            
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
    
    # 拼接音频
    final_audio = AudioSegment.silent(duration=100)
    pause = AudioSegment.silent(duration=200)
    for seg in segments:
        final_audio = final_audio.append(seg, crossfade=50).append(pause, crossfade=0)
    
    voice_path = os.path.join(cfg["output_dir"], "podcast_voice.mp3")
    final_audio.export(voice_path, format="mp3", bitrate="192k")

    # 动态生成片头语音
    logger.info(f"[tts_and_mix_with_roles] 开始生成片头，风格: {intro_style}, 主持人模式: {host_mode}, 自定义文案: {bool(custom_intro_script)}")
    intro_voice = generate_intro_voice(cfg, intro_style, speed=speed, 
                                        voice_a=voice_a, voice_b=voice_b, host_mode=host_mode,
                                        custom_script=custom_intro_script)
    
    # 获取片头背景音乐文件路径
    # 如果有自定义BGM，优先使用
    if custom_intro_bgm and os.path.exists(custom_intro_bgm):
        intro_bgm_file = custom_intro_bgm
        # 自定义BGM使用loop策略
        bgm_strategy = "loop"
        logger.info(f"[tts_and_mix_with_roles] 使用自定义片头BGM: {intro_bgm_file}")
    else:
        intro_bgm_dir = os.path.join(os.path.dirname(cfg["assets_bgm_dir"]), "intro_bgm")
        bgm_filename = get_intro_bgm_filename(intro_style)
        intro_bgm_file = os.path.join(intro_bgm_dir, bgm_filename)
        
        # 如果新目录不存在或文件不存在，回退到旧目录
        if not os.path.exists(intro_bgm_file):
            old_bgm_map = {
                "tech": "kejigan.mp3",
                "business": "shangye.mp3",
                "growth": "chengzhang.mp3",
                "entertainment": "yingshi.mp3",
                "general": "tongyong.MP3",
            }
            old_filename = old_bgm_map.get(intro_style, "tongyong.MP3")
            intro_bgm_file = os.path.join(cfg["assets_bgm_dir"], old_filename)
            
            if not os.path.exists(intro_bgm_file):
                intro_bgm_file = os.path.join(cfg["assets_bgm_dir"], "tongyong.MP3")
        
        # 获取BGM长度调整策略
        bgm_strategy = get_bgm_length_strategy(intro_style)
    
    loop_crossfade_ms = get_loop_crossfade_ms()
    logger.info(f"[tts_and_mix_with_roles] BGM长度调整策略: {bgm_strategy}, 循环交叉淡化: {loop_crossfade_ms}ms")
    
    out_mp3 = os.path.join(cfg["output_dir"], "podcast_final.mp3")
    
    try:
        export_with_dynamic_intro(
            final_audio, 
            intro_voice, 
            intro_bgm_file if os.path.exists(intro_bgm_file) else None, 
            out_mp3,
            bgm_strategy=bgm_strategy,
            loop_crossfade_ms=loop_crossfade_ms
        )
    except Exception as e:
        logger.error(f"动态片头合成失败: {e}, 回退到旧方式")
        try:
            export_with_intro(final_audio, out_mp3, intro_path=intro_bgm_file if os.path.exists(intro_bgm_file) else None)
        except Exception:
            mix_intro_with_voice(intro_bgm_file if os.path.exists(intro_bgm_file) else None, voice_path, out_mp3)
    
    transcript_path = os.path.join(cfg["output_dir"], "podcast_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write('\n'.join(lines))
    
    return out_mp3, transcript_path


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

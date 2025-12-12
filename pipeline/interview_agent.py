"""
Interview Agent for Interview Podcast Mode.

This module implements the interview-style podcast creation workflow where an AI
interviewer helps users express their ideas through multi-turn conversation.

**Feature: interview-podcast-mode**
"""
import re
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from clients.hunyuan_api_client import HunyuanAPIClient
from clients.bocha_client import BochaClient
from utils.config_loader import load_ini
from utils.enhanced_url_fetcher import fetch_url_enhanced
from utils.pdf_loader import extract_text_from_pdf

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class InterviewSession:
    """采访会话数据结构"""
    session_id: str
    created_at: datetime
    updated_at: datetime = field(default_factory=datetime.now)
    
    # 对话历史
    messages: List[Dict[str, Any]] = field(default_factory=list)
    
    # 用户观点
    key_points: List[Dict[str, Any]] = field(default_factory=list)
    
    # 素材
    materials: List[Dict[str, Any]] = field(default_factory=list)
    
    # 用户风格分析
    user_style: Dict[str, Any] = field(default_factory=dict)


# In-memory session storage
_sessions: Dict[str, InterviewSession] = {}


# URL detection regex pattern
URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+|'
    r'www\.[^\s<>"{}|\\^`\[\]]+'
)


def detect_urls(text: str) -> List[str]:
    """
    Detect URLs in a text message.
    
    Args:
        text: The text to search for URLs
        
    Returns:
        List of detected URLs
    """
    if not text:
        return []
    return URL_PATTERN.findall(text)


class InterviewAgent:
    """采访智能体 - 管理对话、搜索、风格分析"""
    
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        Initialize the interview agent.
        
        Args:
            cfg: Configuration dictionary. If None, loads from config file.
        """
        self.cfg = cfg or load_ini()
    
    def _get_llm_client(self) -> HunyuanAPIClient:
        """Get configured LLM client for interview conversations."""
        return HunyuanAPIClient(
            secret_id=self.cfg["hunyuan_api_secret_id"],
            secret_key=self.cfg["hunyuan_api_secret_key"],
            region=self.cfg["hunyuan_api_region"],
            model=self.cfg.get("hunyuan_api_model", "hunyuan-turbos-latest"),
            temperature=self.cfg.get("hunyuan_api_temperature", 0.8),
            top_p=self.cfg.get("hunyuan_api_top_p", 0.8),
            max_tokens=self.cfg.get("hunyuan_api_max_tokens", 2000),
        )
    
    def start_session(self) -> InterviewSession:
        """
        创建新的采访会话
        
        Returns:
            InterviewSession: 新创建的会话对象
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = InterviewSession(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            messages=[],
            key_points=[],
            materials=[],
            user_style={}
        )
        
        _sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[InterviewSession]:
        """
        获取会话状态
        
        Args:
            session_id: 会话ID
            
        Returns:
            InterviewSession if found, None otherwise
        """
        return _sessions.get(session_id)
    
    def chat(self, session_id: str, message: str, attached_material_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        处理用户消息，返回 AI 回复
        
        Args:
            session_id: 会话ID
            message: 用户消息
            attached_material_ids: 附加的素材ID列表（对话模式下用户引用的素材）
            
        Returns:
            {
                "reply": str,           # AI 回复
                "key_points": List[str], # 更新后的观点列表
                "detected_urls": List[str],  # 检测到的 URL
                "message_count": int    # 当前消息数量
            }
            
        Raises:
            ValueError: If session not found
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        # Detect URLs in the message
        detected_urls = detect_urls(message)
        
        # 获取附加素材的内容
        attached_materials = []
        if attached_material_ids:
            for material_id in attached_material_ids:
                for m in session.materials:
                    if m.get("id") == material_id:
                        attached_materials.append(m)
                        break
        
        # Add user message to history
        user_msg = {
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "detected_urls": detected_urls,
            "search_triggered": False,
            "attached_materials": [m.get("id") for m in attached_materials]
        }
        session.messages.append(user_msg)
        
        # Build conversation history for LLM (with attached materials context)
        llm_messages = self._build_llm_messages(session, attached_materials)
        
        # Get AI response
        client = self._get_llm_client()
        resp = client.chat(llm_messages, stream=False)
        
        # Extract reply content
        reply = ""
        try:
            choices = resp.get("Choices") or resp.get("choices") or []
            if choices:
                msg = choices[0].get("Message") or choices[0].get("message") or {}
                reply = msg.get("Content") or msg.get("content") or ""
        except Exception:
            reply = "抱歉，我遇到了一些问题。请再试一次。"
        
        # Add assistant message to history
        assistant_msg = {
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.now().isoformat(),
            "detected_urls": [],
            "search_triggered": False
        }
        session.messages.append(assistant_msg)
        
        # Extract key points from user message
        self._extract_key_points(session, message)
        
        # Update session timestamp
        session.updated_at = datetime.now()
        
        return {
            "reply": reply,
            "key_points": [kp["point"] for kp in session.key_points],
            "detected_urls": detected_urls,
            "message_count": len(session.messages)
        }
    
    def _build_llm_messages(self, session: InterviewSession, attached_materials: Optional[List[Dict]] = None) -> List[Dict[str, str]]:
        """Build message list for LLM API call."""
        system_prompt = """你是一位专业的播客采访者，正在帮助用户创作播客内容。你的任务是：

1. 通过提问引导用户表达他们的想法和观点
2. 对用户的回答表示理解和认可，然后深入追问
3. 帮助用户组织和完善他们的思路
4. 在适当的时候总结用户的核心观点

采访风格：
- 友好、专业、有同理心
- 提问要具体，避免泛泛而谈
- 每次只问一个问题，让用户有充分表达的空间
- 适时给予鼓励和肯定

如果用户分享了URL、文档或话题素材，请仔细阅读素材内容，结合用户的问题进行深入讨论。"""

        messages = [{"Role": "system", "Content": system_prompt}]
        
        # Add conversation history (except the last user message which we'll handle separately)
        for msg in session.messages[:-1]:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({
                "Role": role,
                "Content": msg["content"]
            })
        
        # Handle the last user message with attached materials
        if session.messages:
            last_msg = session.messages[-1]
            if last_msg["role"] == "user":
                user_content = last_msg["content"]
                
                # If there are attached materials, prepend their content
                if attached_materials:
                    materials_context = self._format_materials_context(attached_materials)
                    user_content = f"{materials_context}\n\n用户消息：{user_content}"
                
                messages.append({
                    "Role": "user",
                    "Content": user_content
                })
        
        return messages
    
    def _format_materials_context(self, materials: List[Dict]) -> str:
        """Format attached materials as context for LLM."""
        if not materials:
            return ""
        
        parts = ["【用户引用的素材】"]
        for i, m in enumerate(materials, 1):
            m_type = m.get("type", "unknown")
            m_source = m.get("source", "")
            m_summary = m.get("summary", "")
            m_content = m.get("content", "")[:3000]  # Limit content length
            
            type_label = {"url": "链接", "document": "文档", "topic": "话题"}.get(m_type, "素材")
            parts.append(f"\n素材{i} ({type_label}): {m_source}")
            if m_summary:
                parts.append(f"摘要: {m_summary}")
            if m_content and m_content != m_source:
                parts.append(f"内容:\n{m_content}")
        
        return "\n".join(parts)
    
    def _extract_key_points(self, session: InterviewSession, message: str) -> None:
        """Extract key points from user message and add to session."""
        # Simple heuristic: messages longer than 50 chars likely contain opinions
        if len(message.strip()) > 50:
            # Check if this point is already captured (avoid duplicates)
            existing_points = [kp["point"] for kp in session.key_points]
            
            # Create a summary point (first 100 chars as placeholder)
            # In production, this would use LLM to extract actual key points
            point_summary = message[:100] + ("..." if len(message) > 100 else "")
            
            if point_summary not in existing_points:
                session.key_points.append({
                    "point": point_summary,
                    "source_message_idx": len(session.messages) - 1,
                    "confidence": 0.8
                })
    
    def add_material(self, session_id: str, material_type: str, content: str) -> Dict[str, Any]:
        """
        添加素材（URL/文档/话题）
        
        Args:
            session_id: 会话ID
            material_type: "url" | "document" | "topic"
            content: URL 地址 / 文档文本或文件路径 / 话题关键词
        
        Returns:
            {
                "id": str,           # 素材ID
                "summary": str,      # 素材摘要
                "source": str,       # 来源
                "ai_thoughts": str   # AI 对素材的思考
            }
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        material_id = str(uuid.uuid4())[:8]
        
        try:
            if material_type == "url":
                result = self._add_url_material(content)
            elif material_type == "document":
                result = self._add_document_material(content)
            elif material_type == "topic":
                result = self._add_topic_material(content)
            else:
                raise ValueError(f"Unknown material type: {material_type}")
        except Exception as e:
            logger.error(f"Failed to process material: {e}")
            # 返回一个基本的结果，而不是抛出异常
            result = {
                "summary": f"素材处理失败: {str(e)[:100]}",
                "source": content if material_type == "url" else f"{material_type}: {content[:50]}",
                "ai_thoughts": "抱歉，处理这个素材时遇到了问题。你可以尝试重新添加，或者直接在对话中描述相关内容。",
                "raw_content": content
            }
        
        summary = result.get("summary", "")
        source = result.get("source", "")
        ai_thoughts = result.get("ai_thoughts", "")
        raw_content = result.get("raw_content", content)
        
        material = {
            "id": material_id,
            "type": material_type,
            "content": raw_content,
            "summary": summary,
            "source": source,
            "ai_thoughts": ai_thoughts,
            "added_at": datetime.now().isoformat()
        }
        
        session.materials.append(material)
        session.updated_at = datetime.now()
        
        return {
            "id": material_id,
            "summary": summary,
            "source": source,
            "ai_thoughts": ai_thoughts
        }
    
    def _add_url_material(self, url: str) -> Dict[str, Any]:
        """
        Process URL material using enhanced URL fetcher.
        
        Args:
            url: The URL to fetch and analyze
            
        Returns:
            Dict with summary, source, ai_thoughts, and raw_content
        """
        logger.info(f"Fetching URL content: {url}")
        
        # Use enhanced URL fetcher to get content
        fetch_result = fetch_url_enhanced(url)
        
        if not fetch_result.get("success") or not fetch_result.get("text"):
            logger.warning(f"Failed to fetch URL: {url}")
            return {
                "summary": f"无法获取URL内容: {url}",
                "source": url,
                "ai_thoughts": "抱歉，我无法访问这个链接。请检查URL是否正确，或者尝试复制粘贴网页内容。",
                "raw_content": url
            }
        
        text_content = fetch_result["text"]
        
        # Generate summary using LLM
        summary = self._generate_content_summary(text_content, "网页")
        
        # Generate AI thoughts about the content
        ai_thoughts = self._generate_ai_thoughts(text_content, "url", url)
        
        return {
            "summary": summary,
            "source": url,
            "ai_thoughts": ai_thoughts,
            "raw_content": text_content[:5000]  # Limit stored content
        }
    
    def _add_document_material(self, content: str) -> Dict[str, Any]:
        """
        Process document material. Content can be:
        - A file path to a PDF
        - Raw text content
        
        Args:
            content: File path or raw text content
            
        Returns:
            Dict with summary, source, ai_thoughts, and raw_content
        """
        import os
        
        text_content = ""
        source = "用户上传文档"
        
        # Check if content is a file path
        if content and os.path.exists(content):
            if content.lower().endswith('.pdf'):
                logger.info(f"Extracting text from PDF: {content}")
                text_content = extract_text_from_pdf(content)
                source = f"PDF文档: {os.path.basename(content)}"
            else:
                # Try to read as text file
                try:
                    with open(content, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    source = f"文档: {os.path.basename(content)}"
                except Exception as e:
                    logger.warning(f"Failed to read file: {e}")
                    text_content = content
        else:
            # Treat as raw text content
            text_content = content
        
        if not text_content or not text_content.strip():
            return {
                "summary": "文档内容为空或无法解析",
                "source": source,
                "ai_thoughts": "抱歉，我无法从这个文档中提取有效内容。请尝试其他格式或直接粘贴文本。",
                "raw_content": ""
            }
        
        # Generate summary using LLM
        summary = self._generate_content_summary(text_content, "文档")
        
        # Generate AI thoughts about the content
        ai_thoughts = self._generate_ai_thoughts(text_content, "document", source)
        
        return {
            "summary": summary,
            "source": source,
            "ai_thoughts": ai_thoughts,
            "raw_content": text_content[:5000]  # Limit stored content
        }
    
    def _add_topic_material(self, topic: str) -> Dict[str, Any]:
        """
        Process topic material using Bocha search API.
        
        Args:
            topic: The topic keyword to search
            
        Returns:
            Dict with summary, source, ai_thoughts, and raw_content
        """
        logger.info(f"Searching topic: {topic}")
        
        search_results = []
        
        try:
            # Initialize Bocha client
            bocha_client = BochaClient(
                base_url=self.cfg.get("bocha_base_url", ""),
                api_id=self.cfg.get("bocha_api_id", ""),
                api_key=self.cfg.get("bocha_api_key", "")
            )
            
            # Search for the topic
            search_results = bocha_client.search(topic, count=5)
            logger.info(f"Found {len(search_results)} search results for topic: {topic}")
            
        except Exception as e:
            logger.warning(f"Search failed for topic '{topic}': {e}")
        
        if not search_results:
            return {
                "summary": f"关于「{topic}」的搜索未找到相关结果",
                "source": f"话题搜索: {topic}",
                "ai_thoughts": f"抱歉，我没有找到关于「{topic}」的相关信息。我们可以基于你自己的了解来讨论这个话题。",
                "raw_content": ""
            }
        
        # Format search results
        formatted_results = []
        for i, result in enumerate(search_results[:5], 1):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            url = result.get("url", "")
            formatted_results.append(f"{i}. {title}\n   {snippet}\n   来源: {url}")
        
        raw_content = "\n\n".join(formatted_results)
        
        # Generate summary of search results
        summary = self._generate_search_summary(search_results, topic)
        
        # Generate AI thoughts about the topic
        ai_thoughts = self._generate_ai_thoughts(raw_content, "topic", topic)
        
        return {
            "summary": summary,
            "source": f"话题搜索: {topic}",
            "ai_thoughts": ai_thoughts,
            "raw_content": raw_content
        }
    
    def _generate_content_summary(self, content: str, content_type: str) -> str:
        """
        Generate a summary of the content using LLM.
        
        Args:
            content: The text content to summarize
            content_type: Type description (e.g., "网页", "文档")
            
        Returns:
            Summary string
        """
        # Truncate content if too long
        truncated = content[:3000] if len(content) > 3000 else content
        
        prompt = f"""请用2-3句话简洁概括以下{content_type}的主要内容：

{truncated}

请直接输出摘要，不需要任何前缀或格式标记。"""

        try:
            client = self._get_llm_client()
            messages = [
                {"Role": "system", "Content": "你是一个专业的内容摘要助手，擅长提取文本的核心信息。"},
                {"Role": "user", "Content": prompt}
            ]
            
            resp = client.chat(messages, stream=False)
            choices = resp.get("Choices") or resp.get("choices") or []
            if choices:
                msg = choices[0].get("Message") or choices[0].get("message") or {}
                return msg.get("Content") or msg.get("content") or ""
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
        
        # Fallback: return first 200 chars
        return truncated[:200] + ("..." if len(truncated) > 200 else "")
    
    def _generate_search_summary(self, results: List[Dict], topic: str) -> str:
        """
        Generate a summary of search results.
        
        Args:
            results: List of search result dicts
            topic: The search topic
            
        Returns:
            Summary string
        """
        if not results:
            return f"未找到关于「{topic}」的相关信息"
        
        # Format results for LLM
        formatted = []
        for r in results[:5]:
            formatted.append(f"- {r.get('title', '')}: {r.get('snippet', '')}")
        
        results_text = "\n".join(formatted)
        
        prompt = f"""基于以下关于「{topic}」的搜索结果，用2-3句话概括主要发现：

{results_text}

请直接输出摘要，不需要任何前缀或格式标记。"""

        try:
            client = self._get_llm_client()
            messages = [
                {"Role": "system", "Content": "你是一个专业的信息整理助手，擅长从搜索结果中提取关键信息。"},
                {"Role": "user", "Content": prompt}
            ]
            
            resp = client.chat(messages, stream=False)
            choices = resp.get("Choices") or resp.get("choices") or []
            if choices:
                msg = choices[0].get("Message") or choices[0].get("message") or {}
                return msg.get("Content") or msg.get("content") or ""
        except Exception as e:
            logger.warning(f"Failed to generate search summary: {e}")
        
        # Fallback
        return f"找到{len(results)}条关于「{topic}」的相关结果"
    
    def _generate_ai_thoughts(self, content: str, material_type: str, source: str) -> str:
        """
        Generate AI thoughts/insights about the material.
        
        Args:
            content: The material content
            material_type: Type of material (url, document, topic)
            source: Source description
            
        Returns:
            AI thoughts string
        """
        # Truncate content if too long
        truncated = content[:2000] if len(content) > 2000 else content
        
        type_desc = {
            "url": "网页内容",
            "document": "文档内容",
            "topic": "搜索结果"
        }.get(material_type, "内容")
        
        prompt = f"""作为播客采访者，基于以下{type_desc}，分享你的一个有趣观察或思考，用于引导后续对话：

来源: {source}
内容:
{truncated}

请用1-2句话分享你的想法，语气要自然、有启发性。直接输出，不需要任何前缀。"""

        try:
            client = self._get_llm_client()
            messages = [
                {"Role": "system", "Content": "你是一位专业的播客采访者，善于从素材中发现有趣的角度和话题。"},
                {"Role": "user", "Content": prompt}
            ]
            
            resp = client.chat(messages, stream=False)
            choices = resp.get("Choices") or resp.get("choices") or []
            if choices:
                msg = choices[0].get("Message") or choices[0].get("message") or {}
                return msg.get("Content") or msg.get("content") or ""
        except Exception as e:
            logger.warning(f"Failed to generate AI thoughts: {e}")
        
        # Fallback responses
        fallbacks = {
            "url": "这个链接的内容很有意思，我们可以深入讨论其中的观点。",
            "document": "这份文档提供了很好的背景信息，可以帮助我们更好地展开讨论。",
            "topic": "关于这个话题，我找到了一些有趣的信息，可以作为我们讨论的参考。"
        }
        return fallbacks.get(material_type, "这个素材可以帮助丰富我们的讨论。")
    
    def analyze_style(self, session_id: str) -> Dict[str, Any]:
        """
        分析用户说话风格
        
        Returns:
            {
                "tone": str,           # 语气（轻松/严肃/幽默等）
                "vocabulary": List[str], # 常用词汇
                "expressions": List[str], # 特色表达
                "sentence_style": str   # 句式特点
            }
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        # Collect user messages
        user_messages = [
            msg["content"] for msg in session.messages 
            if msg["role"] == "user"
        ]
        
        if len(user_messages) < 5:
            # Not enough data for style analysis
            return {
                "tone": "待分析",
                "vocabulary": [],
                "expressions": [],
                "sentence_style": "需要更多对话内容"
            }
        
        # Use LLM to analyze style
        combined_text = "\n".join(user_messages)
        
        style_prompt = f"""分析以下用户对话内容的说话风格，返回JSON格式：

用户对话内容：
{combined_text}

请分析并返回以下格式的JSON（不要包含markdown标记）：
{{
    "tone": "语气特点（如：轻松、严肃、幽默、热情等）",
    "vocabulary": ["常用词汇1", "常用词汇2", "常用词汇3"],
    "expressions": ["特色表达1", "特色表达2"],
    "sentence_style": "句式特点描述"
}}"""

        client = self._get_llm_client()
        messages = [
            {"Role": "system", "Content": "你是一个语言风格分析专家，擅长分析用户的说话方式和表达习惯。"},
            {"Role": "user", "Content": style_prompt}
        ]
        
        resp = client.chat(messages, stream=False)
        
        try:
            choices = resp.get("Choices") or resp.get("choices") or []
            if choices:
                msg = choices[0].get("Message") or choices[0].get("message") or {}
                content = msg.get("Content") or msg.get("content") or ""
                
                # Try to parse JSON from response
                import json
                # Remove potential markdown code blocks
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                
                style_data = json.loads(content)
                session.user_style = style_data
                session.updated_at = datetime.now()
                return style_data
        except Exception:
            pass
        
        # Default response if parsing fails
        default_style = {
            "tone": "自然",
            "vocabulary": [],
            "expressions": [],
            "sentence_style": "口语化"
        }
        session.user_style = default_style
        return default_style
    
    def generate_script(self, session_id: str, host_mode: str = "dual") -> Dict[str, Any]:
        """
        根据会话内容生成播客脚本
        
        Args:
            session_id: 会话ID
            host_mode: 主持人模式，"dual" 为双人播客（A/B交替），"single" 为单人播客
        
        Returns:
            {
                "script": str,        # 生成的脚本
                "sources": List[Dict], # 引用来源
                "style_applied": Dict,  # 应用的风格特征
                "warning": Optional[str]  # 警告信息（如对话太短）
                "host_mode": str      # 主持人模式
            }
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        warning = None
        user_message_count = len([m for m in session.messages if m["role"] == "user"])
        
        # Check for short conversation
        if user_message_count < 3:
            warning = "对话轮数较少，建议继续交流以获得更丰富的内容。"
        
        # Analyze style if not done yet
        if not session.user_style:
            self.analyze_style(session_id)
        
        # Build script generation prompt
        key_points_text = "\n".join([f"- {kp['point']}" for kp in session.key_points])
        materials_text = "\n".join([
            f"- [{m['type']}] {m['summary']} (来源: {m['source']})"
            for m in session.materials
        ])
        
        conversation_text = "\n".join([
            f"{'用户' if m['role'] == 'user' else 'AI'}: {m['content']}"
            for m in session.messages
        ])
        
        style_desc = ""
        if session.user_style:
            style_desc = f"""
用户风格特点：
- 语气：{session.user_style.get('tone', '自然')}
- 常用词汇：{', '.join(session.user_style.get('vocabulary', []))}
- 特色表达：{', '.join(session.user_style.get('expressions', []))}
- 句式特点：{session.user_style.get('sentence_style', '口语化')}
"""

        # 根据 host_mode 设置不同的脚本格式要求
        if host_mode == "dual":
            format_instruction = """
【脚本格式要求 - 双人访谈播客】
这是一个访谈形式的双人播客，A 是主持人，B 是嘉宾（就是用户本人）。

角色设定：
- A（主持人）：负责开场、引导话题、提问、总结。语气专业、友好，帮助 B 更好地表达观点。
- B（嘉宾/用户）：分享自己的观点、经验、故事。使用第一人称（"我觉得..."、"我的经验是..."），保留用户的个人风格。

格式要求：
- 每一段对话必须以 "A:" 或 "B:" 开头
- A 和 B 交替发言，形成自然的访谈对话
- B 的发言应该基于用户在对话中表达的真实观点
- 不要出现第三方"嘉宾"的概念，B 就是嘉宾本人

格式示例：
A: 欢迎收听本期播客！今天我们邀请到一位特别的嘉宾，来聊聊他在这个领域的亲身经历。
B: 大家好！很高兴能来分享我的一些想法和经验。
A: 能先简单介绍一下你的背景吗？
B: 好的，我是一名研究生，主要研究方向是人工智能..."""
        else:
            format_instruction = """
【脚本格式要求 - 单人分享播客】
这是一个单人分享形式的播客，以用户（"我"）的第一人称视角进行独白分享。

角色设定：
- 唯一角色就是用户本人，以第一人称"我"来分享观点和经历
- AI 帮助整理、丰富、组织用户在对话中表达的内容
- 语气自然、真诚，像是在和朋友分享

格式要求：
- 不需要添加任何角色标签（不要 A: 或 B:）
- 保持连贯的独白风格
- 使用第一人称（"我觉得"、"我的经验是"、"在我看来"）
- 内容应该反映用户的真实观点和个人风格

格式示例：
大家好，今天想和大家聊聊AI是怎么改变我的日常工作的。

作为一个研究生，我每天都要处理大量的文献和代码。说实话，AI在这方面给我的帮助真的很大..."""

        script_prompt = f"""基于以下采访对话内容，生成一段播客脚本。

【对话记录】
{conversation_text}

【用户核心观点】
{key_points_text if key_points_text else "暂无明确观点"}

【参考素材】
{materials_text if materials_text else "暂无参考素材"}

{style_desc}

{format_instruction}

【内容要求】
1. 脚本必须体现用户在对话中表达的真实观点和个人风格
2. 如果有参考素材，请在适当位置自然地引用
3. 保持口语化、自然的表达方式，像真实的播客对话
4. 脚本长度适中，约500-1000字
5. 严格按照上述格式要求输出，不要混淆角色

请直接输出脚本内容。双人模式下确保每行都以 A: 或 B: 开头，单人模式下不要添加任何角色标签。"""

        client = self._get_llm_client()
        messages = [
            {"Role": "system", "Content": "你是一个专业的播客脚本编剧，擅长将对话内容转化为生动的播客脚本。你必须严格按照指定的格式输出脚本。"},
            {"Role": "user", "Content": script_prompt}
        ]
        
        resp = client.chat(messages, stream=False)
        
        script = ""
        try:
            choices = resp.get("Choices") or resp.get("choices") or []
            if choices:
                msg = choices[0].get("Message") or choices[0].get("message") or {}
                script = msg.get("Content") or msg.get("content") or ""
        except Exception:
            script = "脚本生成失败，请重试。"
        
        # 后处理：确保双人播客脚本格式正确
        if host_mode == "dual" and script:
            script = self._ensure_dual_host_format(script)
        
        # Build sources list
        sources = [
            {"type": m["type"], "source": m["source"], "summary": m["summary"]}
            for m in session.materials
        ]
        
        result = {
            "script": script,
            "sources": sources,
            "style_applied": session.user_style,
            "host_mode": host_mode
        }
        
        if warning:
            result["warning"] = warning
        
        return result
    
    def _ensure_dual_host_format(self, script: str) -> str:
        """
        确保双人播客脚本的每一行都有正确的 A:/B: 标签。
        
        Args:
            script: 原始脚本
            
        Returns:
            格式化后的脚本，每行都以 A: 或 B: 开头
        """
        lines = script.strip().split('\n')
        labeled_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否已有正确的标签
            if line.startswith('A:') or line.startswith('B:'):
                labeled_lines.append(line)
            elif line.startswith('A：') or line.startswith('B：'):
                # 处理中文冒号的情况
                labeled_lines.append('A:' + line[2:] if line.startswith('A') else 'B:' + line[2:])
            else:
                # 没有标签的行，按顺序交替添加
                label = 'A' if (len(labeled_lines) % 2 == 0) else 'B'
                labeled_lines.append(f"{label}: {line}")
        
        return '\n'.join(labeled_lines)


# Module-level functions for convenience
def start_session() -> InterviewSession:
    """Create a new interview session."""
    agent = InterviewAgent()
    return agent.start_session()


def get_session(session_id: str) -> Optional[InterviewSession]:
    """Get an existing session by ID."""
    return _sessions.get(session_id)


def clear_sessions() -> None:
    """Clear all sessions (for testing)."""
    _sessions.clear()

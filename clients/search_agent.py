"""
智能搜索代理，根据主题和指令生成更精确的搜索查询
"""
from typing import Dict, Any, List, Optional
from clients.hunyuan_api_client import HunyuanAPIClient


class SearchAgent:
    """
    智能搜索代理，用于根据主题和指令生成更精确的搜索查询
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        初始化搜索代理
        
        参数:
            cfg: 配置信息，包含API密钥等
        """
        self.api = HunyuanAPIClient(
            secret_id=cfg["hunyuan_api_secret_id"],
            secret_key=cfg["hunyuan_api_secret_key"],
            region=cfg["hunyuan_api_region"],
            model=cfg["hunyuan_api_model"],
            temperature=0.3,  # 使用较低温度以获得确定性结果
            top_p=0.9,
            max_tokens=100,
        )
    
    def generate_search_query(self, topic: str, instruction: Optional[str] = None, search_focus: List[str] = None) -> str:
        """
        根据主题、指令和搜索重点生成更精确的搜索查询
        
        参数:
            topic: 原始主题
            instruction: 特殊指令
            search_focus: 指令分析器提取的搜索重点关键词
            
        返回:
            优化后的搜索查询
        """
        if not instruction and not search_focus:
            return topic
        
        # 准备搜索重点内容
        focus_content = ""
        if search_focus and len(search_focus) > 0:
            focus_content = f"\n\n搜索重点: {', '.join(search_focus)}"
            
        prompt = f"""
作为一个搜索优化专家，你的任务是根据用户的主题、特殊指令和搜索重点，生成一个更精确的搜索查询。
这个查询将用于获取相关信息，以生成播客内容。

原始主题: {topic}
特殊指令: {instruction or ''}{"" if not instruction else ''}{focus_content}

请分析特殊指令和搜索重点中的关键要求，并将其融入到搜索查询中，使搜索结果能够满足这些特殊要求。
例如，如果指令要求“提到具体的电影名称”，你应该在查询中包含“2023-2025热门电影排行榜”等内容。

请直接输出优化后的搜索查询，不要包含任何解释或其他内容。查询应该是一个简洁的短语或句子。
        """
        
        messages = [
            {"Role": "system", "Content": "你是一个专业的搜索查询优化专家，擅长生成精确的搜索查询"},
            {"Role": "user", "Content": prompt},
        ]
        
        try:
            resp = self.api.chat(messages, stream=False)
            content = ""
            choices = resp.get("Choices") or resp.get("choices") or []
            if choices:
                msg = choices[0].get("Message") or choices[0].get("message") or {}
                content = msg.get("Content") or msg.get("content") or ""
            
            # 清理结果，确保是一个简洁的查询
            query = content.strip().replace('"', '').replace("'", "")
            if query:
                return query
            return topic
        except Exception as e:
            print(f"搜索查询生成失败: {e}")
            return topic

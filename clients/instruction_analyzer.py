"""
指令分析器
使用LLM分析用户指令，提取关键信息
"""
from typing import Dict, Any, List, Optional
import logging
import re
import json
from clients.hunyuan_api_client import HunyuanAPIClient

# 配置日志
logger = logging.getLogger(__name__)

class InstructionAnalyzer:
    """
    指令分析器
    使用LLM分析用户指令，提取关键信息
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        初始化指令分析器
        
        参数:
            cfg: 配置信息，包含API密钥等
        """
        self.api = HunyuanAPIClient(
            secret_id=cfg["hunyuan_api_secret_id"],
            secret_key=cfg["hunyuan_api_secret_key"],
            region=cfg["hunyuan_api_region"],
            model=cfg["hunyuan_api_model"],
            temperature=0.1,  # 低温度以获得确定性结果
            top_p=0.9,
            max_tokens=200,
        )
    
    def analyze_instruction(self, instruction: str, mode: str, content: str = "", file_titles: List[str] = None) -> Dict[str, Any]:
        """
        分析用户指令，返回结构化的分析结果
        
        参数:
            instruction: 用户指令
            mode: 输入模式 (query, url, doc)
            content: 输入内容 (查询、URL或文档内容)
            file_titles: 上传的文件标题列表（用于多文件场景）
            
        返回:
            包含分析结果的字典，包括：
            - is_english: 是否需要生成英文内容
            - search_focus: 搜索重点关键词
            - content_style: 内容风格偏好
            - other_requirements: 其他特殊要求
        """
        if not instruction:
            return {
                "is_english": False,
                "search_focus": file_titles if file_titles else [],  # 如果有文件列表，默认全部作为搜索重点
                "content_style": "standard",
                "other_requirements": ""
            }
        
        # 构建文件信息描述
        file_info = ""
        if file_titles and len(file_titles) > 0:
            file_info = f"\n上传的文件列表: {', '.join(file_titles)}"
        
        prompt = f"""
分析以下用户指令，提取关键信息并以JSON格式返回结果。

用户输入模式: {mode}
用户输入内容: {content[:100] + '...' if len(content) > 100 else content}{file_info}
用户指令: {instruction}

请分析这个指令并提取以下信息:
1. 是否要求生成英文内容（注意区分"使用英文"和"不要使用英文"等否定表达）
2. 搜索应该关注的关键词或方面（重要：如果有多个上传文件，应该包含所有文件名，确保每个文件都被关注）
3. 内容风格偏好（如正式、轻松、专业、通俗等）
4. 其他特殊要求

以下是一些示例：
- "使用英文生成" -> 需要生成英文内容
- "不要用英文" -> 不需要生成英文内容
- "重点关注经济影响" -> 搜索应关注经济影响
- "风格轻松一点" -> 内容风格偏好为轻松
- 上传了3个文件 -> search_focus应包含所有3个文件名

重要提示：如果用户上传了多个文件，search_focus必须包含所有文件名，以确保生成的内容能够均衡覆盖所有上传的资料。

请以JSON格式返回，格式如下:
```json
{{
  "is_english": true/false,
  "search_focus": ["文件名1", "文件名2", "关键词1", ...],
  "content_style": "formal/casual/professional/popular/...",
  "other_requirements": "其他特殊要求描述"
}}
```
只返回JSON，不要有其他解释。
"""
        
        messages = [
            {"Role": "system", "Content": "你是一个精确的指令分析助手，擅长从用户指令中提取关键信息并以结构化格式返回"},
            {"Role": "user", "Content": prompt},
        ]
        
        try:
            resp = self.api.chat(messages, stream=False)
            content = ""
            choices = resp.get("Choices") or resp.get("choices") or []
            if choices:
                msg = choices[0].get("Message") or choices[0].get("message") or {}
                content = msg.get("Content") or msg.get("content") or ""
            
            # 提取JSON
            # 尝试找到JSON部分
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```|(\{[\s\S]*\})', content)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                try:
                    result = json.loads(json_str)
                    logger.info(f"指令分析成功: {result}")
                    return result
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON解析失败: {e}, 内容: {json_str}")
            
            # 如果无法解析JSON，返回默认值并记录原始响应
            logger.warning(f"无法从响应中提取JSON，原始响应: {content}")
            return self._fallback_analysis(instruction)
            
        except Exception as e:
            logger.error(f"指令分析失败: {e}")
            # 失败时回退到简单的字符串匹配
            return self._fallback_analysis(instruction)
    
    def _fallback_analysis(self, instruction: str) -> Dict[str, Any]:
        """
        当LLM分析失败时的后备分析方法
        使用简单的字符串匹配
        """
        # 检查是否要求英文内容
        english_keywords = ["english", "in english", "generate in english", "use english", "英文", "用英文", "英语", "使用英文", "使用英语"]
        is_english = False
        
        # 检查否定词
        negation_words = ["不", "不要", "别", "don't", "do not", "no", "not"]
        has_negation = any(neg in instruction.lower() for neg in negation_words)
        
        # 如果有英文关键词但没有否定词，则认为需要英文内容
        if not has_negation and any(keyword in instruction.lower() for keyword in english_keywords):
            is_english = True
        
        # 提取可能的搜索重点
        search_focus = []
        focus_indicators = ["关注", "重点", "focus", "concentrate", "emphasize"]
        for indicator in focus_indicators:
            if indicator in instruction.lower():
                # 简单提取关注点后面的内容
                pos = instruction.lower().find(indicator)
                if pos >= 0:
                    remaining = instruction[pos + len(indicator):].strip()
                    if remaining:
                        # 取前30个字符作为搜索重点
                        focus = remaining[:30].strip()
                        if focus:
                            search_focus.append(focus)
        
        # 检测内容风格
        content_style = "standard"
        style_keywords = {
            "formal": ["正式", "严肃", "学术", "专业"],
            "casual": ["轻松", "随意", "休闲", "聊天"],
            "professional": ["专业", "技术", "深入"],
            "popular": ["通俗", "大众", "简单", "易懂"]
        }
        
        for style, keywords in style_keywords.items():
            if any(keyword in instruction.lower() for keyword in keywords):
                content_style = style
                break
        
        return {
            "is_english": is_english,
            "search_focus": search_focus,
            "content_style": content_style,
            "other_requirements": instruction
        }

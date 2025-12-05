"""
提示词自适应调整器
根据输入材料和特殊指令自动调整提示词，以生成适当长度的播客内容
"""
from typing import Dict, Any, List, Optional, Tuple
from clients.hunyuan_api_client import HunyuanAPIClient


class PromptAdjuster:
    """
    提示词自适应调整器
    根据输入材料和特殊指令自动调整提示词，以生成适当长度的播客内容
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        初始化提示词调整器
        
        参数:
            cfg: 配置信息，包含API密钥等
        """
        self.api = HunyuanAPIClient(
            secret_id=cfg["hunyuan_api_secret_id"],
            secret_key=cfg["hunyuan_api_secret_key"],
            region=cfg["hunyuan_api_region"],
            model=cfg["hunyuan_api_model"],
            temperature=0.2,  # 使用低温度以获得确定性结果
            top_p=0.9,
            max_tokens=200,
        )
    
    def analyze_content(self, mode: str, content: str, sources: List[Dict[str, Any]], instruction: Optional[str] = None) -> Dict[str, Any]:
        """
        分析输入内容，确定适合的播客长度和结构
        
        参数:
            mode: 输入模式（query, url, doc）
            content: 输入内容（查询词、URL或文档内容）
            sources: 检索到的资料列表
            instruction: 特殊指令
            
        返回:
            包含建议长度和结构的字典
        """
        # 准备分析材料
        analysis_material = ""
        
        # 添加输入内容
        analysis_material += f"输入模式: {mode}\n"
        analysis_material += f"输入内容: {content[:500]}...\n\n" if len(content) > 500 else f"输入内容: {content}\n\n"
        
        # 添加资料摘要
        analysis_material += f"资料数量: {len(sources)}\n"
        
        # 计算资料总字数
        total_chars = sum(len(s.get("snippet", "")) for s in sources)
        analysis_material += f"资料总字数: {total_chars}\n"
        
        # 添加资料标题列表
        analysis_material += "资料标题列表:\n"
        for i, s in enumerate(sources[:5]):  # 只取前5个资料的标题
            analysis_material += f"- {s.get('title', '无标题')}\n"
        if len(sources) > 5:
            analysis_material += f"...（还有{len(sources)-5}个资料）\n"
        
        # 添加特殊指令
        if instruction:
            analysis_material += f"\n特殊指令: {instruction}\n"
        
        # 检查指令中是否有明确的时长要求
        time_constraint = ""
        if instruction:
            import re
            # 匹配各种时长表达方式
            time_patterns = [
                r'(\d+)\s*分钟以内',
                r'(\d+)\s*分钟内',
                r'不超过\s*(\d+)\s*分钟',
                r'(\d+)\s*minute',
                r'(\d+)-(\d+)\s*分钟',
            ]
            for pattern in time_patterns:
                match = re.search(pattern, instruction, re.IGNORECASE)
                if match:
                    if len(match.groups()) == 2:
                        time_constraint = f"用户明确要求时长为 {match.group(1)}-{match.group(2)} 分钟"
                    else:
                        time_constraint = f"用户明确要求时长不超过 {match.group(1)} 分钟"
                    break
        
        # 构建提示词
        constraint_note = ""
        if time_constraint:
            constraint_note = f"""
【重要】{time_constraint}，请务必遵循用户的时长要求！
"""
        
        prompt = f"""
作为播客内容规划专家，请分析以下输入材料，并给出适合的播客长度和结构建议。

{analysis_material}
{constraint_note}
请根据以下标准进行评估：
1. 短播客（1-3分钟，约300-900字）：适合用户要求短时长、简单主题、概述性内容
2. 中短播客（3-5分钟，约900-1500字）：适合简单主题、资料较少
3. 中等播客（8-15分钟，约2400-4500字）：适合一般主题、资料适中、有一定深度
4. 长播客（20-30分钟，约6000-9000字）：适合复杂主题、资料丰富、需要深入探讨

【关键规则】如果用户在特殊指令中明确指定了时长要求（如"一分钟以内"、"3分钟"等），必须优先遵循用户要求，而不是根据资料量自动判断！

请以JSON格式回答，包含以下字段：
- podcast_length: "short", "medium", 或 "long"
- word_count_range: 建议的字数范围，如 "900-1500"
- time_range: 建议的时长范围（分钟），如 "3-5"
- structure_points: 建议的内容要点数量，如 4
- depth_suggestion: 深度建议，如 "概述性介绍" 或 "深入分析"
- reasoning: 简短的推理过程

只返回JSON格式的结果，不要有其他文字。
        """
        
        messages = [
            {"Role": "system", "Content": "你是一个专业的播客内容规划专家，擅长根据输入材料确定合适的播客长度和结构"},
            {"Role": "user", "Content": prompt},
        ]
        
        try:
            resp = self.api.chat(messages, stream=False)
            content = ""
            choices = resp.get("Choices") or resp.get("choices") or []
            if choices:
                msg = choices[0].get("Message") or choices[0].get("message") or {}
                content = msg.get("Content") or msg.get("content") or ""
            
            # 尝试解析JSON
            import json
            import re
            
            # 提取JSON部分（可能被包裹在代码块中）
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content
            
            # 清理可能的非JSON字符
            json_str = re.sub(r'^[^{]*', '', json_str)
            json_str = re.sub(r'[^}]*$', '', json_str)
            
            result = json.loads(json_str)
            return result
        except Exception as e:
            print(f"分析内容失败: {e}")
            # 返回默认值（中等长度）
            return {
                "podcast_length": "medium",
                "word_count_range": "2400-4500",
                "time_range": "8-15",
                "structure_points": 6,
                "depth_suggestion": "一般深度分析",
                "reasoning": "使用默认值，因为分析过程出错"
            }
    
    def adjust_prompt(self, base_prompt: str, analysis_result: Dict[str, Any]) -> str:
        """
        根据分析结果调整提示词
        
        参数:
            base_prompt: 基础提示词
            analysis_result: 分析结果
            
        返回:
            调整后的提示词
        """
        podcast_length = analysis_result.get("podcast_length", "medium")
        word_count_range = analysis_result.get("word_count_range", "2400-4500")
        time_range = analysis_result.get("time_range", "8-15")
        structure_points = analysis_result.get("structure_points", 6)
        depth_suggestion = analysis_result.get("depth_suggestion", "一般深度分析")
        
        # 替换时长要求
        adjusted_prompt = base_prompt.replace(
            "- 目标时长：8-15分钟（约2400-4500字）",
            f"- 目标时长：{time_range}分钟（约{word_count_range}字）"
        )
        
        # 替换总字数控制
        adjusted_prompt = adjusted_prompt.replace(
            "- 总字数控制在2400-4500字",
            f"- 总字数控制在{word_count_range}字"
        )
        
        # 调整内容结构
        if podcast_length == "short":
            # 短播客：简化结构
            structure_section = f"""【内容结构】（严格遵循）
1. 开场白（1-2轮）：简短欢迎+话题引入
2. 核心内容（{structure_points-2}轮）：
   - 简明扼要地介绍主要观点
   - 每个要点配合简短案例/数据[引用编号]
3. 结尾（1轮）：简短总结+行动建议"""
            
            # 替换内容结构部分
            import re
            adjusted_prompt = re.sub(
                r"【内容结构】（严格遵循）\n.*?(?=\n\n【对话风格】)",
                structure_section,
                adjusted_prompt,
                flags=re.DOTALL
            )
            
        elif podcast_length == "long":
            # 长播客：扩展结构
            structure_section = f"""【内容结构】（严格遵循）
1. 开场白（3-4轮）：热情欢迎+话题背景介绍+为什么重要+本期内容概述
2. 核心内容（{structure_points-6}轮）：
   - 按逻辑层层递进（背景→历史→现状→分析→影响→未来趋势）
   - 每个要点配合详细案例/数据/研究[引用编号]
   - B适时提问、总结、过渡，深入探讨关键点
3. 深度讨论（4-6轮）：多角度思考/争议点分析/专家观点对比/案例深度剖析
4. 结尾（3-4轮）：核心观点回顾+实用建议+延伸阅读+互动号召"""
            
            # 替换内容结构部分
            import re
            adjusted_prompt = re.sub(
                r"【内容结构】（严格遵循）\n.*?(?=\n\n【对话风格】)",
                structure_section,
                adjusted_prompt,
                flags=re.DOTALL
            )
        
        # 添加深度建议
        depth_note = f"\n- {depth_suggestion}"
        adjusted_prompt = adjusted_prompt.replace(
            "【对话风格】",
            f"【深度建议】{depth_note}\n\n【对话风格】"
        )
        
        return adjusted_prompt

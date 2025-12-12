# -*- coding: utf-8 -*-
"""
片头文案配置
每个风格对应一套片头文案，双人模式时 A/B 交替朗读
"""

# 风格分类映射
INTRO_STYLE_MAP = {
    "科技": "tech",
    "商业": "business",
    "财经": "business",
    "生活": "life",
    "日常": "life",
    "文化": "culture",
    "历史": "culture",
    "娱乐": "entertainment",
    "轻松": "entertainment",
    "教育": "education",
    "学习": "education",
    "健康": "health",
    "养生": "health",
    "情感": "emotion",
    "心理": "emotion",
    "成长": "growth",
    "个人成长": "growth",
    "通用": "general",
}

# 风格代码列表（用于UI下拉框）
INTRO_STYLES = [
    ("科技", "tech"),
    ("商业/财经", "business"),
    ("生活/日常", "life"),
    ("文化/历史", "culture"),
    ("娱乐/轻松", "entertainment"),
    ("教育/学习", "education"),
    ("健康/养生", "health"),
    ("情感/心理", "emotion"),
    ("个人成长", "growth"),
    ("通用", "general"),
    ("自定义", "custom"),
]

# 自定义片头文案最大字数限制
CUSTOM_INTRO_MAX_CHARS = 200

# 背景音乐文件映射
INTRO_BGM_FILES = {
    "tech": "bgm_tech.mp3",
    "business": "bgm_business.mp3",
    "life": "bgm_life.mp3",
    "culture": "bgm_culture.mp3",
    "entertainment": "bgm_entertainment.mp3",
    "education": "bgm_education.mp3",
    "health": "bgm_health.mp3",
    "emotion": "bgm_emotion.mp3",
    "growth": "bgm_growth.mp3",
    "general": "bgm_general.mp3",
}

# BGM长度调整策略配置
# "stretch": 变速放缩（通过改变播放速度来匹配语音长度）
# "loop": 循环/裁剪（通过循环或裁剪来匹配语音长度）
BGM_LENGTH_STRATEGY = {
    "tech": "stretch",       # 科技风格：变速
    "culture": "stretch",    # 文化风格：变速
    "business": "loop",      # 商业风格：循环
    "life": "loop",          # 生活风格：循环
    "entertainment": "loop", # 娱乐风格：循环
    "education": "loop",     # 教育风格：循环
    "health": "loop",        # 健康风格：循环
    "emotion": "loop",       # 情感风格：循环
    "growth": "loop",        # 成长风格：循环
    "general": "loop",       # 通用风格：循环
}

# 循环策略参数
LOOP_CROSSFADE_MS = 150  # 循环接缝处交叉淡化时长（毫秒）

# 片头文案配置（双人模式：A/B交替）
# 每个风格的文案是一个列表，奇数索引(0,2,4...)为A，偶数索引(1,3,5...)为B
INTRO_SCRIPTS = {
    "tech": [
        "这里解码未来",
        "探索创新",
        "从 AI 到元宇宙",
        "从芯片到量子计算",
        "我们关注科技如何重塑世界",
        "深入浅出，专业解析",
        "带你了解技术背后的故事",
        "欢迎来到科技前沿",
    ],
    "business": [
        "这里洞察商业",
        "解读趋势",
        "从创业到上市",
        "从资本到战略",
        "我们关注商业如何改变格局",
        "数据说话，逻辑为王",
        "带你看懂市场背后的博弈",
        "欢迎来到财经视角",
    ],
    "life": [
        "这里记录生活",
        "发现美好",
        "从清晨到日落",
        "从厨房到远方",
        "我们关注平凡中的不平凡",
        "轻松有趣，贴近真实",
        "带你感受生活的温度",
        "欢迎来到日常频道",
    ],
    "culture": [
        "这里穿越时光",
        "品读经典",
        "从古老文明到近代风云",
        "从东方到西方",
        "我们关注历史如何照进现实",
        "厚重而不沉闷",
        "带你重温那些被遗忘的故事",
        "欢迎来到文化长廊",
    ],
    "entertainment": [
        "这里释放快乐",
        "拒绝无聊",
        "从热梗到冷知识",
        "从吐槽到段子",
        "我们关注一切有趣的事",
        "不正经但有营养",
        "带你笑着度过每一天",
        "欢迎来到快乐星球",
    ],
    "education": [
        "这里点亮思维",
        "激发潜能",
        "从方法到实践",
        "从入门到精通",
        "我们关注知识如何改变命运",
        "干货满满，拒绝空谈",
        "带你高效学习每一天",
        "欢迎来到成长课堂",
    ],
    "health": [
        "这里守护健康",
        "关爱身心",
        "从饮食到运动",
        "从睡眠到心态",
        "我们关注如何活得更好",
        "科学靠谱，拒绝焦虑",
        "带你找到适合自己的节奏",
        "欢迎来到健康生活",
    ],
    "emotion": [
        "这里倾听内心",
        "理解情绪",
        "从亲密关系到自我成长",
        "从焦虑到释然",
        "我们关注心灵的每一次波动",
        "温暖而不说教",
        "带你与自己和解",
        "欢迎来到心灵角落",
    ],
    "growth": [
        "这里见证蜕变",
        "突破自我",
        "从迷茫到清晰",
        "从平凡到卓越",
        "我们关注每一次成长的瞬间",
        "脚踏实地，仰望星空",
        "带你成为更好的自己",
        "欢迎来到成长之路",
    ],
    # 通用风格：无语音，仅播放背景音乐
    "general": [],
}


def get_intro_script(style: str, host_mode: str = "dual", custom_script: str = None) -> list:
    """
    获取片头文案
    
    Args:
        style: 风格代码 (tech, business, life, etc.)
        host_mode: 主持人模式 (dual/single)
        custom_script: 自定义片头文案（多行文本，每行一句）
    
    Returns:
        文案列表，如果是通用风格则返回空列表
    """
    # 如果是自定义模式且有自定义文案
    if style == "custom" and custom_script:
        # 按行分割，过滤空行
        lines = [line.strip() for line in custom_script.strip().split('\n') if line.strip()]
        if not lines:
            return []
        
        if host_mode == "single":
            # 单人模式：合并所有行为一段
            return [" ".join(lines)]
        else:
            # 双人模式：返回行列表，奇数行给A，偶数行给B
            return lines
    
    scripts = INTRO_SCRIPTS.get(style, [])
    if not scripts:
        return []
    
    if host_mode == "single":
        # 单人模式：合并所有文案为一段
        return [" ".join(scripts)]
    else:
        # 双人模式：返回原始列表，A/B交替
        return scripts


def parse_custom_intro_script(custom_script: str, max_chars: int = None) -> tuple:
    """
    解析并验证自定义片头文案
    
    Args:
        custom_script: 用户输入的自定义文案
        max_chars: 最大字数限制，默认使用 CUSTOM_INTRO_MAX_CHARS
    
    Returns:
        (is_valid, lines_or_error): 如果有效返回 (True, 行列表)，否则返回 (False, 错误信息)
    """
    if max_chars is None:
        max_chars = CUSTOM_INTRO_MAX_CHARS
    
    if not custom_script or not custom_script.strip():
        return False, "请输入片头文案"
    
    # 检查字数
    total_chars = len(custom_script.replace('\n', '').replace(' ', ''))
    if total_chars > max_chars:
        return False, f"片头文案超过{max_chars}字限制（当前{total_chars}字）"
    
    # 按行分割
    lines = [line.strip() for line in custom_script.strip().split('\n') if line.strip()]
    if not lines:
        return False, "请输入至少一行片头文案"
    
    return True, lines


def get_intro_bgm_filename(style: str) -> str:
    """
    获取片头背景音乐文件名
    
    Args:
        style: 风格代码
    
    Returns:
        背景音乐文件名
    """
    return INTRO_BGM_FILES.get(style, "bgm_general.mp3")


def get_bgm_length_strategy(style: str) -> str:
    """
    获取BGM长度调整策略
    
    Args:
        style: 风格代码
    
    Returns:
        策略代码: "stretch" 或 "loop"
    """
    return BGM_LENGTH_STRATEGY.get(style, "loop")


def get_loop_crossfade_ms() -> int:
    """
    获取循环策略的交叉淡化时长
    
    Returns:
        交叉淡化时长（毫秒）
    """
    return LOOP_CROSSFADE_MS


def style_name_to_code(name: str) -> str:
    """
    将中文风格名称转换为代码
    
    Args:
        name: 中文风格名称
    
    Returns:
        风格代码
    """
    return INTRO_STYLE_MAP.get(name, "general")

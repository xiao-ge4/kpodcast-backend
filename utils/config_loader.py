import os
import json
import configparser
from typing import Dict, Any, List


def load_ini() -> Dict[str, Any]:
    """
    加载配置，优先级：环境变量 > config.ini 文件
    
    环境变量命名规则：PODCAST_<SECTION>_<KEY>（全大写，下划线分隔）
    例如：
      - PODCAST_TENCENT_SECRET_ID
      - PODCAST_COS_BUCKET
      - PODCAST_HUNYUAN_API_MODEL
    """
    # 关闭插值功能，避免 ini 值中含有 %（如 URL 编码）时触发格式化错误
    cfg = configparser.ConfigParser(interpolation=None)
    # 以当前文件所在目录为基准，定位到项目根目录
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    
    # 尝试加载 config.ini（可选，环境变量可完全替代）
    ini = os.getenv("ini", "")
    ini_loaded = False
    if not ini or not os.path.exists(ini):
        possible_configs = [
            os.path.join(base_dir, "config.ini"),
            os.path.join(base_dir, "Source_config_podcast.ini"),
            os.path.join(base_dir, "config_podcast.ini")
        ]
        for config_path in possible_configs:
            if os.path.exists(config_path):
                ini = config_path
                break
    
    if ini and os.path.exists(ini):
        cfg.read(ini, encoding="utf-8")
        ini_loaded = True
        print(f"✅ 已加载配置文件: {ini}")
    else:
        print("ℹ️ 未找到配置文件，将使用环境变量")
    
    def get_config(sec: str, key: str, default: str = "") -> str:
        """获取配置值，优先级：环境变量 > ini 文件 > 默认值"""
        # 环境变量名：PODCAST_<SECTION>_<KEY>
        env_key = f"PODCAST_{sec.upper()}_{key.upper()}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            return env_val
        # 从 ini 文件读取
        if ini_loaded and cfg.has_section(sec):
            return cfg.get(sec, key, fallback=default)
        return default
    
    # 简写
    g = get_config
    
    # 解析音色列表
    def parse_list(s: str) -> List:
        try:
            return json.loads(s) if s else []
        except Exception:
            return []
    
    voice_numbers = parse_list(g("tencent", "voice_number", "[]"))
    voice_labels = parse_list(g("tencent", "voice_role", "[]"))
    
    return {
        # Bocha 搜索
        "bocha_base_url": g("bocha", "base_url", ""),
        "bocha_search_path": g("bocha", "search_path", "/wiki/api/search"),
        "bocha_api_id": g("bocha", "api_id", ""),
        "bocha_api_key": g("bocha", "api_key", ""),
        
        # Hunyuan 模型
        "hunyuan_model_id": g("hunyuan", "model_id", "tencent/Hunyuan-MT-7B"),
        
        # TTS
        "tts_provider": g("tts", "provider", "tencent"),
        
        # 腾讯云（TTS + Hunyuan API 共用）
        "tencent_secret_id": g("tencent", "secret_id", ""),
        "tencent_secret_key": g("tencent", "secret_key", ""),
        "tencent_region": g("tencent", "region", "ap-beijing"),
        "voice_role_a": g("tencent", "voice_role_a", "101001"),
        "voice_role_b": g("tencent", "voice_role_b", "101015"),
        
        # 存储路径
        "output_dir": g("storage", "output_dir", os.path.join(os.getcwd(), "outputs")),
        "assets_bgm_dir": g("storage", "assets_bgm_dir", os.path.join(os.getcwd(), "assets", "bgm")),
        
        # BGM
        "bgm_history": g("ui", "intro_bgm_history", "history.mp3"),
        "bgm_entertainment": g("ui", "intro_bgm_entertainment", "entertainment.mp3"),
        "bgm_serious": g("ui", "intro_bgm_serious", "serious.mp3"),
        
        # Hunyuan API（云端调用）
        "hunyuan_api_secret_id": g("tencent", "secret_id", ""),
        "hunyuan_api_secret_key": g("tencent", "secret_key", ""),
        "hunyuan_api_region": g("tencent", "region", "ap-beijing"),
        "hunyuan_api_model": g("hunyuan_api", "model", "hunyuan-turbos-latest"),
        "hunyuan_api_temperature": float(g("hunyuan_api", "temperature", "1")),
        "hunyuan_api_top_p": float(g("hunyuan_api", "top_p", "0.5")),
        "hunyuan_api_max_tokens": int(g("hunyuan_api", "max_tokens", "256")),
        
        # TTS 备选音色
        "voice_numbers": voice_numbers,
        "voice_labels": voice_labels,
        
        # 搜索相关
        "supplementary_search_count": int(g("search", "supplementary_search_count", "4")),
        
        # Web 抽取
        "url_extract_cookie": g("web_extract", "cookie", ""),
        "url_extract_headers_json": g("web_extract", "headers_json", ""),
        "url_extract_headers": g("web_extract", "headers", ""),
        "web_extract_render_mode": g("web_extract", "render_mode", "off"),
        "web_extract_render_wait_ms": int(g("web_extract", "render_wait_ms", "1200")),
        "web_extract_render_timeout_ms": int(g("web_extract", "render_timeout_ms", "15000")),
        
        # COS 云存储
        "cos_enabled": g("cos", "enabled", "false").lower() == "true",
        "cos_secret_id": g("cos", "secret_id", ""),
        "cos_secret_key": g("cos", "secret_key", ""),
        "cos_region": g("cos", "region", "ap-guangzhou"),
        "cos_bucket": g("cos", "bucket", ""),
    }



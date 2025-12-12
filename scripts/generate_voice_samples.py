"""
一次性脚本：为所有音色生成试听样本音频
运行方式：python scripts/generate_voice_samples.py
"""
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients.tencent_tts import synthesize_tencent_tts
from utils.config_loader import load_ini

# 音色配置
VOICES = [
    {"id": "101001", "name": "智瑜", "lang": "zh"},
    {"id": "101015", "name": "智萌", "lang": "zh"},
    {"id": "501001", "name": "智兰", "lang": "zh"},
    {"id": "501006", "name": "千嶂", "lang": "zh"},
    {"id": "501005", "name": "飞镜", "lang": "zh"},
    {"id": "502005", "name": "智小解", "lang": "zh"},
    {"id": "601009", "name": "爱小芊", "lang": "zh"},
    {"id": "601007", "name": "爱小叶", "lang": "zh"},
    {"id": "501002", "name": "智菊", "lang": "zh"},
    {"id": "501003", "name": "智宇", "lang": "zh"},
    {"id": "501008", "name": "WeJames", "lang": "en"},
    {"id": "501009", "name": "WeWinny", "lang": "en"},
]

# 示例文本
SAMPLE_TEXT_ZH = "你好，我是播客主持人"
SAMPLE_TEXT_EN = "Hello, I'm your podcast host"


def main():
    # 加载配置
    cfg = load_ini()
    
    secret_id = cfg.get("tencent_secret_id", "")
    secret_key = cfg.get("tencent_secret_key", "")
    region = cfg.get("tencent_region", "ap-beijing")
    
    if not secret_id or not secret_key:
        print("❌ 错误：请在 config.ini 中配置腾讯云密钥")
        return
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "voice_samples")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 输出目录: {output_dir}")
    print(f"🎤 开始生成 {len(VOICES)} 个音色样本...\n")
    
    success_count = 0
    fail_count = 0
    
    for voice in VOICES:
        voice_id = voice["id"]
        voice_name = voice["name"]
        lang = voice["lang"]
        
        # 选择示例文本
        text = SAMPLE_TEXT_EN if lang == "en" else SAMPLE_TEXT_ZH
        
        print(f"🔊 生成 {voice_id}:{voice_name} ({lang})...", end=" ")
        
        # 调用 TTS
        result = synthesize_tencent_tts(
            text=text,
            secret_id=secret_id,
            secret_key=secret_key,
            region=region,
            voice=voice_id,
            speed=0,
            codec="mp3"
        )
        
        if result.get("success"):
            # 保存音频文件
            output_path = os.path.join(output_dir, f"voice_{voice_id}.mp3")
            with open(output_path, "wb") as f:
                f.write(result["bytes"])
            print(f"✅ 成功 ({len(result['bytes'])} bytes)")
            success_count += 1
        else:
            print(f"❌ 失败: {result.get('error', '未知错误')}")
            fail_count += 1
    
    print(f"\n📊 完成！成功: {success_count}, 失败: {fail_count}")


if __name__ == "__main__":
    main()

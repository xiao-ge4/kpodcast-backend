"""
测试腾讯云混元 API 连接
使用与 podcast 后端相同的配置
"""
import sys
from utils.config_loader import load_ini
from clients.hunyuan_api_client import HunyuanAPIClient

def test_hunyuan_api():
    """测试混元 API 连接"""
    print("=" * 50)
    print("测试腾讯云混元 API 连接")
    print("=" * 50)
    
    # 加载配置
    try:
        cfg = load_ini()
        print("✅ 配置文件加载成功")
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False
    
    # 检查必要配置
    required_keys = [
        "hunyuan_api_secret_id",
        "hunyuan_api_secret_key", 
        "hunyuan_api_region",
        "hunyuan_api_model"
    ]
    
    print("\n配置检查:")
    for key in required_keys:
        value = cfg.get(key)
        if value:
            # 隐藏敏感信息
            if "secret" in key.lower() or "key" in key.lower():
                display = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            else:
                display = value
            print(f"  {key}: {display}")
        else:
            print(f"  ❌ {key}: 未设置")
            return False
    
    # 创建 API 客户端
    print("\n创建 API 客户端...")
    try:
        api = HunyuanAPIClient(
            secret_id=cfg["hunyuan_api_secret_id"],
            secret_key=cfg["hunyuan_api_secret_key"],
            region=cfg["hunyuan_api_region"],
            model=cfg["hunyuan_api_model"],
            temperature=0.8,
            top_p=0.8,
            max_tokens=100,  # 测试用小一点
        )
        print("✅ API 客户端创建成功")
    except Exception as e:
        print(f"❌ API 客户端创建失败: {e}")
        return False
    
    # 发送测试请求
    print("\n发送测试请求...")
    print("请求内容: '你好，请用一句话介绍自己'")
    print("等待响应中（可能需要 10-30 秒）...")
    
    try:
        # 腾讯云混元 API 需要大写的 Role 和 Content
        messages = [{"Role": "user", "Content": "你好，请用一句话介绍自己"}]
        response = api.chat(messages, stream=False)
        print(f"\n✅ API 响应成功!")
        print(f"响应内容: {response[:200]}..." if len(response) > 200 else f"响应内容: {response}")
        return True
    except Exception as e:
        print(f"\n❌ API 请求失败: {e}")
        print("\n可能原因:")
        print("  1. 网络问题 - 检查是否能访问腾讯云服务")
        print("  2. API 密钥错误 - 检查 secret_id 和 secret_key")
        print("  3. 服务繁忙 - 稍后重试")
        print("  4. 超时 - 网络延迟过高")
        return False


def test_tts_api():
    """测试腾讯云 TTS API"""
    print("\n" + "=" * 50)
    print("测试腾讯云 TTS API 连接")
    print("=" * 50)
    
    cfg = load_ini()
    
    print("\nTTS 配置检查:")
    print(f"  secret_id: {cfg.get('tencent_secret_id', '未设置')[:8]}...")
    print(f"  secret_key: {cfg.get('tencent_secret_key', '未设置')[:8]}...")
    print(f"  region: {cfg.get('tencent_region', '未设置')}")
    
    try:
        from clients.tencent_tts import synthesize_tencent_tts
        
        print("✅ TTS 模块导入成功")
        print("发送 TTS 测试请求...")
        
        result = synthesize_tencent_tts(
            text="你好，这是一个测试",
            secret_id=cfg["tencent_secret_id"],
            secret_key=cfg["tencent_secret_key"],
            region=cfg.get("tencent_region", "ap-beijing"),
            voice="501006",
            speed=0
        )
        
        if result.get("success"):
            audio_size = len(result.get("bytes", b""))
            print(f"✅ TTS 合成成功! 音频大小: {audio_size} bytes")
            return True
        else:
            print(f"❌ TTS 合成失败: {result.get('error')}")
            return False
    except Exception as e:
        print(f"❌ TTS 测试失败: {e}")
        return False


if __name__ == "__main__":
    print("\n🎙️ Podcast Generator API 测试\n")
    
    # 测试混元 API
    hunyuan_ok = test_hunyuan_api()
    
    # 测试 TTS API
    tts_ok = test_tts_api()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    print(f"  混元 LLM API: {'✅ 正常' if hunyuan_ok else '❌ 异常'}")
    print(f"  腾讯云 TTS:   {'✅ 正常' if tts_ok else '❌ 异常'}")
    
    if hunyuan_ok and tts_ok:
        print("\n🎉 所有测试通过！可以正常使用 Podcast Generator")
    else:
        print("\n⚠️ 部分测试失败，请检查配置和网络")
    
    sys.exit(0 if (hunyuan_ok and tts_ok) else 1)

from typing import Dict, Any
import base64
import json
import uuid


def synthesize_tencent_tts(text: str, secret_id: str, secret_key: str, region: str,
                            voice: str = "501006", speed: int = 0, codec: str = "mp3") -> Dict[str, Any]:
    """使用腾讯云 TTS 合成语音。

    参数说明：
    - text: 待合成文本（建议清洗后分段调用）
    - secret_id/secret_key/region: 腾讯云凭据与地域
    - voice: 音色编号（示例：男 501006，女 601007）
    - speed: 语速（-2..2，0为正常）
    - codec: 输出格式 'mp3' 或 'wav'
    返回：{"success": True, "bytes": b"..."} 或 {"success": False, "error": "..."}
    """
    try:
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.tts.v20190823 import tts_client, models

        cred = credential.Credential(secret_id, secret_key)
        http_profile = HttpProfile()
        http_profile.endpoint = "tts.tencentcloudapi.com"
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        client = tts_client.TtsClient(cred, region or "", client_profile)

        req = models.TextToVoiceRequest()
        try:
            voice_type = int(voice)
        except Exception:
            voice_type = 501006
        try:
            spd = int(speed)
        except Exception:
            spd = 0
        spd = max(-2, min(2, spd))
        params = {
            "Text": text,
            "SessionId": str(uuid.uuid4()),
            "Volume": 1,
            "Speed": spd,
            "ProjectId": 0,
            "ModelType": 1,
            "VoiceType": voice_type,
            "PrimaryLanguage": 1,
            "SampleRate": 16000,
            "Codec": codec or "mp3",
            "EnableSubtitle": False
        }
        req.from_json_string(json.dumps(params, ensure_ascii=False))
        resp = client.TextToVoice(req)
        audio_b64 = resp.Audio
        if not audio_b64:
            return {"success": False, "error": "empty audio"}
        return {"success": True, "bytes": base64.b64decode(audio_b64)}
    except Exception as e:
        return {"success": False, "error": str(e)}



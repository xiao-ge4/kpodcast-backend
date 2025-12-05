import json
from typing import List, Dict, Any
from tencentcloud.common import credential
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models


class HunyuanAPIClient:
    def __init__(self, secret_id: str, secret_key: str, region: str, model: str = "hunyuan-turbos-latest",
                 temperature: float = 1.0, top_p: float = 0.5, max_tokens: int = 256):
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.region = region
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        cred = credential.Credential(self.secret_id, self.secret_key)
        self.client = hunyuan_client.HunyuanClient(cred, self.region)

    def chat(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        req = models.ChatCompletionsRequest()
        payload = {
            "Model": self.model,
            "Messages": messages,
            "Temperature": self.temperature,
            "TopP": self.top_p,
            "MaxTokens": self.max_tokens,
            "Stream": bool(stream),
        }
        req.from_json_string(json.dumps(payload, ensure_ascii=False))
        resp = self.client.ChatCompletions(req)
        return json.loads(resp.to_json_string())











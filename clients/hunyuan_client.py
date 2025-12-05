from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class HunyuanClient:
    def __init__(self, model_id: str = "tencent/Hunyuan-MT-7B"):
        self.model_id = model_id
        self.pipe = None
        # 优先使用 translation 管线（官方示例）
        try:
            self.pipe = pipeline(
                task="translation",
                model=model_id,
                device_map="auto",
                trust_remote_code=True,
            )
            self.tokenizer = self.pipe.tokenizer
            self.model = getattr(self.pipe, "model", None)
            return
        except Exception:
            self.pipe = None
        # 回退到 AutoModelForCausalLM（与模型卡示例一致）
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map="auto",
            trust_remote_code=True,
        )

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        # 若是翻译管线，则直接返回生成文本（用于本项目：把提示当输入，得到可用中文文本）
        if self.pipe is not None:
            try:
                out = self.pipe(prompt, max_new_tokens=max_new_tokens)
                if isinstance(out, list) and out:
                    return out[0].get("translation_text") or out[0].get("generated_text") or str(out[0])
            except Exception:
                pass
        # 回退到 CausalLM 生成
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)



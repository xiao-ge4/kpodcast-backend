import os
import requests
from typing import List, Dict, Any


class BochaClient:
    def __init__(self, base_url: str, api_id: str, api_key: str, search_path: str = "/wiki/api/search"):
        self.base_url = (base_url or "").rstrip("/")
        self.api_id = api_id
        self.api_key = api_key
        self.search_path = search_path

    def _parse_items(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(data, dict):
            return []
        # 常见返回形态：
        # 1) { data: { webPages: { value: [...] } } }
        # 2) { data: [...] } / { results: [...] } / { items: [...] }
        cand_lists = []
        try:
            cand_lists.append((((data.get("data") or {}).get("webPages") or {}).get("value")))
        except Exception:
            pass
        for key in ("data", "results", "items"):
            v = data.get(key)
            if isinstance(v, list):
                cand_lists.append(v)
        # 取第一个是 list 的
        for arr in cand_lists:
            if isinstance(arr, list):
                out = []
                for it in arr:
                    if not isinstance(it, dict):
                        continue
                    out.append({
                        "title": it.get("title") or it.get("name") or "",
                        "url": it.get("url") or it.get("link") or "",
                        "snippet": it.get("snippet") or it.get("summary") or it.get("content") or ""
                    })
                if out:
                    return out
        return []

    def search(self, query: str, count: int = 8) -> List[Dict[str, Any]]:
        # 优先尝试文档中的飞书域名（X-API-ID / X-API-KEY）
        errors = []
        try:
            url = f"{self.base_url}{self.search_path}"
            headers = {
                "X-API-ID": self.api_id or "",
                "X-API-KEY": self.api_key or "",
                "Content-Type": "application/json"
            }
            payload = {"q": query, "count": count}
            r = requests.post(url, json=payload, headers=headers, timeout=20)
            if r.status_code == 200:
                return self._parse_items(r.json())
            else:
                errors.append(f"{url} -> {r.status_code} {r.text[:200]}")
        except Exception as e:
            errors.append(f"pair-auth error: {e}")

        # 回退：官方域名 Bearer Token 形态
        try:
            url2 = "https://api.bochaai.com/v1/web-search"
            headers2 = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload2 = {"query": query, "count": count, "summary": False}
            r2 = requests.post(url2, json=payload2, headers=headers2, timeout=20)
            if r2.status_code == 200:
                return self._parse_items(r2.json())
            else:
                errors.append(f"{url2} -> {r2.status_code} {r2.text[:200]}")
        except Exception as e:
            errors.append(f"bearer-auth error: {e}")

        raise RuntimeError("Bocha search failed: " + " | ".join(errors))



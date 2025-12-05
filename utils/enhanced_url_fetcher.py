HAVE_PLAYWRIGHT = False
try:
    from playwright.sync_api import sync_playwright
    HAVE_PLAYWRIGHT = True
except Exception:
    # 未安装时按关闭处理
    HAVE_PLAYWRIGHT = False
"""
增强的网页内容提取模块
使用多种工具级联提取策略，提高网页内容提取的成功率和质量
支持现代网页和移动端网页
"""
from typing import Dict, Any, List, Optional, Tuple
import requests
import trafilatura
import logging
import re
from urllib.parse import urlparse
from html import unescape
import os
import json
from utils.config_loader import load_ini

# 备用编码探测
try:
    from charset_normalizer import from_bytes as cn_from_bytes
except Exception:
    cn_from_bytes = None

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("enhanced_url_fetcher")

# 尝试导入可选的依赖库
HAVE_NEWSPAPER = False
HAVE_READABILITY = False
# 移除 requests-html 依赖，因为它需要下载 Chromium 浏览器

try:
    import newspaper
    from newspaper import Article
    HAVE_NEWSPAPER = True
except ImportError:
    logger.warning("newspaper3k 库未安装，将不使用该提取方法")

try:
    from readability import Document
    HAVE_READABILITY = True
except ImportError:
    logger.warning("readability-lxml 库未安装，将不使用该提取方法")


def validate_content(text: str, url: str = "") -> Tuple[bool, float]:
    """
    验证提取的内容质量
    
    参数:
        text: 提取的文本内容
        url: 原始URL，用于上下文判断
        
    返回:
        (是否有效, 质量分数)
    """
    if not text or len(text.strip()) < 100:
        return False, 0.0
    
    # 计算文本质量分数 (0.0-1.0)
    score = 0.0
    
    # 1. 长度评分 (最高0.4分)
    length = len(text.strip())
    if length > 10000:
        score += 0.4
    elif length > 5000:
        score += 0.3
    elif length > 2000:
        score += 0.2
    elif length > 500:
        score += 0.1
    
    # 2. 结构评分 (最高0.3分)
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 5:
        score += 0.3
    elif len(paragraphs) > 3:
        score += 0.2
    elif len(paragraphs) > 1:
        score += 0.1
    
    # 3. 内容多样性评分 (最高0.3分)
    # 检查是否包含数字
    if re.search(r'\d', text):
        score += 0.1
    # 检查是否包含标点符号
    if re.search(r'[,.;:!?，。；：！？]', text):
        score += 0.1
    # 检查段落平均长度
    avg_para_len = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)
    if avg_para_len > 100:
        score += 0.1
    
    return score > 0.3, score  # 分数大于0.3认为是有效内容


def extract_with_trafilatura(url: str, html_content: Optional[str] = None) -> str:
    """使用trafilatura提取网页内容"""
    try:
        if html_content:
            text = trafilatura.extract(html_content, include_comments=False, include_tables=True) or ""
        else:
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=True) or ""
        return text
    except Exception as e:
        logger.warning(f"trafilatura提取失败: {e}")
        return ""


def extract_with_newspaper(url: str, html_content: Optional[str] = None) -> str:
    """使用newspaper3k提取网页内容"""
    if not HAVE_NEWSPAPER:
        return ""
    
    try:
        article = Article(url)
        if html_content:
            article.set_html(html_content)
        else:
            article.download()
        article.parse()
        return article.text
    except Exception as e:
        logger.warning(f"newspaper提取失败: {e}")
        return ""


def extract_with_readability(url: str, html_content: Optional[str] = None) -> str:
    """使用readability-lxml提取网页内容"""
    if not HAVE_READABILITY:
        return ""
    
    try:
        if not html_content:
            response = requests.get(
                url,
                timeout=20,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                },
            )
            html_content = _smart_decode(response.content, response.headers)
        
        doc = Document(html_content)
        content = doc.summary()
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', ' ', content)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.warning(f"readability提取失败: {e}")
        return ""


# 移除 requests-html 相关函数


def _smart_decode(content: bytes, headers: Optional[Dict[str, Any]] = None) -> str:
    """尽可能正确地将网页字节解码为字符串，避免乱码。"""
    # 1) Header charset
    charset = None
    if headers:
        ctype = headers.get('Content-Type') or headers.get('content-type') or ''
        m = re.search(r'charset=([A-Za-z0-9_-]+)', ctype)
        if m:
            charset = m.group(1).strip().lower()
    # 2) Try meta charset inside first 4KB to reduce cost
    if not charset:
        head = content[:4096].decode('latin-1', errors='ignore')
        m = re.search(r'<meta[^>]*charset=["\']?([A-Za-z0-9_-]+)', head, re.IGNORECASE)
        if m:
            charset = m.group(1).strip().lower()
        else:
            m = re.search(r'<meta[^>]*content=["\'][^"\']*charset=([A-Za-z0-9_-]+)["\']', head, re.IGNORECASE)
            if m:
                charset = m.group(1).strip().lower()
    # 3) Charset-normalizer guess
    candidates = []
    if charset:
        candidates.append(charset)
    if cn_from_bytes is not None:
        try:
            best = cn_from_bytes(content).best()
            if best and best.encoding:
                candidates.append(best.encoding.lower())
        except Exception:
            pass
    # 4) Fallback list
    candidates.extend(["utf-8", "gbk", "gb2312"])  # 常见中文网页编码
    seen = set()
    for enc in candidates:
        enc = enc.strip().lower()
        if enc in seen:
            continue
        seen.add(enc)
        try:
            txt = content.decode(enc, errors='ignore')
            # 简单质量判断：避免出现大量替换字符或全是ASCII却看似中文被错解
            if txt:
                return unescape(txt)
        except Exception:
            continue
    # 最后一搏
    try:
        return content.decode('utf-8', errors='ignore')
    except Exception:
        return content.decode('latin-1', errors='ignore')


def _extract_canonical(html: str) -> Optional[str]:
    """从HTML中提取canonical或og:url。"""
    m = re.search(r'<link[^>]*rel=["\']canonical["\'][^>]*href=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'<meta[^>]*property=["\']og:url["\'][^>]*content=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _render_with_playwright(url: str, headers: Dict[str, str], cookie_str: str, wait_ms: int, timeout_ms: int) -> Optional[str]:
    """使用 Playwright 渲染页面，返回渲染后的 HTML。需要已安装 playwright 及浏览器。
    headers: 额外请求头
    cookie_str: 形如 "k1=v1; k2=v2"
    """
    if not HAVE_PLAYWRIGHT:
        logger.warning("未安装 Playwright，跳过渲染模式")
        return None
    try:
        with sync_playwright() as p:
            # 启动浏览器，添加反检测参数
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-features=site-per-process',
                ]
            )
            
            # 检测是否为移动端 User-Agent
            user_agent = headers.get('User-Agent') or headers.get('user-agent') or ""
            is_mobile = 'Mobile' in user_agent or 'Android' in user_agent
            
            # 创建上下文，如果是移动端则模拟移动设备
            # 清理 headers，移除可能导致问题的字段
            clean_headers = {}
            for k, v in headers.items():
                # 跳过某些可能导致问题的 header
                if k.lower() not in ['host', 'content-length', 'connection']:
                    clean_headers[k] = v
            
            context_params = {
                'user_agent': user_agent or None,
                'extra_http_headers': clean_headers,
                'accept_downloads': False,
                'java_script_enabled': True,
                'ignore_https_errors': True,
                'locale': "zh-CN",
            }
            
            # 如果是移动端 UA，添加移动端视口
            if is_mobile:
                context_params.update({
                    'viewport': {'width': 360, 'height': 640},
                    'device_scale_factor': 3,
                    'is_mobile': True,
                    'has_touch': True,
                })
            
            context = browser.new_context(**context_params)
            
            # 注入 cookie
            if cookie_str:
                try:
                    parsed = urlparse(url)
                    domain = parsed.hostname or ""
                    cookies = []
                    for part in cookie_str.split(';'):
                        if '=' in part:
                            k, v = part.split('=', 1)
                            cookies.append({
                                'name': k.strip(),
                                'value': v.strip(),
                                'domain': f'.{domain}' if not domain.startswith('.') else domain,
                                'path': '/',
                            })
                    if cookies:
                        context.add_cookies(cookies)
                        logger.info(f"注入了 {len(cookies)} 个 Cookie")
                except Exception as ce:
                    logger.warning(f"渲染模式设置 Cookie 失败: {ce}")
            
            # 创建页面
            page = context.new_page()
            page.set_default_timeout(timeout_ms)
            
            # 尝试访问页面
            try:
                # 直接访问 URL（Playwright 会自动处理 URL 编码）
                response = page.goto(url, wait_until='domcontentloaded', timeout=timeout_ms)
                logger.info(f"页面访问成功，状态: {response.status if response else 'unknown'}")
            except Exception as goto_error:
                logger.warning(f"页面访问失败: {goto_error}")
                # 不要尝试重新编码，直接抛出错误
                raise goto_error
            
            # 等待页面加载
            if wait_ms and wait_ms > 0:
                page.wait_for_timeout(wait_ms)
            
            # 额外等待网络静默
            try:
                page.wait_for_load_state('networkidle', timeout=5000)
            except Exception:
                pass
            
            # 获取最终 URL 和内容
            final_url = page.url
            html = page.content()
            title = page.title()
            
            logger.info(f"页面标题: {title}")
            logger.info(f"最终 URL: {final_url[:100]}...")
            logger.info(f"内容长度: {len(html)} 字符")
            
            # 检查是否被重定向到验证页面
            if "验证" in title or "验证" in html[:1000]:
                logger.warning("检测到验证页面，可能需要更新 Cookie")
            
            context.close()
            browser.close()
            return html
    except Exception as e:
        logger.warning(f"Playwright 渲染失败: {e}")
        return None


def fetch_url_enhanced(url: str) -> Dict[str, Any]:
    """
    增强版网页内容提取函数
    
    使用多种工具级联提取策略，提高网页内容提取的成功率和质量
    
    参数:
        url: 要提取内容的网页URL
        
    返回:
        包含提取结果的字典:
        {
            "success": 是否成功提取内容,
            "text": 提取的文本内容,
            "status": HTTP状态码,
            "quality_score": 内容质量分数,
            "extractor": 使用的提取器名称
        }
    """
    logger.info(f"开始提取URL内容: {url}")
    
    # 初始化结果
    result = {
        "success": False,
        "text": "",
        "status": 0,
        "quality_score": 0.0,
        "extractor": "none"
    }
    
    try:
        # 从环境变量构建会话：可注入 Cookie 与额外 Headers
        session = requests.Session()
        base_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        # 额外Headers：优先使用 ini 中直接提供的 headers 块（Python/JSON 皆可）
        cfg = load_ini()
        extra_headers = {}
        direct_headers = {}
        # 1) 直接 headers 块（优先级最高）
        try:
            raw_direct = cfg.get('url_extract_headers') if isinstance(cfg, dict) else ''
            if raw_direct:
                # 先尝试按 JSON 解析
                try:
                    direct_headers = json.loads(raw_direct)
                except Exception:
                    # 再尝试将 Python 风格字典替换为 JSON 兼容再 loads
                    tmp = raw_direct.strip()
                    # 简单规范化：单引号 -> 双引号
                    if tmp.startswith('{') and tmp.endswith('}'):
                        tmp2 = tmp.replace("'", '"')
                        direct_headers = json.loads(tmp2)
        
        except Exception:
            direct_headers = {}
        env_headers = os.getenv('URL_EXTRACT_HEADERS_JSON')
        cfg_headers = cfg.get('url_extract_headers_json') if isinstance(cfg, dict) else ""
        if env_headers or cfg_headers:
            try:
                extra_headers = json.loads((env_headers or cfg_headers) or '{}')
            except Exception:
                extra_headers = {}
        # 合并优先级：base < ini/env JSON < direct headers
        merged_headers = {**base_headers, **extra_headers, **direct_headers}
        session.headers.update(merged_headers)
        # Cookie（形如 "k1=v1; k2=v2" ）
        cookie_str = os.getenv('URL_EXTRACT_COOKIE') or os.getenv('URL_EXTRACT_COOKIES') or (cfg.get('url_extract_cookie') if isinstance(cfg, dict) else "")
        if cookie_str:
            for part in cookie_str.split(';'):
                if '=' in part:
                    k, v = part.split('=', 1)
                    session.cookies.set(k.strip(), v.strip())
        
        # 检查是否启用 Playwright 渲染
        render_mode = (cfg.get('web_extract_render_mode') or 'off').lower()
        logger.info(f"当前渲染配置: mode={render_mode}, HAVE_PLAYWRIGHT={HAVE_PLAYWRIGHT}")
        
        html_content = None
        canonical_url = None
        
        # 如果启用 Playwright，优先使用它（避免被反爬虫拦截）
        if render_mode == 'playwright' and HAVE_PLAYWRIGHT:
            wait_ms = int(cfg.get('web_extract_render_wait_ms') or 2000)
            timeout_ms = int(cfg.get('web_extract_render_timeout_ms') or 25000)
            logger.info("优先使用 Playwright 渲染模式")
            rendered_html = _render_with_playwright(url, dict(session.headers), cookie_str, wait_ms, timeout_ms)
            if rendered_html:
                html_content = rendered_html
                canonical_url = _extract_canonical(html_content)
                result["status"] = 200  # Playwright 成功视为 200
                logger.info("Playwright 渲染成功，获取到页面内容")
        
        # 如果 Playwright 失败或未启用，尝试普通 HTTP 请求
        if not html_content:
            try:
                response = session.get(url, timeout=20)
                response.raise_for_status()
                html_content = _smart_decode(response.content, response.headers)
                result["status"] = response.status_code
                canonical_url = _extract_canonical(html_content)
                logger.info(f"HTTP 请求成功，状态码: {response.status_code}")
            except Exception as http_error:
                logger.warning(f"HTTP 请求失败: {http_error}")
                # 如果 HTTP 也失败，且没有 html_content，则无法继续
                if not html_content:
                    raise
        
        # 1. 首先尝试trafilatura（轻量级，速度快）
        logger.info("尝试使用trafilatura提取")
        text = extract_with_trafilatura(url, html_content)
        valid, score = validate_content(text, url)
        
        if valid:
            logger.info(f"trafilatura提取成功，质量分数: {score:.2f}")
            result["success"] = True
            result["text"] = text
            result["quality_score"] = score
            result["extractor"] = "trafilatura"
            return result
        
        # 2. 尝试newspaper3k
        if HAVE_NEWSPAPER:
            logger.info("尝试使用newspaper3k提取")
            text = extract_with_newspaper(url, html_content)
            valid, score = validate_content(text, url)
            
            if valid:
                logger.info(f"newspaper3k提取成功，质量分数: {score:.2f}")
                result["success"] = True
                result["text"] = text
                result["quality_score"] = score
                result["extractor"] = "newspaper3k"
                return result
        
        # 3. 尝试readability
        if HAVE_READABILITY:
            logger.info("尝试使用readability提取")
            text = extract_with_readability(url, html_content)
            valid, score = validate_content(text, url)
            
            if valid:
                logger.info(f"readability提取成功，质量分数: {score:.2f}")
                result["success"] = True
                result["text"] = text
                result["quality_score"] = score
                result["extractor"] = "readability"
                return result
        
        # 若存在 canonical，尝试对 canonical 重新获取并再跑三器
        if canonical_url and canonical_url != url:
            try:
                logger.info(f"发现canonical: {canonical_url}，尝试跟随并重新提取")
                resp2 = session.get(
                    canonical_url,
                    timeout=20,
                    headers={ 'Referer': url, **session.headers },
                )
                html2 = _smart_decode(resp2.content, resp2.headers)
                # trafilatura
                text = extract_with_trafilatura(canonical_url, html2)
                valid, score = validate_content(text, canonical_url)
                if not valid and HAVE_NEWSPAPER:
                    text = extract_with_newspaper(canonical_url, html2)
                    valid, score = validate_content(text, canonical_url)
                if not valid and HAVE_READABILITY:
                    text = extract_with_readability(canonical_url, html2)
                    valid, score = validate_content(text, canonical_url)
                if valid:
                    result["success"] = True
                    result["text"] = text
                    result["quality_score"] = score
                    result["extractor"] = "canonical_follow"
                    return result
            except Exception as ce:
                logger.warning(f"canonical 提取失败: {ce}")
        
        # 5. 如果所有方法都失败，但我们有HTML内容，尝试直接提取可见文本
        logger.info("所有提取方法失败，尝试直接从HTML提取文本")
        # 移除script, style等标签
        cleaned_html = re.sub(r'<(script|style|meta|link).*?</\1>|<(script|style|meta|link).*?>', '', html_content, flags=re.DOTALL)
        # 移除HTML标签，保留文本
        raw_text = re.sub(r'<[^>]+>', ' ', cleaned_html)
        # 清理空白字符
        raw_text = re.sub(r'\s+', ' ', raw_text).strip()
        
        valid, score = validate_content(raw_text, url)
        if valid:
            logger.info(f"直接HTML提取成功，质量分数: {score:.2f}")
            result["success"] = True
            result["text"] = raw_text
            result["quality_score"] = score
            result["extractor"] = "raw_html"
            return result
        
        # 所有方法都失败
        logger.warning(f"所有提取方法均失败，URL: {url}")
        result["text"] = raw_text  # 返回最基本的提取结果
        return result
        
    except Exception as e:
        logger.error(f"URL内容提取过程中发生错误: {e}")
        result["error"] = str(e)
        return result


# 兼容原有fetch_url接口
def fetch_url(url: str) -> Dict[str, Any]:
    """
    兼容原有fetch_url接口的增强版网页内容提取函数
    
    参数:
        url: 要提取内容的网页URL
        
    返回:
        包含提取结果的字典:
        {
            "success": 是否成功提取内容,
            "text": 提取的文本内容,
            "status": HTTP状态码
        }
    """
    result = fetch_url_enhanced(url)
    # 简化返回结果，保持与原接口兼容
    return {
        "success": result["success"],
        "text": result["text"],
        "status": result["status"]
    }


if __name__ == "__main__":
    # 测试代码
    test_url = "https://www.example.com"
    result = fetch_url_enhanced(test_url)
    print(f"提取结果: 成功={result['success']}, 提取器={result['extractor']}, 质量分数={result['quality_score']}")
    print(f"内容长度: {len(result['text'])}")
    print(f"内容预览: {result['text'][:200]}...")

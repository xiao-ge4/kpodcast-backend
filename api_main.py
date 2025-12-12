"""
FastAPI 版本的播客生成器 API
完整移植自 Gradio 版本 app.py
"""
import logging
# 抑制 pdfminer 的颜色解析警告
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
import tempfile
import traceback
import asyncio
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# 创建线程池用于运行同步代码（如 Playwright）
executor = ThreadPoolExecutor(max_workers=4)

# 导入原有的功能模块
from pipeline.podcast_pipeline_new import run_end_to_end, generate_script_only, synthesize_audio_only
from utils.config_loader import load_ini
from clients.hunyuan_api_client import HunyuanAPIClient
from utils.pdf_loader import save_uploaded_files, process_pdf_files, extract_text_from_pdf, merge_pdf_contents

# COS 客户端（延迟初始化）
cos_client = None

app = FastAPI(title="Podcast Generator API")


def init_cos_client():
    """初始化 COS 客户端"""
    global cos_client
    if cfg.get("cos_enabled") and cfg.get("cos_secret_id") and cfg.get("cos_bucket"):
        try:
            from clients.cos_client import COSClient
            cos_client = COSClient(
                secret_id=cfg["cos_secret_id"],
                secret_key=cfg["cos_secret_key"],
                region=cfg["cos_region"],
                bucket=cfg["cos_bucket"]
            )
            print(f"✅ COS 客户端初始化成功: bucket={cfg['cos_bucket']}")
        except ImportError:
            print("⚠️ COS SDK 未安装，请运行: pip install cos-python-sdk-v5")
            cos_client = None
        except Exception as e:
            print(f"⚠️ COS 客户端初始化失败: {e}")
            cos_client = None
    else:
        print("ℹ️ COS 云存储未启用或配置不完整")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局配置
cfg = load_ini()

# 启动时初始化 COS 客户端
init_cos_client()


def detect_content_style(text: str, cfg: Dict[str, Any]) -> str:
    """
    使用LLM判断内容属于哪种风格
    
    参数:
        text: 内容文本
        cfg: 配置信息
    
    返回:
        风格代码: 'tech', 'business', 'life', 'culture', 'entertainment', 
                 'education', 'health', 'emotion', 'growth' 或 'general'
    """
    # 初始化LLM客户端
    api = HunyuanAPIClient(
        secret_id=cfg["hunyuan_api_secret_id"],
        secret_key=cfg["hunyuan_api_secret_key"],
        region=cfg["hunyuan_api_region"],
        model=cfg["hunyuan_api_model"],
        temperature=0.1,  # 使用低温度以获得确定性结果
        top_p=0.9,
        max_tokens=10,
    )
    
    # 构建提示词
    prompt = f"""
请判断以下内容最适合哪个类别，只回答类别名称，不要解释：

{text[:2000]}

可选类别：
1. 科技（技术、创新、数字产品、IT、人工智能、编程）
2. 商业（经济、创业、投资、市场营销、财经、金融）
3. 生活（日常、美食、旅行、家居、生活方式）
4. 文化（历史、艺术、文学、传统、人文）
5. 娱乐（电影、电视、音乐、游戏、综艺、轻松话题）
6. 教育（学习、考试、技能培训、知识科普）
7. 健康（养生、医疗、运动、饮食健康、心理健康）
8. 情感（恋爱、婚姻、人际关系、心理咨询）
9. 成长（个人发展、自我提升、职业规划、励志）

如果不属于以上任何类别，请回答"通用"。
请只回答一个词：科技、商业、生活、文化、娱乐、教育、健康、情感、成长或通用。
    """
    
    # 使用大写的 Role 和 Content（腾讯云混元 API 要求）
    messages = [
        {"Role": "system", "Content": "你是一个精确的文本分类助手，只输出单个分类结果，不做解释"},
        {"Role": "user", "Content": prompt},
    ]
    
    try:
        resp = api.chat(messages, stream=False)
        content = ""
        choices = resp.get("Choices") or resp.get("choices") or []
        if choices:
            msg = choices[0].get("Message") or choices[0].get("message") or {}
            content = msg.get("Content") or msg.get("content") or ""
        
        # 将中文回答映射到风格代码
        style_map = {
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
            "通用": "general"
        }
        
        # 提取关键词并映射
        for key, value in style_map.items():
            if key in content:
                print(f"检测到内容风格: {key} -> {value}")
                return value
        
        # 默认返回通用
        return "general"
    except Exception as e:
        print(f"LLM调用失败: {e}")
        return "general"


@app.get("/")
def root():
    return {"message": "Podcast Generator API", "version": "1.0"}


@app.post("/api/generate")
async def generate_podcast(
    mode: str = Form(...),
    host_mode: str = Form("dual"),  # 主持人模式 "single" 或 "dual"
    query: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    doc: Optional[str] = Form(None),
    instruction: Optional[str] = Form(None),
    style: str = Form("chat"),  # 播客风格
    custom_style: Optional[str] = Form(None),  # 自定义风格描述
    intro_style: str = Form("tongyong"),
    auto_detect: bool = Form(False),
    tts_speed: int = Form(0),
    voice_a: str = Form("501006:千嶂"),
    voice_b: Optional[str] = Form("601007:爱小叶"),  # 改为可选，单人模式不需要
    pdf_files: Optional[List[UploadFile]] = File(None)
):
    """
    生成播客
    完整移植自原版 app.py 的 ui_run 函数
    """
    try:
        # 解析音色（提取数字部分）
        voice_a_num = voice_a.split(":")[0] if ":" in voice_a else voice_a
        voice_b_num = voice_b.split(":")[0] if ":" in voice_b else voice_b
        tts_speed_val = int(tts_speed)
        
        print(f"🎤 选择的音色: voice_a={voice_a} -> {voice_a_num}, voice_b={voice_b} -> {voice_b_num}")

        # 用于存储 PDF 文档列表
        pdf_documents = []
        pdf_text = ""
        extracted_topic = ""

        # ========== 处理 PDF 文件（与原版一致）==========
        if mode == "PDF文件" and pdf_files:
            try:
                print(f"PDF文件类型: {type(pdf_files)}")
                
                # 保存上传的文件到临时目录
                temp_dir = tempfile.mkdtemp()
                file_paths = []
                
                for pdf_file in pdf_files:
                    file_path = os.path.join(temp_dir, pdf_file.filename)
                    with open(file_path, "wb") as f:
                        content = await pdf_file.read()
                        f.write(content)
                    file_paths.append(file_path)
                
                print(f"处理后的文件路径: {file_paths}")
                
                # 提取 PDF 文件内容
                if file_paths:
                    try:
                        # 使用 PDF 处理函数，返回每个文件的内容和文件名
                        pdf_documents = process_pdf_files(file_paths)
                        
                        if pdf_documents:
                            # 合并所有 PDF 文档的内容
                            pdf_text = merge_pdf_contents(pdf_documents)
                            print(f"PDF文本长度: {len(pdf_text) if pdf_text else 0}")
                            
                            # 重要：将模式设置为文档模式
                            mode = "文档"
                            
                            # 尝试从文本中提取主题作为查询
                            try:
                                # 从所有上传 PDF 各取一段样本进行主题提取
                                samples = []
                                for d in pdf_documents[:5]:
                                    title_part = d.get('title', '')
                                    text_part = (d.get('content') or '')[:30000]
                                    samples.append(f"【{title_part}】\n{text_part}")
                                content_sample = "\n\n".join(samples)[:150000]
                                
                                # 使用混元 API 提取主题
                                hunyuan_client = HunyuanAPIClient(
                                    secret_id=cfg["hunyuan_api_secret_id"],
                                    secret_key=cfg["hunyuan_api_secret_key"],
                                    region=cfg["hunyuan_api_region"]
                                )
                                extract_prompt = f"""请从以下文本中提取主要主题，用准确的短语表达，不要超过20个字：

{content_sample}

主题："""
                                # 使用大写的 Role 和 Content
                                response = hunyuan_client.chat([{"Role": "user", "Content": extract_prompt}])
                                choices = response.get("Choices") or response.get("choices") or []
                                if choices:
                                    msg = choices[0].get("Message") or choices[0].get("message") or {}
                                    topic = msg.get("Content") or msg.get("content") or ""
                                    if topic and len(topic) <= 50:
                                        extracted_topic = topic.strip()
                                        print(f"从文档提取的主题: {extracted_topic}")
                            except Exception as e:
                                print(f"提取主题异常: {e}")
                            
                            print(f"使用自定义方式处理{len(pdf_documents)}个PDF文档")
                            
                            # 将合并的文本设置为文档内容
                            doc = pdf_text
                        else:
                            print("PDF文本提取为空")
                            raise HTTPException(status_code=400, detail="无法从上传的PDF文件中提取文本。请确保文件是有效的PDF格式。")
                    except HTTPException:
                        raise
                    except Exception as e:
                        print(f"PDF处理异常: {e}")
                        raise HTTPException(status_code=400, detail=f"处理PDF文件时出错: {e}")
                else:
                    print("没有有效的文件路径")
                    raise HTTPException(status_code=400, detail="无法处理上传的PDF文件。请确保文件是PDF格式并重新上传。")
            except HTTPException:
                raise
            except Exception as e:
                print(f"PDF处理异常: {e}")
                print(traceback.format_exc())
                raise HTTPException(status_code=400, detail=f"处理PDF文件时出错: {e}")

        # ========== 自动检测片头风格（与原版一致）==========
        if auto_detect:
            if mode == "Query":
                detected_style = detect_content_style(query or "", cfg)
            elif mode == "URL":
                detected_style = detect_content_style(url or "", cfg)
            elif pdf_text:
                # 如果 PDF 内容过长，只取前 2000 字进行风格检测
                sample_text = pdf_text[:2000] if len(pdf_text) > 2000 else pdf_text
                detected_style = detect_content_style(sample_text, cfg)
            else:
                detected_style = detect_content_style(doc or "", cfg)
            
            # 使用检测到的风格
            intro_style = detected_style
            print(f"自动检测到的片头风格: {intro_style}")

        # ========== 调用播客生成流程（在线程池中运行）==========
        loop = asyncio.get_event_loop()
        
        if mode == "Query":
            res = await loop.run_in_executor(
                executor,
                lambda: run_end_to_end(
                    "query", query, 
                    style=style, custom_style=custom_style,
                    intro_style=intro_style, 
                    speed=tts_speed_val,
                    voice_a=voice_a_num, voice_b=voice_b_num, 
                    instruction=instruction,
                    host_mode=host_mode
                )
            )
        elif mode == "URL":
            res = await loop.run_in_executor(
                executor,
                lambda: run_end_to_end(
                    "url", url, 
                    style=style, custom_style=custom_style,
                    intro_style=intro_style, 
                    speed=tts_speed_val,
                    voice_a=voice_a_num, voice_b=voice_b_num, 
                    instruction=instruction,
                    host_mode=host_mode
                )
            )
        else:  # 文档模式
            # 构建增强指令（与原版一致）
            enhanced_instruction = instruction or ""
            
            # 提取文件标题列表
            file_titles = []
            if pdf_documents:
                file_titles = [doc_info.get("title", "") for doc_info in pdf_documents if doc_info.get("title")]
                print(f"上传的文件列表: {file_titles}")
                
                # 如果是 PDF 文件上传，使用提取的主题作为标题
                if extracted_topic:
                    if enhanced_instruction:
                        enhanced_instruction += "\n"
                    enhanced_instruction += f"主题：{extracted_topic}"
                    print(f"增强指令中添加主题：{extracted_topic}")
                
                # 明确要求均衡使用所有主要资料
                enhanced_instruction += "\n请综合所有上传的主要文档内容生成主题与脚本，确保每个主要资料至少引用一次，并尽量均衡使用各主要资料。"
            
            res = await loop.run_in_executor(
                executor,
                lambda: run_end_to_end(
                    "doc", doc,
                    style=style, custom_style=custom_style,
                    intro_style=intro_style,
                    speed=tts_speed_val,
                    voice_a=voice_a_num, voice_b=voice_b_num,
                    instruction=enhanced_instruction,
                    file_titles=file_titles,
                    pdf_documents=pdf_documents,
                    host_mode=host_mode
                )
            )
            
            # 如果有 PDF 文档，为前端显示重新构建 sources（只保留摘要）
            if pdf_documents:
                # 将原始的 sources 保存下来，作为补充资料
                supplementary_sources = [s for s in res.get("sources", []) if not s.get("is_primary", False)]
                
                # 创建新的 sources 列表，包含每个 PDF 文档作为主要资料（截断内容用于显示）
                new_sources = []
                for doc_info in pdf_documents:
                    content = doc_info.get("content", "")
                    clean_content = ''.join(char for char in content if char.isprintable() or char.isspace())
                    if not clean_content.strip():
                        clean_content = content
                    
                    # 限制内容长度（用于前端显示）
                    snippet = clean_content[:2000] + "..." if len(clean_content) > 2000 else clean_content
                    
                    new_sources.append({
                        "title": doc_info.get("title", "未知文档"),
                        "url": "",
                        "snippet": snippet,
                        "is_primary": True
                    })
                
                # 添加补充资料
                new_sources.extend(supplementary_sources)
                res["sources"] = new_sources

        # ========== 返回结果 ==========
        audio_path = res.get("audio_path", "")
        script = res.get("script", "")
        sources = res.get("sources", [])
        
        # 生成播客标题
        podcast_title = ""
        if mode == "Query":
            podcast_title = query[:50] if query else "未命名播客"
        elif mode == "URL":
            podcast_title = url[:50] if url else "未命名播客"
        elif extracted_topic:
            podcast_title = extracted_topic
        elif pdf_documents:
            podcast_title = pdf_documents[0].get("title", "未命名播客")[:50]
        else:
            podcast_title = "未命名播客"
        
        # 如果启用了 COS，上传音频和脚本到云存储
        audio_url = None
        script_url = None
        podcast_id = None
        
        if cos_client and audio_path:
            try:
                # 获取完整的本地文件路径
                if not os.path.isabs(audio_path):
                    local_audio_path = os.path.join("outputs", os.path.basename(audio_path))
                else:
                    local_audio_path = audio_path
                
                if os.path.exists(local_audio_path):
                    # 上传完整播客（音频 + 脚本 + 更新历史记录）
                    upload_result = cos_client.upload_podcast(
                        audio_path=local_audio_path,
                        script_content=script,
                        title=podcast_title,
                        sources=sources
                    )
                    audio_url = upload_result.get("audio_url")
                    script_url = upload_result.get("script_url")
                    podcast_id = upload_result.get("id")
                    print(f"✅ 播客已上传到 COS: id={podcast_id}")
                else:
                    print(f"⚠️ 音频文件不存在: {local_audio_path}")
            except Exception as e:
                print(f"⚠️ COS 上传失败: {e}")
                traceback.print_exc()

        return {
            "id": podcast_id,
            "audio_path": audio_path,
            "audio_url": audio_url,
            "script_url": script_url,
            "script": script,
            "sources": sources,
            "title": podcast_title,
            "host_mode": host_mode  # 新增：返回使用的主持人模式
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.post("/api/generate-script")
async def generate_script(
    mode: str = Form(...),
    host_mode: str = Form("dual"),
    query: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    doc: Optional[str] = Form(None),
    instruction: Optional[str] = Form(None),
    style: str = Form("chat"),
    custom_style: Optional[str] = Form(None),
    pdf_files: Optional[List[UploadFile]] = File(None)
):
    """
    第一阶段：只生成脚本，不合成语音
    返回脚本文本供用户编辑确认
    """
    try:
        pdf_documents = []
        pdf_text = ""
        extracted_topic = ""

        # 处理 PDF 文件
        if mode == "PDF文件" and pdf_files:
            try:
                temp_dir = tempfile.mkdtemp()
                file_paths = []
                
                for pdf_file in pdf_files:
                    file_path = os.path.join(temp_dir, pdf_file.filename)
                    with open(file_path, "wb") as f:
                        content = await pdf_file.read()
                        f.write(content)
                    file_paths.append(file_path)
                
                if file_paths:
                    pdf_documents = process_pdf_files(file_paths)
                    if pdf_documents:
                        pdf_text = merge_pdf_contents(pdf_documents)
                        mode = "文档"
                        
                        # 提取主题
                        try:
                            samples = []
                            for d in pdf_documents[:5]:
                                title_part = d.get('title', '')
                                text_part = (d.get('content') or '')[:30000]
                                samples.append(f"【{title_part}】\n{text_part}")
                            content_sample = "\n\n".join(samples)[:150000]
                            
                            hunyuan_client = HunyuanAPIClient(
                                secret_id=cfg["hunyuan_api_secret_id"],
                                secret_key=cfg["hunyuan_api_secret_key"],
                                region=cfg["hunyuan_api_region"]
                            )
                            extract_prompt = f"""请从以下文本中提取主要主题，用准确的短语表达，不要超过20个字：

{content_sample}

主题："""
                            response = hunyuan_client.chat([{"Role": "user", "Content": extract_prompt}])
                            choices = response.get("Choices") or response.get("choices") or []
                            if choices:
                                msg = choices[0].get("Message") or choices[0].get("message") or {}
                                topic = msg.get("Content") or msg.get("content") or ""
                                if topic and len(topic) <= 50:
                                    extracted_topic = topic.strip()
                        except Exception as e:
                            print(f"提取主题异常: {e}")
                        
                        doc = pdf_text
                    else:
                        raise HTTPException(status_code=400, detail="无法从PDF文件中提取文本")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"处理PDF文件时出错: {e}")

        # 调用脚本生成
        loop = asyncio.get_event_loop()
        
        if mode == "Query":
            res = await loop.run_in_executor(
                executor,
                lambda: generate_script_only(
                    "query", query,
                    style=style, custom_style=custom_style,
                    instruction=instruction,
                    host_mode=host_mode
                )
            )
        elif mode == "URL":
            res = await loop.run_in_executor(
                executor,
                lambda: generate_script_only(
                    "url", url,
                    style=style, custom_style=custom_style,
                    instruction=instruction,
                    host_mode=host_mode
                )
            )
        else:
            enhanced_instruction = instruction or ""
            file_titles = []
            if pdf_documents:
                file_titles = [doc_info.get("title", "") for doc_info in pdf_documents if doc_info.get("title")]
                if extracted_topic:
                    if enhanced_instruction:
                        enhanced_instruction += "\n"
                    enhanced_instruction += f"主题：{extracted_topic}"
                enhanced_instruction += "\n请综合所有上传的主要文档内容生成主题与脚本，确保每个主要资料至少引用一次。"
            
            res = await loop.run_in_executor(
                executor,
                lambda: generate_script_only(
                    "doc", doc,
                    style=style, custom_style=custom_style,
                    instruction=enhanced_instruction,
                    host_mode=host_mode,
                    file_titles=file_titles,
                    pdf_documents=pdf_documents
                )
            )
            
            # 重建 sources 用于前端显示
            if pdf_documents:
                supplementary_sources = [s for s in res.get("sources", []) if not s.get("is_primary", False)]
                new_sources = []
                for doc_info in pdf_documents:
                    content = doc_info.get("content", "")
                    clean_content = ''.join(char for char in content if char.isprintable() or char.isspace())
                    if not clean_content.strip():
                        clean_content = content
                    snippet = clean_content[:2000] + "..." if len(clean_content) > 2000 else clean_content
                    new_sources.append({
                        "title": doc_info.get("title", "未知文档"),
                        "url": "",
                        "snippet": snippet,
                        "is_primary": True
                    })
                new_sources.extend(supplementary_sources)
                res["sources"] = new_sources

        # 生成标题
        podcast_title = ""
        if mode == "Query":
            podcast_title = query[:50] if query else "未命名播客"
        elif mode == "URL":
            podcast_title = url[:50] if url else "未命名播客"
        elif extracted_topic:
            podcast_title = extracted_topic
        elif pdf_documents:
            podcast_title = pdf_documents[0].get("title", "未命名播客")[:50]
        else:
            podcast_title = "未命名播客"

        return {
            "script": res.get("script", ""),
            "sources": res.get("sources", []),
            "title": podcast_title,
            "host_mode": host_mode
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成脚本失败: {str(e)}")


@app.post("/api/synthesize")
async def synthesize_audio(
    script: str = Form(...),
    host_mode: str = Form("dual"),
    intro_style: str = Form("general"),
    tts_speed: int = Form(0),
    voice_a: str = Form("501006:千嶂"),
    voice_b: Optional[str] = Form("601007:爱小叶"),
    title: str = Form("未命名播客"),
    sources: Optional[str] = Form(None),  # JSON 字符串
    custom_intro_script: Optional[str] = Form(None),  # 自定义片头文案
    custom_intro_bgm: Optional[UploadFile] = File(None)  # 自定义片头BGM文件
):
    """
    第二阶段：根据用户确认的脚本合成语音
    """
    import json
    
    try:
        voice_a_num = voice_a.split(":")[0] if ":" in voice_a else voice_a
        voice_b_num = voice_b.split(":")[0] if ":" in voice_b else voice_b
        tts_speed_val = int(tts_speed)
        
        print(f"🎤 合成语音: voice_a={voice_a_num}, voice_b={voice_b_num}, host_mode={host_mode}, intro_style={intro_style}")
        
        # 处理自定义BGM文件
        custom_bgm_path = None
        if custom_intro_bgm and intro_style == "custom":
            # 保存上传的BGM文件到临时目录
            import tempfile
            temp_dir = tempfile.gettempdir()
            custom_bgm_path = os.path.join(temp_dir, f"custom_bgm_{custom_intro_bgm.filename}")
            with open(custom_bgm_path, "wb") as f:
                content = await custom_intro_bgm.read()
                f.write(content)
            print(f"📁 自定义BGM已保存: {custom_bgm_path}")
        
        # 解析 sources
        sources_list = []
        if sources:
            try:
                sources_list = json.loads(sources)
            except:
                sources_list = []
        
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            executor,
            lambda: synthesize_audio_only(
                script=script,
                intro_style=intro_style,
                speed=tts_speed_val,
                voice_a=voice_a_num,
                voice_b=voice_b_num,
                host_mode=host_mode,
                custom_intro_script=custom_intro_script if intro_style == "custom" else None,
                custom_intro_bgm=custom_bgm_path
            )
        )
        
        audio_path = res.get("audio_path", "")
        
        # 上传到 COS
        audio_url = None
        script_url = None
        podcast_id = None
        
        if cos_client and audio_path:
            try:
                if not os.path.isabs(audio_path):
                    local_audio_path = os.path.join("outputs", os.path.basename(audio_path))
                else:
                    local_audio_path = audio_path
                
                if os.path.exists(local_audio_path):
                    upload_result = cos_client.upload_podcast(
                        audio_path=local_audio_path,
                        script_content=script,
                        title=title,
                        sources=sources_list
                    )
                    audio_url = upload_result.get("audio_url")
                    script_url = upload_result.get("script_url")
                    podcast_id = upload_result.get("id")
                    print(f"✅ 播客已上传到 COS: id={podcast_id}")
            except Exception as e:
                print(f"⚠️ COS 上传失败: {e}")
                traceback.print_exc()

        return {
            "id": podcast_id,
            "audio_path": audio_path,
            "audio_url": audio_url,
            "script_url": script_url,
            "script": script,
            "title": title,
            "host_mode": host_mode
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"合成语音失败: {str(e)}")


@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    """获取音频文件"""
    audio_path = os.path.join("outputs", filename)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="音频文件不存在")
    return FileResponse(audio_path, media_type="audio/mpeg")


@app.get("/api/voices")
def get_voices():
    """获取可用的音色列表（包含试听 URL）"""
    nums = cfg.get("voice_numbers") or []  # 注意是 voice_numbers（复数）
    labels = cfg.get("voice_labels") or []  # 注意是 voice_labels（复数）
    choices = [
        {
            "value": f"{n}:{l}", 
            "label": f"{n}:{l}",
            "sample_url": f"/api/voice-sample/{n}"
        } 
        for n, l in zip(nums, labels)
    ]
    if not choices:
        choices = [
            {"value": "501006:千嶂", "label": "501006:千嶂", "sample_url": "/api/voice-sample/501006"},
            {"value": "601007:爱小叶", "label": "601007:爱小叶", "sample_url": "/api/voice-sample/601007"}
        ]
    return {"voices": choices}


@app.get("/api/voice-sample/{voice_id}")
def get_voice_sample(voice_id: str):
    """获取音色试听样本音频"""
    sample_path = os.path.join("assets", "voice_samples", f"voice_{voice_id}.mp3")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="音色样本不存在")
    return FileResponse(sample_path, media_type="audio/mpeg")


@app.get("/api/history")
def get_history(limit: int = 50):
    """获取播客历史记录列表（从 COS 读取）"""
    if not cos_client:
        return {"history": [], "message": "COS 未启用"}
    
    try:
        history = cos_client.get_history(limit=limit)
        return {"history": history}
    except Exception as e:
        print(f"获取历史记录失败: {e}")
        return {"history": [], "error": str(e)}


@app.get("/api/podcast/{podcast_id}")
def get_podcast_detail(podcast_id: str):
    """获取单个播客的详细信息（包含完整脚本）"""
    if not cos_client:
        raise HTTPException(status_code=503, detail="COS 未启用")
    
    try:
        detail = cos_client.get_podcast_detail(podcast_id)
        if detail:
            return detail
        else:
            raise HTTPException(status_code=404, detail="播客不存在")
    except HTTPException:
        raise
    except Exception as e:
        print(f"获取播客详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Slides Generation API Endpoints ==========

@app.post("/api/generate-slides")
async def generate_slides(
    script: str = Form(...),
    title: str = Form("未命名演示文稿"),
    style: str = Form("professional")  # professional/minimal/creative
):
    """
    从播客脚本生成 Slidev Markdown
    
    Args:
        script: 播客脚本文本
        title: 演示文稿标题
        style: 幻灯片风格 (professional/minimal/creative)
    
    Returns:
        {
            "markdown": str,  # Slidev 格式的 Markdown
            "title": str,
            "slide_count": int
        }
    
    Requirements: 1.1, 6.1
    """
    from pipeline.slides_generator import extract_key_points, count_slides
    
    try:
        # Validate input
        if not script or not script.strip():
            raise HTTPException(status_code=400, detail="脚本内容不能为空")
        
        # Generate Slidev Markdown using LLM
        loop = asyncio.get_event_loop()
        markdown = await loop.run_in_executor(
            executor,
            lambda: extract_key_points(cfg, script, title, style)
        )
        
        # Count slides
        slide_count = count_slides(markdown)
        
        return {
            "markdown": markdown,
            "title": title,
            "slide_count": slide_count
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成幻灯片失败: {str(e)}")


@app.post("/api/preview-slides")
async def preview_slides(
    markdown: str = Form(...)
):
    """
    预览 Slidev Markdown
    
    Args:
        markdown: Slidev 格式的 Markdown
    
    Returns:
        {
            "html": str,  # 渲染后的 HTML
            "slide_count": int
        }
    
    Requirements: 2.3, 3.1
    """
    from pipeline.slides_generator import render_preview_html, count_slides
    
    try:
        # Validate input
        if not markdown or not markdown.strip():
            raise HTTPException(status_code=400, detail="Markdown 内容不能为空")
        
        # Render HTML preview
        html = render_preview_html(markdown)
        
        # Count slides
        slide_count = count_slides(markdown)
        
        return {
            "html": html,
            "slide_count": slide_count
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"预览生成失败: {str(e)}")


@app.post("/api/export-slides")
async def export_slides(
    markdown: str = Form(...),
    format: str = Form("pdf"),  # pdf/pptx
    title: str = Form("presentation")
):
    """
    导出幻灯片为 PDF 或 PPTX
    
    Args:
        markdown: Slidev 格式的 Markdown
        format: 导出格式 (pdf/pptx)
        title: 文件名（不含扩展名）
    
    Returns:
        {
            "file_url": str,  # COS 文件 URL（如果上传成功）
            "file_path": str,  # 本地文件路径
            "format": str,
            "fallback_available": bool  # 是否有降级方案
        }
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 6.2, 7.3, 7.4
    """
    print(f"📊 开始导出幻灯片: format={format}, title={title}")
    
    try:
        from pipeline.slides_generator import export_to_pdf, export_to_pptx
    except ImportError as e:
        print(f"❌ 导入 slides_generator 失败: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"模块导入失败: {e}")
    
    try:
        # Validate input
        if not markdown or not markdown.strip():
            raise HTTPException(status_code=400, detail="Markdown 内容不能为空")
        
        if format not in ["pdf", "pptx"]:
            raise HTTPException(status_code=400, detail="格式必须是 pdf 或 pptx")
        
        # Create output directory if not exists
        output_dir = os.path.join("outputs", "slides")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        safe_title = "".join(c for c in title if c.isalnum() or c in "._- ").strip()[:50] or "presentation"
        filename = f"{safe_title}_{timestamp}_{unique_id}.{format}"
        output_path = os.path.join(output_dir, filename)
        
        # Export to requested format
        loop = asyncio.get_event_loop()
        
        print(f"📄 准备导出到: {output_path}")
        
        try:
            if format == "pdf":
                print("📄 调用 export_to_pdf...")
                file_path = await loop.run_in_executor(
                    executor,
                    lambda: export_to_pdf(markdown, output_path)
                )
            else:  # pptx
                print("📄 调用 export_to_pptx...")
                file_path = await loop.run_in_executor(
                    executor,
                    lambda: export_to_pptx(markdown, output_path)
                )
            print(f"✅ 导出成功: {file_path}")
        except RuntimeError as e:
            print(f"❌ RuntimeError: {e}")
            traceback.print_exc()
            # Export failed - provide fallback
            error_msg = str(e)
            if "内存" in error_msg or "memory" in error_msg.lower():
                raise HTTPException(
                    status_code=507,
                    detail={
                        "error": "资源不足，无法导出",
                        "message": error_msg,
                        "fallback_available": True,
                        "suggestion": "请下载 Markdown 文件，使用本地 Slidev CLI 导出"
                    }
                )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": f"{format.upper()} 导出失败",
                    "message": error_msg,
                    "fallback_available": True,
                    "suggestion": "请尝试其他格式或下载 Markdown 文件"
                }
            )
        
        # Try to upload to COS
        file_url = None
        if cos_client and os.path.exists(file_path):
            try:
                # Upload to COS（存储到 podcasts 文件夹，与音频共享公有读权限）
                date_prefix = datetime.now().strftime('%Y/%m/%d')
                cos_key = f"podcasts/{date_prefix}/slides_{filename}"
                
                cos_client.client.upload_file(
                    Bucket=cos_client.bucket,
                    LocalFilePath=file_path,
                    Key=cos_key,
                    PartSize=10,
                    MAXThread=5,
                    EnableMD5=True
                )
                
                # 使用直接 URL（与音频一致，存储桶需设置为公有读）
                file_url = f"https://{cos_client.bucket}.cos.{cos_client.region}.myqcloud.com/{cos_key}"
                print(f"✅ 幻灯片已上传到 COS: {file_url}")
                
            except Exception as e:
                print(f"⚠️ COS 上传失败，使用本地文件: {e}")
                traceback.print_exc()
                # Continue with local file path
        
        return {
            "file_url": file_url,
            "file_path": f"/api/slides-file/{filename}",
            "format": format,
            "fallback_available": True
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        print(f"❌ ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ 导出异常: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "导出失败",
                "message": str(e),
                "fallback_available": True,
                "suggestion": "请下载 Markdown 文件"
            }
        )


@app.get("/api/slides-file/{filename}")
def get_slides_file(filename: str):
    """获取导出的幻灯片文件"""
    file_path = os.path.join("outputs", "slides", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # Determine media type
    if filename.endswith('.pdf'):
        media_type = "application/pdf"
    elif filename.endswith('.pptx'):
        media_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    else:
        media_type = "application/octet-stream"
    
    return FileResponse(file_path, media_type=media_type, filename=filename)


# ========== Interview Mode API Endpoints (新增) ==========
# Feature: interview-podcast-mode

from pipeline.interview_agent import InterviewAgent, get_session as get_interview_session

# Global interview agent instance
_interview_agent = None

def get_interview_agent() -> InterviewAgent:
    """Get or create the interview agent instance."""
    global _interview_agent
    if _interview_agent is None:
        _interview_agent = InterviewAgent(cfg)
    return _interview_agent


@app.post("/api/interview/start")
async def start_interview() -> Dict[str, Any]:
    """
    开始新的采访会话
    
    Returns:
        {
            "session_id": str,
            "welcome_message": str
        }
    
    Requirements: 7.1, 2.1
    """
    try:
        agent = get_interview_agent()
        session = agent.start_session()
        
        welcome_message = (
            "你好！我是你的播客创作助手。今天我们来聊聊你想做的播客内容。\n\n"
            "你可以告诉我你想讨论的话题，分享你的想法和观点。"
            "在对话过程中，你也可以随时添加URL链接、上传文档，或者让我搜索某个话题的相关信息。\n\n"
            "那么，你想聊些什么呢？"
        )
        
        return {
            "session_id": session.session_id,
            "welcome_message": welcome_message
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")


@app.post("/api/interview/chat")
async def interview_chat(
    session_id: str = Form(...),
    message: str = Form(...),
    attached_material_ids: Optional[str] = Form(None)  # JSON array of material IDs
) -> Dict[str, Any]:
    """
    发送消息并获取 AI 回复
    
    Args:
        session_id: 会话ID
        message: 用户消息
        attached_material_ids: 附加的素材ID列表（JSON数组字符串）
    
    Returns:
        {
            "reply": str,
            "key_points": List[str],
            "message_count": int,
            "detected_materials": List[Dict]
        }
    
    Requirements: 7.2, 7.3
    """
    import json
    
    try:
        # Validate input
        if not message or not message.strip():
            raise HTTPException(status_code=400, detail="消息不能为空")
        
        agent = get_interview_agent()
        
        # Check if session exists
        session = get_interview_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # Parse attached material IDs
        material_ids = []
        if attached_material_ids:
            try:
                material_ids = json.loads(attached_material_ids)
                if not isinstance(material_ids, list):
                    material_ids = []
            except json.JSONDecodeError:
                material_ids = []
        
        # Process chat in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: agent.chat(session_id, message, material_ids if material_ids else None)
        )
        
        # Build detected materials from URLs
        detected_materials = []
        for url in result.get("detected_urls", []):
            detected_materials.append({
                "type": "url",
                "content": url,
                "status": "detected"
            })
        
        return {
            "reply": result.get("reply", ""),
            "key_points": result.get("key_points", []),
            "message_count": result.get("message_count", 0),
            "detected_materials": detected_materials
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理消息失败: {str(e)}")


@app.post("/api/interview/material")
async def add_interview_material(
    session_id: str = Form(...),
    material_type: str = Form(...),  # url/document/topic
    content: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
) -> Dict[str, Any]:
    """
    添加素材到会话
    
    Args:
        session_id: 会话ID
        material_type: 素材类型 (url/document/topic)
        content: URL地址 / 话题关键词 / 文档文本
        file: 上传的文件（可选）
    
    Returns:
        {
            "material_id": str,
            "summary": str,
            "ai_thoughts": str,
            "source": str
        }
    
    Requirements: 7.4, 3.1, 3.2, 3.3
    """
    try:
        # Validate material type
        if material_type not in ["url", "document", "topic"]:
            raise HTTPException(status_code=400, detail="素材类型必须是 url、document 或 topic")
        
        # Check if session exists
        session = get_interview_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        agent = get_interview_agent()
        
        # Handle file upload for document type
        material_content = content
        if material_type == "document" and file:
            # Save uploaded file to temp location
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                file_content = await file.read()
                f.write(file_content)
            material_content = file_path
        elif not content:
            raise HTTPException(status_code=400, detail="内容不能为空")
        
        # Process material in thread pool
        logger.info(f"Processing material: type={material_type}, content_preview={material_content[:100] if material_content else 'None'}...")
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: agent.add_material(session_id, material_type, material_content)
        )
        
        logger.info(f"Material processed successfully: id={result.get('id', '')}")
        
        return {
            "material_id": result.get("id", ""),
            "summary": result.get("summary", ""),
            "ai_thoughts": result.get("ai_thoughts", ""),
            "source": result.get("source", "")
        }
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"ValueError in add_interview_material: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Exception in add_interview_material: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"添加素材失败: {str(e)}")


@app.post("/api/interview/generate")
async def generate_interview_script(
    session_id: str = Form(...),
    host_mode: str = Form("dual")  # "dual" 双人播客（A/B交替），"single" 单人播客
) -> Dict[str, Any]:
    """
    根据采访内容生成播客脚本
    
    Args:
        session_id: 会话ID
        host_mode: 主持人模式，"dual" 为双人播客（默认），"single" 为单人播客
    
    Returns:
        {
            "script": str,          # 双人模式下每行以 A:/B: 开头
            "sources": List[Dict],
            "style_analysis": Dict,
            "warning": str,         # 如果对话太短
            "host_mode": str        # 返回使用的主持人模式
        }
    
    Requirements: 7.5, 4.1, 4.2
    """
    try:
        # Check if session exists
        session = get_interview_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # Validate host_mode
        if host_mode not in ("dual", "single"):
            host_mode = "dual"
        
        agent = get_interview_agent()
        
        # Generate script in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: agent.generate_script(session_id, host_mode=host_mode)
        )
        
        return {
            "script": result.get("script", ""),
            "sources": result.get("sources", []),
            "style_analysis": result.get("style_applied", {}),
            "warning": result.get("warning"),
            "host_mode": result.get("host_mode", host_mode)
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成脚本失败: {str(e)}")


@app.get("/api/interview/session/{session_id}")
async def get_interview_session_state(session_id: str) -> Dict[str, Any]:
    """
    获取会话状态
    
    Args:
        session_id: 会话ID
    
    Returns:
        {
            "session_id": str,
            "message_count": int,
            "key_points": List[str],
            "materials": List[Dict],
            "created_at": str
        }
    
    Requirements: 6.1, 6.2, 6.3
    """
    try:
        session = get_interview_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        return {
            "session_id": session.session_id,
            "message_count": len(session.messages),
            "key_points": [kp["point"] for kp in session.key_points],
            "materials": [
                {
                    "id": m["id"],
                    "type": m["type"],
                    "summary": m["summary"],
                    "source": m["source"],
                    "added_at": m["added_at"]
                }
                for m in session.materials
            ],
            "created_at": session.created_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取会话状态失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

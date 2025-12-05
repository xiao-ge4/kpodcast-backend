"""
FastAPI 版本的播客生成器 API
完整移植自 Gradio 版本 app.py
"""
import logging
# 抑制 pdfminer 的颜色解析警告
logging.getLogger("pdfminer").setLevel(logging.ERROR)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
import tempfile
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 创建线程池用于运行同步代码（如 Playwright）
executor = ThreadPoolExecutor(max_workers=4)

# 导入原有的功能模块
from pipeline.podcast_pipeline_new import run_end_to_end
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
    完全移植自原版 app.py
    
    参数:
        text: 内容文本
        cfg: 配置信息
    
    返回:
        风格名称: 'chengzhang', 'kejigan', 'shangye', 'yingshi', 'zhichang' 或 'tongyong'
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
    
    # 构建提示词（与原版一致）
    prompt = f"""
请判断以下内容最适合哪个类别，只回答类别名称，不要解释：

{text}

可选类别：
1. 成长（个人发展、自我提升、心理健康）
2. 科技（技术、创新、数字产品、IT）
3. 商业（经济、创业、投资、市场营销）
4. 影视（电影、电视、娱乐、艺术）
5. 职场（工作、职业发展、团队管理）

如果不属于以上任何类别，请回答"通用"。
请只回答一个词：成长、科技、商业、影视、职场或通用。
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
        
        # 将中文回答映射到文件名（与原版一致）
        style_map = {
            "成长": "chengzhang",
            "科技": "kejigan",
            "商业": "shangye",
            "影视": "yingshi",
            "职场": "zhichang",
            "通用": "tongyong"
        }
        
        # 提取关键词并映射
        for key, value in style_map.items():
            if key in content:
                return value
        
        # 默认返回通用
        return "tongyong"
    except Exception as e:
        print(f"LLM调用失败: {e}")
        return "tongyong"


@app.get("/")
def root():
    return {"message": "Podcast Generator API", "version": "1.0"}


@app.post("/api/generate")
async def generate_podcast(
    mode: str = Form(...),
    query: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    doc: Optional[str] = Form(None),
    instruction: Optional[str] = Form(None),
    style: str = Form("news"),
    intro_style: str = Form("tongyong"),
    auto_detect: bool = Form(False),
    tts_speed: int = Form(0),
    voice_a: str = Form("501006:千嶂"),
    voice_b: str = Form("601007:爱小叶"),
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
                    style=style, intro_style=intro_style, 
                    speed=tts_speed_val,
                    voice_a=voice_a_num, voice_b=voice_b_num, 
                    instruction=instruction
                )
            )
        elif mode == "URL":
            res = await loop.run_in_executor(
                executor,
                lambda: run_end_to_end(
                    "url", url, 
                    style=style, intro_style=intro_style, 
                    speed=tts_speed_val,
                    voice_a=voice_a_num, voice_b=voice_b_num, 
                    instruction=instruction
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
                    style=style, intro_style=intro_style,
                    speed=tts_speed_val,
                    voice_a=voice_a_num, voice_b=voice_b_num,
                    instruction=enhanced_instruction,
                    file_titles=file_titles,
                    pdf_documents=pdf_documents  # 传递完整的PDF文档列表
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
            "title": podcast_title
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    """获取音频文件"""
    audio_path = os.path.join("outputs", filename)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="音频文件不存在")
    return FileResponse(audio_path, media_type="audio/mpeg")


@app.get("/api/voices")
def get_voices():
    """获取可用的音色列表"""
    nums = cfg.get("voice_number") or []
    labels = cfg.get("voice_role") or []
    choices = [{"value": f"{n}:{l}", "label": f"{n}:{l}"} for n, l in zip(nums, labels)]
    if not choices:
        choices = [
            {"value": "501006:千嶂", "label": "501006:千嶂"},
            {"value": "601007:爱小叶", "label": "601007:爱小叶"}
        ]
    return {"voices": choices}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

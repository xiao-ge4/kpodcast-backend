import gradio as gr
import os
import tempfile
import uuid
import traceback
from typing import List, Dict, Any, Tuple, Union
# 使用新的pipeline模块
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline.podcast_pipeline_new import run_end_to_end
from utils.config_loader import load_ini
from clients.hunyuan_api_client import HunyuanAPIClient
from utils.pdf_loader import save_uploaded_files, process_pdf_files, extract_text_from_pdf, merge_pdf_contents

# 全局变量，用于存储PDF文档列表
# 这将在PDF文件上传时设置，并在生成参考资料时使用
pdf_documents_global = []



def detect_content_style(text: str, cfg: Dict[str, Any]) -> str:
    """
    使用LLM判断内容属于哪种风格
    
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
    
    # 构建提示词
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
        
        # 将中文回答映射到文件名
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


def ui_run(mode, query, instruction, url, doc, pdf_files, style, intro_style, tts_speed_val=0, voice_a=None, voice_b=None, auto_detect=False):
    try:
        # 重置全局变量
        global pdf_documents_global
        pdf_documents_global = []
        
        cfg = load_ini()
        
        # 处理PDF文件
        pdf_text = ""
        if mode == "PDF文件" and pdf_files is not None:
            try:
                print(f"PDF文件类型: {type(pdf_files)}")
                
                # 处理文件路径
                file_paths = []
                
                # 如果是列表类型
                if isinstance(pdf_files, list):
                    for file_item in pdf_files:
                        # 如果是字符串路径
                        if isinstance(file_item, str):
                            file_paths.append(file_item)
                        # 如果是Gradio文件对象
                        elif hasattr(file_item, 'name'):
                            file_paths.append(file_item.name)
                # 如果是单个文件
                elif hasattr(pdf_files, 'name'):
                    file_paths.append(pdf_files.name)
                
                print(f"处理后的文件路径: {file_paths}")
                
                # 提取PDF文件内容
                if file_paths:
                    try:
                        # 使用修改后的PDF处理函数，返回每个文件的内容和文件名
                        pdf_documents = process_pdf_files(file_paths)
                        
                        # 如果有PDF文档
                        if pdf_documents:
                            # 合并所有PDF文档的内容作为一个字符串，用于提取主题和其他处理
                            pdf_text = merge_pdf_contents(pdf_documents)
                            print(f"PDF文本长度: {len(pdf_text) if pdf_text else 0}")
                            
                            # 重要修改：将模式设置为文档模式，而不是PDF模式
                            mode = "文档"
                            
                            # 尝试从文本中提取主题作为查询
                            try:
                                # 从所有上传PDF各取一段样本进行主题提取，避免只关注第一个文档
                                samples = []
                                for d in pdf_documents[:5]:
                                    title_part = d.get('title', '')
                                    text_part = (d.get('content') or '')[:30000]
                                    samples.append(f"【{title_part}】\n{text_part}")
                                content_sample = "\n\n".join(samples)[:150000]
                                # 使用混元API提取主题
                                hunyuan_client = HunyuanAPIClient(
                                    secret_id=cfg["hunyuan_api_secret_id"],
                                    secret_key=cfg["hunyuan_api_secret_key"],
                                    region=cfg["hunyuan_api_region"]
                                )
                                extract_prompt = f"""请从以下文本中提取主要主题，用准确的短语表达，不要超过20个字：

{content_sample}

主题："""
                                # 使用正确的方法名和参数格式，注意大写的Role和Content
                                response = hunyuan_client.chat([{"Role": "user", "Content": extract_prompt}])
                                extracted_topic = response.get("Choices", [{}])[0].get("Message", {}).get("Content", "")
                                if extracted_topic and len(extracted_topic) <= 50:
                                    query = extracted_topic.strip()
                                    print(f"从文档提取的主题: {query}")
                                    
                                    # 将提取的文档主题作为查询主题
                                    # 这将在后续的指令增强中使用
                                    topic_or_url_or_text = query
                            except Exception as e:
                                print(f"提取主题异常: {e}")
                                # 如果提取失败，使用默认主题
                            
                            # 将PDF文档列表保存在全局变量中，供后续使用
                            # 不需要再次声明全局变量，因为已经在函数开始时声明了
                            pdf_documents_global = pdf_documents
                            
                            # 将合并的文本设置为文档内容
                            doc = pdf_text
                        else:
                            print("PDF文本提取为空")
                            return None, "错误：无法从上传的PDF文件中提取文本。请确保文件是有效的PDF格式。", []
                    except Exception as e:
                        print(f"PDF处理异常: {e}")
                        return None, f"错误：处理PDF文件时出错 - {e}", []
                else:
                    print("没有有效的文件路径")
                    return None, "错误：无法处理上传的PDF文件。请确保文件是PDF格式并重新上传。", []
            except Exception as e:
                import traceback
                print(f"PDF处理异常: {e}")
                print(traceback.format_exc())
                return None, f"错误：处理PDF文件时出错 - {e}", []
        
        # 如果启用自动检测，根据内容判断片头风格
        if auto_detect:
            if mode == "Query":
                detected_style = detect_content_style(query, cfg)
            elif mode == "URL":
                detected_style = detect_content_style(url, cfg)
            elif mode == "PDF":
                # 如果PDF内容过长，只取前2000字进行风格检测
                sample_text = pdf_text[:2000] if len(pdf_text) > 2000 else pdf_text
                detected_style = detect_content_style(sample_text, cfg)
            else:
                detected_style = detect_content_style(doc, cfg)
            
            # 使用检测到的风格
            intro_style = detected_style
        
        # 调用播客生成流程
        try:
            # 修改文档模式的处理逻辑
            if mode == "Query":
                res = run_end_to_end("query", query, style=style, intro_style=intro_style, speed=tts_speed_val,
                                    voice_a=voice_a, voice_b=voice_b, instruction=instruction)
            elif mode == "URL":
                res = run_end_to_end("url", url, style=style, intro_style=intro_style, speed=tts_speed_val,
                                    voice_a=voice_a, voice_b=voice_b, instruction=instruction)
            else:  # 文档模式
                # 如果是PDF文件上传，使用提取的主题作为标题
                if 'topic_or_url_or_text' in locals() and isinstance(topic_or_url_or_text, str) and topic_or_url_or_text:
                    # 在指令中添加主题信息
                    enhanced_instruction = instruction or ""
                    if enhanced_instruction:
                        enhanced_instruction += "\n"
                    enhanced_instruction += f"主题：{topic_or_url_or_text}"
                    # 明确要求均衡使用所有主要资料
                    enhanced_instruction += "\n请综合所有上传的主要文档内容生成主题与脚本，确保每个主要资料至少引用一次，并尽量均衡使用各主要资料。"
                    print(f"增强指令中添加主题：{topic_or_url_or_text}")
                else:
                    enhanced_instruction = instruction or ""
                    enhanced_instruction += "\n请综合所有上传的主要文档内容生成主题与脚本，确保每个主要资料至少引用一次，并尽量均衡使用各主要资料。"
                
                # 检查是否有上传的PDF文档
                if 'pdf_documents_global' in globals() and pdf_documents_global:
                    # 如果有上传的PDF文档，使用自定义的方式处理
                    print(f"使用自定义方式处理{len(pdf_documents_global)}个PDF文档")
                    
                    # 调用原始的run_end_to_end函数，但在返回结果中替换sources
                    res = run_end_to_end("doc", doc, style=style, intro_style=intro_style, speed=tts_speed_val,
                                    voice_a=voice_a, voice_b=voice_b, instruction=enhanced_instruction)
                    
                    # 将原始的sources保存下来，作为补充资料
                    supplementary_sources = [s for s in res.get("sources", []) if not s.get("is_primary", False)]
                    
                    # 创建新的sources列表，包含每个PDF文档作为主要资料
                    new_sources = []
                    
                    # 将每个PDF文档添加为主要资料
                    for i, doc_info in enumerate(pdf_documents_global):
                        # 处理文档内容，清理不可打印字符
                        content = doc_info["content"]
                        clean_content = ''.join(char for char in content if char.isprintable() or char.isspace())
                        
                        # 如果清理后的内容为空，使用原始内容
                        if not clean_content.strip():
                            clean_content = content
                        
                        # 限制内容长度
                        snippet = clean_content[:2000] + "..." if len(clean_content) > 2000 else clean_content
                        
                        new_sources.append({
                            "title": doc_info["title"],
                            "url": "",
                            "snippet": snippet,
                            "is_primary": True
                        })
                    
                    # 添加补充资料
                    new_sources.extend(supplementary_sources)
                    
                    # 替换原始的sources
                    res["sources"] = new_sources
                else:
                    # 如果没有上传的PDF文档，使用原始的方式处理
                    res = run_end_to_end("doc", doc, style=style, intro_style=intro_style, speed=tts_speed_val,
                                    voice_a=voice_a, voice_b=voice_b, instruction=enhanced_instruction)
            
            # 证据展示（DataFrame 需要二维数组/表格）
            rows = []
            for i, s in enumerate(res.get("sources", [])):
                # 获取标题和片段
                title = s.get("title", "")
                url = s.get("url", "")
                snippet = s.get("snippet", "") or ""
                
                # 处理片段内容，确保其可读性
                # 限制长度并清理特殊字符
                clean_snippet = ''.join(char for char in snippet[:300] if char.isprintable() or char.isspace())
                
                # 添加到行中
                rows.append([i + 1, title, url, clean_snippet])
            
            return res.get("audio_path"), res.get("script", ""), rows
        except Exception as e:
            return None, f"错误：{e}", []
    except Exception as e:
        return None, f"错误：{e}", []


cfg = load_ini()
_nums = cfg.get("voice_number") or []
_labels = cfg.get("voice_role") or []
_choices = [f"{n}:{l}" for n, l in zip(_nums, _labels)] or []
if not _choices:
    _choices = ["501006:千嶂", "601007:爱小叶"]

with gr.Blocks(title="播客生成器") as demo:
    gr.Markdown("# 🎤️ 播客生成器（MVP）")
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(["Query", "URL", "文档", "PDF文件"], value="Query", label="输入类型")
            
            # 创建所有输入组件
            query = gr.Textbox(label="主题 Query")
            instruction = gr.Textbox(label="特殊指令", placeholder="可选，例如：风格轻松一点、分三段讲、Generate in English等")
            url = gr.Textbox(label="URL", visible=False)
            doc = gr.Textbox(label="长文档(可直接贴文本)", lines=10, visible=False)
            pdf_files = gr.File(label="上传PDF文件", file_types=[".pdf"], file_count="multiple", visible=False)
            
            # 共用设置组件
            style = gr.Dropdown(["news", "history", "science", "business"], value="news", label="题材模板")
            intro_style = gr.Dropdown(["chengzhang", "kejigan", "shangye", "yingshi", "zhichang", "tongyong"], value="tongyong", label="片头风格", info="选择播客片头风格")
            auto_detect = gr.Checkbox(label="自动检测片头风格", value=False, info="启用后，将根据内容自动选择合适的片头风格")
            tts_speed = gr.Slider(minimum=-2, maximum=2, value=0, step=1, label="语速(-2慢 ~ 2快)")
            with gr.Row():
                voice_a = gr.Dropdown(choices=_choices, label="角色A音色", value=_choices[0] if _choices else None)
                voice_b = gr.Dropdown(choices=_choices, label="角色B音色", value=(_choices[1] if len(_choices)>1 else (_choices[0] if _choices else None)))
            btn = gr.Button("开始生成", variant="primary")
            
            # 添加模式切换时的显示/隐藏逻辑
            def update_visibility(mode_value):
                if mode_value == "Query":
                    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                elif mode_value == "URL":
                    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
                elif mode_value == "文档":
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                elif mode_value == "PDF文件":
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
            
            # 注册模式切换事件
            mode.change(update_visibility, inputs=[mode], outputs=[query, url, doc, pdf_files])
        with gr.Column(scale=1):
            audio = gr.Audio(label="生成音频", type="filepath")
        with gr.Column(scale=1):
            transcript = gr.Textbox(label="逐字稿", lines=20)
            refs = gr.Dataframe(headers=["index","title","url","snippet"], label="参考资料", wrap=True)

    btn.click(ui_run, inputs=[mode, query, instruction, url, doc, pdf_files, style, intro_style, tts_speed, voice_a, voice_b, auto_detect], outputs=[audio, transcript, refs])
    
    gr.Markdown("""
    ### 输入类型说明
    - **Query**: 输入一个主题查询，系统将自动搜索相关内容并生成播客
    - **URL**: 输入一个网页链接，系统将提取网页内容并生成播客
    - **文档**: 直接输入或粘贴文本内容，系统将基于该内容生成播客
    - **PDF文件**: 上传一个或多个PDF文件，系统将提取PDF内容并生成播客
    
    ### 特殊指令说明
    在特殊指令框中，您可以输入各种指令来影响播客生成，例如：
    - 风格指令：“风格轻松一点”、“使用专业语言”
    - 结构指令：“分三段讲”、“重点关注经济影响”
    - 语言指令：“Generate in English”、“使用英文”
    
    ### 片头风格说明
    - **成长**：个人发展、自我提升、心理健康相关内容
    - **科技**：技术、创新、数字产品、IT相关内容
    - **商业**：经济、创业、投资、市场营销相关内容
    - **影视**：电影、电视、娱乐、艺术相关内容
    - **职场**：工作、职业发展、团队管理相关内容
    - **通用**：不属于以上类别的通用内容
    
    勾选“自动检测片头风格”后，系统将根据内容自动选择最合适的片头风格。
    """)
    
    gr.Markdown("""
    ### PDF文件上传说明
    - 支持上传一个或多个PDF文件
    - 系统将自动提取PDF文件中的文本内容
    - 如果上传多个PDF文件，系统将合并所有文件的内容
    - 如果PDF文件较大，处理可能需要一些时间
    - 如果PDF文件内容无法提取，请尝试复制内容并使用“文档”模式
    """)
    

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)

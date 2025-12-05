"""
PDF文档加载工具
支持从PDF文件中提取文本内容
"""
from typing import Dict, Any, List, Optional
import os
import logging
import tempfile
import PyPDF2
import pdfplumber

# 配置日志
logger = logging.getLogger(__name__)

def extract_text_from_pdf_pypdf2(file_path: str) -> str:
    """
    使用PyPDF2从文件中提取文本
    
    参数:
        file_path: PDF文件路径
        
    返回:
        提取的文本内容
    """
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text() or ""
                # 处理编码问题
                try:
                    # 尝试不同的编码方式
                    if not page_text.strip() and hasattr(page, "_extract_text"):
                        # 尝试直接访问底层方法
                        page_text = page._extract_text() or ""
                except Exception as inner_e:
                    logger.warning(f"PyPDF2页面文本提取内部错误: {inner_e}")
                
                # 清理文本中的不可打印字符
                page_text = ''.join(char for char in page_text if char.isprintable() or char.isspace())
                text += page_text + "\n\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PyPDF2提取文本失败: {e}")
        return ""

def extract_text_from_pdf_pdfplumber(file_path: str) -> str:
    """
    使用pdfplumber从PDF文件中提取文本
    
    参数:
        file_path: PDF文件路径
        
    返回:
        提取的文本内容
    """
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                try:
                    # 尝试提取文本
                    page_text = page.extract_text() or ""
                    
                    # 如果提取的文本为空，尝试其他方法
                    if not page_text.strip():
                        # 尝试提取表格数据
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                for row in table:
                                    page_text += " ".join([cell or "" for cell in row if cell]) + "\n"
                    
                    # 清理文本中的不可打印字符
                    page_text = ''.join(char for char in page_text if char.isprintable() or char.isspace())
                    text += page_text + "\n\n"
                except Exception as page_e:
                    logger.warning(f"pdfplumber页面文本提取错误: {page_e}")
        return text.strip()
    except Exception as e:
        logger.error(f"pdfplumber提取文本失败: {e}")
        return ""

def extract_text_from_pdf(file_path: str) -> str:
    """
    从PDF文件中提取文本，尝试多种方法
    
    参数:
        file_path: PDF文件路径
        
    返回:
        提取的文本内容
    """
    # 首先尝试使用pdfplumber
    text = extract_text_from_pdf_pdfplumber(file_path)
    
    # 如果pdfplumber提取失败或提取内容为空，尝试使用PyPDF2
    if not text:
        text = extract_text_from_pdf_pypdf2(file_path)
    
    return text

def process_pdf_files(file_paths: List[str]) -> List[Dict[str, str]]:
    """
    处理多个PDF文件，提取文本内容并返回文件名和内容的列表
    
    参数:
        file_paths: PDF文件路径列表
        
    返回:
        包含每个文件名和内容的字典列表
    """
    pdf_documents = []
    
    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                continue
                
            if not file_path.lower().endswith('.pdf'):
                logger.warning(f"不是PDF文件: {file_path}")
                continue
                
            text = extract_text_from_pdf(file_path)
            if text:
                file_name = os.path.basename(file_path)
                pdf_documents.append({
                    "title": file_name,
                    "content": text
                })
            else:
                logger.warning(f"无法从文件中提取文本: {file_path}")
        except Exception as e:
            logger.error(f"处理PDF文件时出错: {file_path}, 错误: {e}")
    
    return pdf_documents

def merge_pdf_contents(pdf_documents: List[Dict[str, str]]) -> str:
    """
    合并多个PDF文档的内容为一个字符串
    
    参数:
        pdf_documents: 包含每个文件名和内容的字典列表
        
    返回:
        合并后的文本内容
    """
    all_text = []
    
    for doc in pdf_documents:
        all_text.append(f"--- 文件: {doc['title']} ---\n{doc['content']}\n")
    
    return "\n\n".join(all_text)

def save_uploaded_files(uploaded_files: List[Any]) -> List[str]:
    """
    保存上传的文件到临时目录
    
    参数:
        uploaded_files: 上传的文件列表
        
    返回:
        保存后的文件路径列表
    """
    file_paths = []
    
    # 输出调试信息
    logger.info(f"Received uploaded_files: {type(uploaded_files)}")
    
    # 如果没有上传文件，直接返回空列表
    if uploaded_files is None:
        logger.warning("No files uploaded")
        return file_paths
    
    # 确保上传文件是列表
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
        logger.info("Converted single file to list")
    
    logger.info(f"Processing {len(uploaded_files)} files")
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            logger.info(f"Processing file {i}: {type(uploaded_file)}")
            
            # 处理bytes类型的数据
            if isinstance(uploaded_file, bytes):
                # 生成一个随机文件名
                import uuid
                temp_dir = tempfile.gettempdir()
                file_name = f"uploaded_{uuid.uuid4().hex}.pdf"
                file_path = os.path.join(temp_dir, file_name)
                
                # 保存文件内容
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file)
                    
                file_paths.append(file_path)
                logger.info(f"Saved bytes data to {file_path}")
            
            # 处理Gradio的二进制文件上传
            elif isinstance(uploaded_file, tuple) and len(uploaded_file) == 2:
                # Gradio的二进制文件上传格式为(file_name, file_data)
                file_name, file_data = uploaded_file
                temp_dir = tempfile.gettempdir()
                file_path = os.path.join(temp_dir, os.path.basename(file_name))
                
                # 保存文件内容
                with open(file_path, 'wb') as f:
                    f.write(file_data)
                    
                file_paths.append(file_path)
                logger.info(f"Saved binary file to {file_path}")
            
            # 处理常规文件对象
            elif hasattr(uploaded_file, 'name') and hasattr(uploaded_file, 'read'):
                # 正常的文件对象
                temp_dir = tempfile.gettempdir()
                file_name = os.path.basename(uploaded_file.name)
                file_path = os.path.join(temp_dir, file_name)
                
                # 保存文件内容
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.read())
                    
                file_paths.append(file_path)
                logger.info(f"Saved file object to {file_path}")
            
            # 处理字典形式的文件
            elif isinstance(uploaded_file, dict):
                if 'name' in uploaded_file and 'data' in uploaded_file:
                    # Gradio文件对象格式
                    temp_dir = tempfile.gettempdir()
                    file_name = os.path.basename(uploaded_file['name'])
                    file_path = os.path.join(temp_dir, file_name)
                    
                    # 保存文件内容
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file['data'])
                        
                    file_paths.append(file_path)
                    logger.info(f"Saved dict file to {file_path}")
                else:
                    logger.warning(f"Dict missing required keys: {uploaded_file.keys()}")
            
            # 处理字符串路径
            elif isinstance(uploaded_file, str):
                if os.path.exists(uploaded_file):
                    file_paths.append(uploaded_file)
                    logger.info(f"Using existing file path: {uploaded_file}")
                else:
                    logger.warning(f"File path does not exist: {uploaded_file}")
            
            # 其他类型
            else:
                logger.warning(f"Unsupported file object type: {type(uploaded_file)}")
                if hasattr(uploaded_file, '__dict__'):
                    logger.warning(f"Object attributes: {uploaded_file.__dict__}")
                
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}", exc_info=True)
    
    return file_paths

if __name__ == "__main__":
    # 测试代码
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        text = extract_text_from_pdf(pdf_path)
        print(f"提取的文本 ({len(text)} 字符):")
        print(text[:500] + "..." if len(text) > 500 else text)
    else:
        print("请提供PDF文件路径")

from typing import List, Dict, Any
import logging

# 导入增强版网页内容提取器
from utils.enhanced_url_fetcher import fetch_url, fetch_url_enhanced

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("doc_loader")

# 注意：fetch_url 函数现在由 enhanced_url_fetcher 模块提供
# 为了向后兼容，保持相同的接口


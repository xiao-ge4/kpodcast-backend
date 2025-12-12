"""
腾讯云 COS 客户端
用于上传播客音频、脚本文件到云存储，并管理历史记录
"""
from qcloud_cos import CosConfig, CosS3Client
import os
import uuid
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# 历史记录索引文件路径
HISTORY_INDEX_KEY = "podcasts/history_index.json"
MAX_HISTORY_ITEMS = 100  # 最多保留100条历史记录


class COSClient:
    """腾讯云 COS 客户端"""
    
    def __init__(self, secret_id: str, secret_key: str, region: str, bucket: str):
        """
        初始化 COS 客户端
        
        参数:
            secret_id: 腾讯云 SecretId
            secret_key: 腾讯云 SecretKey
            region: 存储桶地域，如 ap-guangzhou
            bucket: 存储桶名称，如 kpodcast-audio-1234567890
        """
        self.region = region
        self.bucket = bucket
        
        config = CosConfig(
            Region=region,
            SecretId=secret_id,
            SecretKey=secret_key,
            Token=None,
            Scheme='https',
            Timeout=120  # 增加超时时间到 120 秒
        )
        self.client = CosS3Client(config)
        logger.info(f"COS 客户端初始化成功: bucket={bucket}, region={region}")
    
    def upload_audio(self, local_path: str, custom_filename: Optional[str] = None) -> str:
        """
        上传音频文件到 COS
        
        参数:
            local_path: 本地文件路径
            custom_filename: 自定义文件名（可选）
            
        返回:
            公开访问的 URL
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"文件不存在: {local_path}")
        
        # 生成云端文件路径
        ext = os.path.splitext(local_path)[1] or '.mp3'
        date_prefix = datetime.now().strftime('%Y/%m/%d')
        
        if custom_filename:
            # 使用自定义文件名
            key = f"podcasts/{date_prefix}/{custom_filename}"
        else:
            # 生成唯一文件名
            unique_id = uuid.uuid4().hex[:12]
            timestamp = datetime.now().strftime('%H%M%S')
            key = f"podcasts/{date_prefix}/podcast_{timestamp}_{unique_id}{ext}"
        
        try:
            # 上传文件
            response = self.client.upload_file(
                Bucket=self.bucket,
                LocalFilePath=local_path,
                Key=key,
                PartSize=10,  # 分块大小 10MB
                MAXThread=5,  # 最大线程数
                EnableMD5=True  # 开启 MD5 校验
            )
            
            # 构建公开访问 URL
            url = f"https://{self.bucket}.cos.{self.region}.myqcloud.com/{key}"
            logger.info(f"音频上传成功: {url}")
            
            return url
            
        except Exception as e:
            logger.error(f"音频上传失败: {e}")
            raise
    
    def delete_audio(self, key: str) -> bool:
        """
        删除 COS 上的音频文件
        
        参数:
            key: 文件路径（不含域名）
            
        返回:
            是否删除成功
        """
        try:
            self.client.delete_object(
                Bucket=self.bucket,
                Key=key
            )
            logger.info(f"音频删除成功: {key}")
            return True
        except Exception as e:
            logger.error(f"音频删除失败: {e}")
            return False
    
    def get_audio_url(self, key: str) -> str:
        """
        获取音频文件的公开访问 URL
        
        参数:
            key: 文件路径
            
        返回:
            公开访问 URL
        """
        return f"https://{self.bucket}.cos.{self.region}.myqcloud.com/{key}"
    
    def check_file_exists(self, key: str) -> bool:
        """
        检查文件是否存在
        
        参数:
            key: 文件路径
            
        返回:
            是否存在
        """
        try:
            self.client.head_object(
                Bucket=self.bucket,
                Key=key
            )
            return True
        except Exception:
            return False
    
    def upload_script(self, script_content: str, audio_key: str) -> str:
        """
        上传脚本文本到 COS
        
        参数:
            script_content: 脚本文本内容
            audio_key: 对应音频文件的 key（用于生成相同路径的脚本文件）
            
        返回:
            公开访问的 URL
        """
        # 生成脚本文件 key（与音频同目录，扩展名改为 .txt）
        script_key = audio_key.rsplit('.', 1)[0] + '.txt'
        
        try:
            # 上传文本内容
            self.client.put_object(
                Bucket=self.bucket,
                Body=script_content.encode('utf-8'),
                Key=script_key,
                ContentType='text/plain; charset=utf-8'
            )
            
            url = f"https://{self.bucket}.cos.{self.region}.myqcloud.com/{script_key}"
            logger.info(f"脚本上传成功: {url}")
            return url
            
        except Exception as e:
            logger.error(f"脚本上传失败: {e}")
            raise
    
    def upload_podcast(self, audio_path: str, script_content: str, title: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        上传完整的播客（音频 + 脚本）并更新历史记录
        
        参数:
            audio_path: 本地音频文件路径
            script_content: 脚本文本内容
            title: 播客标题
            sources: 参考来源列表
            
        返回:
            包含 audio_url, script_url, id 的字典
        """
        # 生成唯一 ID
        podcast_id = uuid.uuid4().hex[:12]
        date_prefix = datetime.now().strftime('%Y/%m/%d')
        timestamp = datetime.now().strftime('%H%M%S')
        
        # 上传音频
        audio_key = f"podcasts/{date_prefix}/podcast_{timestamp}_{podcast_id}.mp3"
        self.client.upload_file(
            Bucket=self.bucket,
            LocalFilePath=audio_path,
            Key=audio_key,
            PartSize=10,
            MAXThread=5,
            EnableMD5=True
        )
        audio_url = f"https://{self.bucket}.cos.{self.region}.myqcloud.com/{audio_key}"
        
        # 上传脚本
        script_key = f"podcasts/{date_prefix}/podcast_{timestamp}_{podcast_id}.txt"
        self.client.put_object(
            Bucket=self.bucket,
            Body=script_content.encode('utf-8'),
            Key=script_key,
            ContentType='text/plain; charset=utf-8'
        )
        script_url = f"https://{self.bucket}.cos.{self.region}.myqcloud.com/{script_key}"
        
        logger.info(f"播客上传成功: audio={audio_url}, script={script_url}")
        
        # 更新历史记录索引
        history_item = {
            "id": podcast_id,
            "title": title[:100] if title else "未命名播客",
            "audio_url": audio_url,
            "script_url": script_url,
            "script_preview": script_content[:200] + "..." if len(script_content) > 200 else script_content,
            "sources": sources or [],
            "created_at": datetime.now().isoformat()
        }
        self._add_to_history(history_item)
        
        return {
            "id": podcast_id,
            "audio_url": audio_url,
            "script_url": script_url
        }
    
    def _add_to_history(self, item: Dict[str, Any]):
        """添加记录到历史索引"""
        try:
            # 读取现有历史
            history = self.get_history()
            
            # 添加新记录到开头
            history.insert(0, item)
            
            # 限制数量
            if len(history) > MAX_HISTORY_ITEMS:
                history = history[:MAX_HISTORY_ITEMS]
            
            # 保存回 COS
            self.client.put_object(
                Bucket=self.bucket,
                Body=json.dumps(history, ensure_ascii=False, indent=2).encode('utf-8'),
                Key=HISTORY_INDEX_KEY,
                ContentType='application/json; charset=utf-8'
            )
            logger.info(f"历史记录已更新，当前共 {len(history)} 条")
            
        except Exception as e:
            logger.error(f"更新历史记录失败: {e}")
    
    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取历史记录列表
        
        参数:
            limit: 返回的最大记录数
            
        返回:
            历史记录列表
        """
        try:
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=HISTORY_INDEX_KEY
            )
            content = response['Body'].get_raw_stream().read().decode('utf-8')
            history = json.loads(content)
            return history[:limit]
        except Exception as e:
            # 文件不存在或解析失败，返回空列表
            logger.info(f"读取历史记录: {e}")
            return []
    
    def get_podcast_detail(self, podcast_id: str) -> Optional[Dict[str, Any]]:
        """
        获取单个播客的详细信息
        
        参数:
            podcast_id: 播客 ID
            
        返回:
            播客详情，包含完整脚本
        """
        history = self.get_history(limit=MAX_HISTORY_ITEMS)
        for item in history:
            if item.get("id") == podcast_id:
                # 获取完整脚本
                try:
                    script_url = item.get("script_url", "")
                    if script_url:
                        # 从 URL 提取 key
                        key = script_url.split(f".cos.{self.region}.myqcloud.com/")[1]
                        response = self.client.get_object(
                            Bucket=self.bucket,
                            Key=key
                        )
                        full_script = response['Body'].get_raw_stream().read().decode('utf-8')
                        item["script"] = full_script
                except Exception as e:
                    logger.error(f"获取脚本失败: {e}")
                    item["script"] = item.get("script_preview", "")
                return item
        return None

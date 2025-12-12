#!/bin/bash
# KPodcast 后端部署脚本 - 腾讯云轻量服务器
# 使用方法: bash deploy_tencent.sh

set -e

echo "=========================================="
echo "  KPodcast 后端部署脚本"
echo "=========================================="

# 1. 更新系统
echo "[1/7] 更新系统包..."
sudo apt update && sudo apt upgrade -y

# 2. 安装 Python 3.11 和依赖
echo "[2/7] 安装 Python 3.11..."
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# 3. 安装系统依赖（音频处理需要）
echo "[3/7] 安装系统依赖..."
sudo apt install -y ffmpeg git nginx

# 4. 创建项目目录
echo "[4/7] 创建项目目录..."
sudo mkdir -p /opt/kpodcast
sudo chown $USER:$USER /opt/kpodcast
cd /opt/kpodcast

# 5. 克隆代码
echo "[5/7] 克隆代码..."
if [ -d "backend" ]; then
    cd backend && git pull && cd ..
else
    git clone https://github.com/xiao-ge4/kpodcast-backend.git backend
fi

# 6. 创建虚拟环境并安装依赖
echo "[6/7] 安装 Python 依赖..."
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 7. 创建配置文件模板
echo "[7/7] 创建配置文件..."
if [ ! -f "config.ini" ]; then
    cat > config.ini << 'EOF'
[bocha]
base_url = https://api.bochaai.com
search_path = /v1/web-search
api_id = YOUR_BOCHA_API_ID
api_key = YOUR_BOCHA_API_KEY

[hunyuan]
model_id = tencent/Hunyuan-MT-7B

[tts]
provider = tencent

[tencent]
secret_id = YOUR_TENCENT_SECRET_ID
secret_key = YOUR_TENCENT_SECRET_KEY
region = ap-beijing
voice_number = [101001, 101015, 501001, 501006, 501005, 502005, 601009, 601007, 501002, 501003, 501008, 501009]
voice_role = ["智瑜", "智萌", "智兰", "千嶂", "飞镜", "智小解", "爱小芊", "爱小叶", "智菊", "智宇", "WeJames", "WeWinny"]

[storage]
output_dir = ./outputs
assets_bgm_dir = ./assets/bgm

[hunyuan_api]
model = hunyuan-turbos-latest
temperature = 0.8
top_p = 0.8
max_tokens = 10000

[search]
supplementary_search_count = 4

[cos]
enabled = true
secret_id = YOUR_COS_SECRET_ID
secret_key = YOUR_COS_SECRET_KEY
region = ap-beijing
bucket = podcast-1308411753
EOF
    echo "⚠️  请编辑 /opt/kpodcast/backend/config.ini 填入你的 API 密钥"
fi

# 创建 outputs 目录
mkdir -p outputs outputs/slides

echo ""
echo "=========================================="
echo "  基础安装完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 编辑配置文件: nano /opt/kpodcast/backend/config.ini"
echo "2. 运行第二个脚本创建系统服务: bash scripts/setup_service.sh"
echo ""

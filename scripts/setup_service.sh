#!/bin/bash
# KPodcast 系统服务配置脚本
# 使用方法: bash setup_service.sh

set -e

echo "=========================================="
echo "  配置 KPodcast 系统服务"
echo "=========================================="

# 1. 创建 systemd 服务文件
echo "[1/3] 创建系统服务..."
sudo tee /etc/systemd/system/kpodcast.service > /dev/null << 'EOF'
[Unit]
Description=KPodcast Backend API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/kpodcast/backend
Environment="PATH=/opt/kpodcast/backend/venv/bin"
ExecStart=/opt/kpodcast/backend/venv/bin/uvicorn api_main:app --host 127.0.0.1 --port 8001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 2. 配置 Nginx 反向代理
echo "[2/3] 配置 Nginx..."
sudo tee /etc/nginx/sites-available/kpodcast > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;  # 后续可改为你的域名

    # KPodcast API
    location /api/ {
        proxy_pass http://127.0.0.1:8001/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        client_max_body_size 50M;
    }

    # 健康检查
    location /health {
        proxy_pass http://127.0.0.1:8001/;
        proxy_http_version 1.1;
    }
}
EOF

# 启用站点配置
sudo ln -sf /etc/nginx/sites-available/kpodcast /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# 3. 启动服务
echo "[3/3] 启动服务..."
sudo systemctl daemon-reload
sudo systemctl enable kpodcast
sudo systemctl start kpodcast

echo ""
echo "=========================================="
echo "  部署完成！"
echo "=========================================="
echo ""
echo "服务状态: sudo systemctl status kpodcast"
echo "查看日志: sudo journalctl -u kpodcast -f"
echo "重启服务: sudo systemctl restart kpodcast"
echo ""
echo "API 地址: http://82.157.17.187/api/"
echo "测试: curl http://82.157.17.187/api/"
echo ""

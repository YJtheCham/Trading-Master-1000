#!/bin/bash
# StockPredict Bot Server 一键部署脚本
# 在 VPS (38.55.107.224) 上执行

set -e

echo "=== StockPredict Bot 部署 ==="

# 1. 安装 Docker
if ! command -v docker &>/dev/null; then
    echo "安装 Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl start docker
    systemctl enable docker
fi

# 2. 创建项目目录
mkdir -p /opt/stock-predict
cd /opt/stock-predict

# 3. 从本机同步代码（需要本机先 git push）
echo "拉取最新代码..."
if [ -d ".git" ]; then
    git pull
else
    git clone https://github.com/YJtheCham/Trading-Master-1000.git .
fi

# 4. 写入飞书配置（需要手动填写）
if [ ! -f "data/config.json" ]; then
    cat > data/config.json << 'CONFIGEOF'
{
  "feishu_app_id": "",
  "feishu_app_secret": "",
  "feishu_verify_token": "",
  "feishu_encrypt_key": "",
  "vps_api_key": "sk-UpK7XsubDHCKP9EcSnnRyPkHX2ExZsXrd9PxilJ9Xo7CR0H2",
  "pushplus_token": ""
}
CONFIGEOF
    echo "⚠️  需要编辑 data/config.json 填入飞书配置！"
fi

# 5. 构建并启动 Docker 容器
echo "构建 Docker 容器..."
docker build -f deploy/Dockerfile -t stock-bot .

echo "启动 Bot Server..."
docker rm -f stock-bot 2>/dev/null || true
docker run -d \
    --name stock-bot \
    --restart unless-stopped \
    -p 8080:8080 \
    -v /opt/stock-predict/data:/app/data \
    stock-bot

echo ""
echo "=== 部署完成 ==="
echo "Bot Server 运行在 http://38.55.107.224:8080"
echo ""
echo "飞书回调 URL: http://38.55.107.224:8080/feishu/event"
echo "PushPlus回调 URL: http://38.55.107.224:8080/pushplus/callback"
echo ""
echo "验证:"
curl -s http://localhost:8080/health || echo "等待启动..."
echo ""
echo "查看日志: docker logs -f stock-bot"
echo "停止: docker stop stock-bot"
echo "重启: docker restart stock-bot"

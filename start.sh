#!/bin/bash
# StockPredict 一键启动
# 用法: bash start.sh

cd "$(dirname "$0")"

# 初始化缓存目录
mkdir -p data/cache

# 首次运行: 尝试拉取股票数据库
if [ ! -f data/stock_db_cache.json ] || [ "$1" = "--update-db" ]; then
    echo "📡 更新股票数据库..."
    python3 tools/update_stock_db.py 2>/dev/null || echo "   ⚠️ 离线模式 (跳过网络更新)"
fi

# 启动
echo "🚀 启动 StockPredict..."
streamlit run webapp/app.py --server.port "${PORT:-8501}" --server.address 0.0.0.0

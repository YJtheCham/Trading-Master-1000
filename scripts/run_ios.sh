#!/bin/bash
# StockPredict iOS 移动端启动脚本
# 启动: bash scripts/run_ios.sh
cd "$(dirname "$0")/.." && streamlit run webapp/ios_app.py --server.port 8502

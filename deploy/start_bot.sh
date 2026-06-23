#!/bin/bash
exec python3 -c "from src.bot.server import run_server; run_server(port=8080, host='0.0.0.0')"

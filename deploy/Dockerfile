FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY webapp/ webapp/
COPY pyproject.toml .
COPY deploy/start_bot.sh .

RUN chmod +x start_bot.sh

EXPOSE 8080

CMD ["./start_bot.sh"]

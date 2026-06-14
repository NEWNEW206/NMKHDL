FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ ./data/
COPY src/ ./src/
COPY main.py .

ENV PYTHONIOENCODING=utf-8

CMD ["python", "main.py"]

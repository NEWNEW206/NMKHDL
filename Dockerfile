FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy requirement và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn và dữ liệu vào Docker
COPY data/ ./data/
COPY src/ ./src/
COPY main.py .

# Tạo sẵn thư mục outputs để mount volume không bị lỗi
RUN mkdir -p outputs

ENV PYTHONIOENCODING=utf-8

# Chạy chương trình chính
CMD ["python", "main.py"]

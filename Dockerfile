FROM python:3.11-slim

WORKDIR /code

# 系统依赖（matplotlib 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码 + 模型
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]

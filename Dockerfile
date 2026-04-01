FROM python:3.10

WORKDIR /app

# 安装系统依赖，支持大部分 Python 包编译
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 升级 pip，并安装依赖
RUN pip install --upgrade pip wheel \
    && pip install --no-cache-dir -r requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
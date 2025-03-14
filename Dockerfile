FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8501

# 启动应用
CMD ["streamlit", "run", "app.py"]

# 使用官方 Python 运行环境作为基础镜像。
FROM python:3.10-slim

# 安装系统依赖（包含 Git）；代理通过 Git 导出增量补丁。
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 容器中的 DGM 工作目录；自修改时代理编辑此目录。
WORKDIR /dgm

# 将构建上下文复制到容器；.dockerignore 排除了评估数据与外层调度等文件。
COPY . .

# 安装 Python 依赖；父代补丁若更新 requirements.txt，运行时还会重新安装。
RUN pip install --no-cache-dir -r requirements.txt

# 保持容器运行，等待宿主机通过 exec_run 启动自修改任务。
CMD ["tail", "-f", "/dev/null"]

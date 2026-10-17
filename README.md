# YOLO26n 装甲板推理作业

## 1. 功能说明
读取视频，调用模型推理，画出检测框、类别、置信度以及4个关键点，并保存视频。

## 2. 环境依赖
Python 3.x, 使用 CPU 运行。

## 3. 运行方法
1. 将 `best.pt` 模型放在根目录。
2. 安装依赖：`pip install -r requirements.txt`
3. 运行：`python infer.py --source test.mp4`

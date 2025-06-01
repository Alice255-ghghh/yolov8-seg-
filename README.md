# YOLOv8n-seg：手势分割识别项目

🎯 **项目简介**  
本项目基于 Ultralytics 的轻量级语义分割模型 **YOLOv8n-seg**，利用公开图像数据集进行训练，最终用于图像中的**手势检测与分割**任务。模型具有速度快、部署方便、精度高等特点，可直接在本地进行图片或视频推理。

---

## 🌟 项目亮点

- ✅ 基于 YOLOv8 最小模型 yolov8n-seg，速度与精度兼顾  
- ✅ 支持自定义数据集，通过 YAML 配置实现灵活扩展  
- ✅ 推理脚本可处理单张图片或完整视频流  
- ✅ 支持 GPU 加速训练与推理（已在 RTX 4060 上测试）  
- ✅ 训练曲线、mAP、loss 等可视化一应俱全  

---

## 📂 项目结构

YOLOv8n-seg/
├── datasets/ # 数据集
│ ├── train/
│ ├── val/
│ └── data.yaml # YOLO 数据集配置文件
├── runs/ # 训练结果（自动生成）
│ └── segment/
├── yolov8n-seg.pt # 训练前模型权重
├── train.py # 训练脚本
├── 推理.py # 图片推理脚本
├── README.md # 项目说明文件（你现在看到的）
└── requirements.txt # Python 依赖项（可选）

yaml
复制
编辑

---

## 🚀 快速开始

### 🔧 环境配置

建议使用虚拟环境，Python 版本推荐 3.10：

```bash
pip install ultralytics opencv-python
🏋️‍♀️ 开始训练
bash
复制
编辑
python train.py
你可以在 train.py 中自定义参数，如：

python
复制
编辑
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0  # 使用GPU
)
🔍 图片推理
bash
复制
编辑
python 推理.py
确保你修改了推理脚本中的模型权重路径与图片路径：

python
复制
编辑
model = YOLO("runs/segment/train16/weights/best.pt")
results = model.predict(source="your-image.jpg")
📈 查看训练效果
训练结束后，会在 runs/segment/train*/ 中生成：

results.png：训练曲线图

best.pt：最佳模型权重

推理预测图像：带分割轮廓的样本预测图

metrics/：验证指标记录

📄 依赖项（可选）
生成 requirements.txt：

bash
复制
编辑
pip freeze > requirements.txt
或者你只需安装核心依赖：

bash
复制
编辑
pip install ultralytics opencv-python
🙏 致谢
Ultralytics 提供了强大的 YOLOv8 框架

Roboflow 提供了部分图像预处理功能（如使用）

📫 联系方式
作者：Alice255-ghghh
邮箱：hinahida255@gmail.com
如有问题请提交 Issue

💡 项目仍在开发中，欢迎关注、收藏和讨论！Star 🌟 一下支持一下吧！

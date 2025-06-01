from ultralytics import YOLO

# 加载模型
model = YOLO("runs/segment/train16/weights/best.pt")  # 换成你的模型路径

# 推理一张图片
# results = model.predict(source="C:\\Users\\zyb13\\Desktop\\YOLO检测\\datasets\\valid\\images\\A0_jpg.rf.133070191f474ea35bb7f23c8b86a2db.jpg", show=True, save=True)  # 换成你的图片路径
  
# 视频推理
# model.predict(source="test.mp4", show=True, save=True)

# 摄像头推理
model.predict(source=0, show=True)
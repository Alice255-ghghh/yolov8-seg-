import cv2
import numpy as np
from ultralytics import YOLO
import math

print("\033[106m" + "正在检测!!!!!!" + "\033[0m")


# 可识别的种类
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
# 颜色

classColor = [
    (193, 182, 255), (214, 112, 218), (180, 105, 255), (237, 149, 100),
    (222, 188, 176), (255, 191, 0), (255, 255, 0), (170, 255, 127),
    (127, 255, 0), (0, 255, 0), (0, 255, 127), (210, 250, 250),
    (181, 228, 255), (173, 222, 255), (0, 140, 255), (112, 164, 244),
    (122, 160, 255), (80, 127, 255), (0, 69, 255), (71, 99, 255),
    (130, 0, 75), (226, 46, 138), (219, 112, 147), (238, 104, 123),
    (205, 92, 106), (139, 61, 72), (255, 0, 255), (255, 0, 255),
    (238, 130, 238), (255, 0, 255), (180, 206, 70), (235, 245, 135),
    (230, 224, 176), (160, 209, 95), (255, 255, 240), (255, 255, 225),
    (238, 238, 175), (231, 242, 212), (209, 206, 0), (139, 139, 0),
    (128, 128, 0), (204, 209, 77), (162, 178, 32), (154, 250, 0),
    (113, 179, 60), (87, 139, 46), (240, 255, 240), (144, 238, 144),
    (122, 167, 233), (225, 228, 255), (250, 250, 255), (128, 0, 0),
    (92, 92, 205), (188, 134, 214), (222, 181, 255), (173, 222, 255),
    (0, 140, 255), (112, 164, 244), (122, 160, 255), (80, 127, 255),
    (0, 69, 255), (71, 99, 255), (130, 0, 75), (226, 46, 138),
    (219, 112, 147), (238, 104, 123), (205, 92, 106), (139, 61, 72),
    (255, 0, 255), (255, 0, 255), (238, 130, 238), (255, 0, 255),
    (87, 139, 46), (240, 255, 240), (144, 238, 144), (122, 167, 233),
    (225, 228, 255), (237, 149, 100), (222, 188, 176), (255, 191, 0),
    (255, 255, 0), (170, 255, 127), (250, 250, 234)
]

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("C:\\Users\\zyb13\\Desktop\\YOLO检测\\runs\\segment\\train16\\weights\\best.pt")

# 循环读取
while True:
    ret, img= cap.read()

    results = model(img,stream=True) # 进行实时检测
    '''印出結果

    '''
    for r in results: # 使用循环
        print("\033[96m" + "本次檢測到 "  + "\033[0m",len(r.boxes.cls) ,"\033[96m" + " 個目標" + "\033[0m")
        # 循环这个boxes

        for i in range(len(r.boxes)):
            mask_np = r.masks.data[i].cpu().numpy()# 将数据转换成numpy数组
            cls = int(r.boxes[i].cls[0])# 获取当前目标的类别索引
            color = classColor[cls]# 根据不同的索引使用不同的颜色来区分对象
            alpha = 0.6  # 透明度

            # 创建一个与原图大小相同的透明overlay图层
            overlay = np.zeros_like(img)

            # 分别对每个通道应用掩码和颜色
            for c in range(3):
                overlay[:, :, c][mask_np == 1] = (color[c] * alpha + img[:, :, c][mask_np == 1] * (1 - alpha)).astype(np.uint8)

            # 将overlay中的掩码区域合并到原图的相应位置
            img[mask_np == 1] = overlay[mask_np == 1]

            # 绘制边界框和类别信息
            x1, y1, x2, y2 = [int(x) for x in r.boxes[i].xyxy[0]]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{classNames[cls]} {r.boxes[i].conf[0]:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

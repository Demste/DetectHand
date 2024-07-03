import cv2
import numpy as np
from pathlib import Path
from utils import draw_bboxes
from deepsort import DeepSort

# YOLOv4 ile nesne algılama için gerekli kütüphaneleri yükleyin
weights_path = 'yolov4-tiny.weights'
cfg_path = 'yolov4-tiny.cfg'
names_path = 'coco.names'

net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

with open(names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# DeepSORT modelini başlatın
deepsort = DeepSort()

# Video kaynağını başlatın
video_path = 'video.mp4'  # Video dosyasının veya kamera kaynağının yolunu belirtin
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv4 ile nesne tespiti
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Sadece 'person' sınıfını al
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # DeepSORT ile nesne takibi
    if boxes:
        outputs = deepsort.update(np.array(boxes), np.array(confidences), np.array(class_ids), frame)

        # Takip sonuçlarını çerçeveye çizme
        for output in outputs:
            bbox, identity = output[:4], output[4]
            draw_bboxes(frame, bbox, identity)

    # Sonuçları gösterme
    cv2.imshow('DeepSORT Object Tracking', frame)

    # Çıkış için 'q' tuşuna basılmasını bekleyin
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()

import cv2

# YOLO yapılandırma ve ağırlık dosyalarının yolları
config_path = 'yolov4-tiny.cfg'
weights_path = 'yolov4-tiny.weights'
video_path="2932300-uhd_4096_2160_24fps.mp4"
# YOLO modelini yükle
net = cv2.dnn.readNet(weights_path, config_path)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), scale=1/255)

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)

def click_button(event,x,y,flags,params):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(x,y)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",click_button)
#classes = ["person", "car", "bicycle", "motorbike", "bus", "truck"]
#colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]


while True:
    ret, frame = cap.read()
    #if not ret:
        #break

    # Model kullanarak tespit yap
    (class_ids, scores, bboxes) = model.detect(frame)
    if len(class_ids)>0:
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox
            cv2.putText(frame,str(class_id),(x,y-10),cv2.FONT_HERSHEY_PLAIN,2,(200,0,50),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(200,0,50),3)

        print("clas ids",class_ids)
        print("scores",scores)    

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(10)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

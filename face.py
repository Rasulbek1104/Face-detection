from ultralytics import YOLO
import cv2


model = YOLO("yolov8n-face.pt") 


cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    
    results = model(frame)  
   
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  
        conf = detection.conf[0] 
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    cv2.imshow("YOLOv8 Face Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

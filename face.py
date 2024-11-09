from ultralytics import YOLO
import cv2

# Load your custom YOLO model
model = YOLO("yolov8n-face.pt")  # Ensure "best.pt" is in the same directory or provide the full path

# Start video capture (0 is usually the default webcam)
cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to capture frames continuously
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLOv8 detection on the current frame
    results = model(frame)  # Detect faces in the frame

    # Loop over detections and draw bounding boxes
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
        conf = detection.conf[0]  # Confidence score

        # Draw bounding box and confidence score
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("YOLOv8 Face Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

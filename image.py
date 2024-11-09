from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# YOLO modelini yuklash
model = YOLO("yolov8n-face.pt")

# Rasm yo'lini to'g'ri ko'rsatish (masalan, `mktb.jpg`)
image_path = "mktb.jpg"

# Modelni rasmda ishlatish va natijalarni olish
results = model(image_path)

# OpenCV yordamida rasmni yuklash
image = cv2.imread(image_path)

# Aniqlangan yuzlar va ularning joylashuvlarini rasmga chizish
for box in results[0].boxes:
    # Aniqlangan yuzning to'rtburchak koordinatalari
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Yuz chegaralarini yashil rangda chizish
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Matplotlib yordamida rasmni ko'rsatish
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR rangdan RGB rangga o'zgartirish
plt.imshow(image)
plt.axis("off")  # Koordinat o'qlarini o'chirish
plt.show()

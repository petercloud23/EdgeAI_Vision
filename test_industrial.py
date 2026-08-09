import cv2
import numpy as np

img = np.zeros((600, 800, 3), dtype=np.uint8)
for y in range(600):
    img[y, :] = (int(30 + y * 0.05), int(40 + y * 0.05), int(50 + y * 0.05))

cv2.rectangle(img, (50, 50), (150, 550), (70, 80, 90), -1)
cv2.rectangle(img, (650, 50), (750, 550), (70, 80, 90), -1)
cv2.line(img, (150, 100), (650, 100), (100, 110, 120), 8)
cv2.rectangle(img, (280, 40), (520, 90), (0, 200, 255), -1)
cv2.putText(img, "ZONA DE EPI OBRIGATORIO", (290, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

cv2.rectangle(img, (350, 200), (450, 520), (180, 100, 50), -1)
cv2.circle(img, (400, 160), 35, (200, 170, 150), -1)

cv2.imwrite("test_industrial.jpg", img)
print("test_industrial.jpg criado com sucesso!")

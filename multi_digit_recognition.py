import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model("digit_model.h5")

# Read multi-digit image
img = cv2.imread("123.png", 0)
img = cv2.resize(img, (400,200))
img = cv2.GaussianBlur(img, (5,5), 0)
_, img = cv2.threshold(img, 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morphological operation to connect broken digits
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(img,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

digits = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if h > 20:
        digit = img[y:y+h, x:x+w]

        # Padding to center digit
        pad = 20
        digit = cv2.copyMakeBorder(
            digit, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=0
        )

        digit = cv2.resize(digit, (28,28))
        digit = digit / 255.0
        digit = digit.reshape(1,28,28,1)

        pred = model.predict(digit)
        digit_class = np.argmax(pred)
        confidence = np.max(pred) * 100

        digits.append((x, digit_class, confidence))

# Sort digits left-to-right
digits = sorted(digits, key=lambda x: x[0])

# Build final number
final_number = "".join(str(d[1]) for d in digits)

print("Predicted Number:", final_number)
print("\nDigit-wise Confidence:")
for d in digits:
    print(f"Digit {d[1]} → {d[2]:.2f}%")

plt.imshow(img, cmap="gray")
plt.title("Processed Image")
plt.axis("off")

import ipywidgets as widgets
from IPython.display import display
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("digit_model.h5")

upload = widgets.FileUpload(accept='.png,.jpg,.jpeg', multiple=False)
button = widgets.Button(description="Recognize Digits")
output = widgets.Output()

def recognize(b):
    with output:
        output.clear_output()
        for name, file in upload.value.items():
            img = cv2.imdecode(np.frombuffer(file['content'], np.uint8), 0)
            img = cv2.resize(img,(400,200))
            img = cv2.GaussianBlur(img,(5,5),0)
            _, img = cv2.threshold(img,0,255,
                                   cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            contours,_ = cv2.findContours(img,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

            digits=[]
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                if h>20:
                    digit = img[y:y+h, x:x+w]
                    pad=20
                    digit = cv2.copyMakeBorder(digit,pad,pad,pad,pad,
                                               cv2.BORDER_CONSTANT,value=0)
                    digit = cv2.resize(digit,(28,28))
                    digit = digit/255.0
                    digit = digit.reshape(1,28,28,1)

                    pred = model.predict(digit)
                    digits.append((x, np.argmax(pred)))

            digits = sorted(digits, key=lambda x:x[0])
            number = "".join(str(d[1]) for d in digits)
            print("Recognized Number:", number)

button.on_click(recognize)

display(upload, button, output)

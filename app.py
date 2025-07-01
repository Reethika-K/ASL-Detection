import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

model=load_model("./Model/asl_model.keras")

labels = [chr(i) for i in range(ord('A'),ord('Z')+1)]

cap=cv.VideoCapture(0)

while True:
    ret,frame=cap.read()
    frame=cv.flip(frame,1)
    if not ret:
        break

    roi= frame[100:300,100:300]
    cv.rectangle(frame,(100,100),(300,300),(255,0,0),2)
    roi_gray=cv.cvtColor(roi,cv.COLOR_BGR2GRAY)
    roi_resized=cv.resize(roi_gray,(28,28))
    roi_normalized=roi_resized.astype('float32')/255.0
    roi_reshaped=roi_normalized.reshape(1,28,28,1)

    pred=model.predict(roi_reshaped,verbose=0)
    label=labels[np.argmax(pred)]

    cv.putText(frame,f"Prediction:{label}",(10,60),cv.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),2)
    cv.imshow("ASL Detection",frame)

    if cv.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv.destroyAllWindows()
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2


def prediction(model,img,im_size = 128):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (im_size, im_size))
    res = img_to_array(im)
    res = preprocess_input(res)
    res = np.expand_dims(res, axis=0)
    prediction = model.predict(res)[0]
    index = np.argmax(prediction)
    return index


model = load_model('/home/pavithran/StonePaperScissor/weights/hand_model.model')
print('Model Loaded')
cam=cv2.VideoCapture(0)
count = 0
while True:
    hasframe,frame_c = cam.read()
    frame=cv2.cvtColor(frame_c,cv2.COLOR_BGR2GRAY)
    _,th=cv2.threshold(frame,140,225,cv2.THRESH_BINARY)
    crp = th[10:300,10:300]
    crp_rgb = cv2.cvtColor(crp,cv2.COLOR_GRAY2RGB)
    res = prediction(model,crp_rgb,im_size=128)
    print(res)
    cv2.rectangle(frame_c,(10,10),(300,300),(0,255,0),1)
    if res == 0:
        cv2.putText(frame_c,'No Hand',(10,320),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    elif res == 1:
        cv2.putText(frame_c, 'Paper',(10,320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    elif res == 2:
        cv2.putText(frame_c, 'Scissor', (10,320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    elif res == 3:
        cv2.putText(frame_c, 'Stone', (10,320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv2.imshow('input',frame_c)
    # cv2.imshow('image',th)
    k = cv2.waitKey(33)
    if k == 13:
        break
    elif k == 32:
        pass
cv2.destroyAllWindows()
cam.release()
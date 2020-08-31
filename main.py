import cv2
import time
import numpy as np
from random import randrange
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

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

cam = cv2.VideoCapture(0)
end_time = time.time() + 3
paper = cv2.imread('/home/pavithran/StonePaperScissor/3.jpg')
scissor = cv2.imread('/home/pavithran/StonePaperScissor/26.jpg')
stone = cv2.imread('/home/pavithran/StonePaperScissor/8.jpg')
c_score = 0
y_score = 0
flag = 0
flag1 = 0
bg = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(bg,'Stone Paper Scissor Game ', (23,180), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), thickness=2)
cv2.putText(bg,'Press Any key to start the game ', (155, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)
cv2.imshow('image',bg)
cv2.waitKey(0)
while True:
    hasframe,frame = cam.read()
    cv2.rectangle(frame, (10, 10), (300, 300), (0, 255, 0), 1)
    frame = frame[:,:320]
    size = bg.shape
    bg[:100, :] = 0
    bg[:480,:320] = frame
    cv2.putText(bg,'COMPUTER '+str(c_score), (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)
    cv2.putText(bg, 'YOU ' + str(y_score), (330, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)
    current_time = time.time()
    # print('hai')
    if current_time < end_time:
        sec = end_time - current_time
        cv2.putText(bg,str(int(sec+1)),(500,80),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),thickness=5)
        im_no = randrange(3)
        flag = 1
    elif flag == 1 and current_time > end_time:
        if im_no == 0:
            im = paper
        elif im_no == 1:
            im = scissor
        elif im_no == 2:
            im = stone
        bg[100:390,340:630] = im
        # print(im_no)
        crp = bg[10:300, 10:300]
        crp = cv2.cvtColor(crp, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(crp, 200, 225, cv2.THRESH_BINARY)
        crp_rgb = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
        res = prediction(model, crp_rgb, im_size=128)
        cv2.putText(bg, 'Press Space to continue', (325, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                    thickness=2)
        cv2.putText(bg, 'Press Enter to Stop', (325, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                    thickness=2)
        if (res == 1 and im_no + 1 == 3) or (res == 2 and im_no + 1 == 1) or (res == 3 and im_no + 1 == 2):
            y_score = y_score + 1
            cv2.putText(bg, 'YOU WIN', (370, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                        thickness=2)
        elif (res == 1 and im_no + 1 ==1) or (res == 2 and im_no + 1 == 2) or (res == 3 and im_no + 1 == 3):
            cv2.putText(bg, 'TIE', (370, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                        thickness=2)
        elif res == 0:
            pass
        elif (res == 3 and im_no + 1 == 1) or (res == 1 and im_no + 1 == 2) or (res == 2 and im_no + 1 == 3):
            c_score = c_score + 1
            cv2.putText(bg, 'COMPUTER WIN', (370, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                        thickness=2)
        flag = 0

    cv2.imshow('image',bg)
    k = cv2.waitKey(33)
    if flag1 == 0:
        flag1 = 1
        k = 32
    if k == 13:
        break
    elif k == 32:
        bg = np.zeros((480, 640, 3), dtype=np.uint8)
        end_time = time.time() + 3
cv2.destroyAllWindows()
cam.release()
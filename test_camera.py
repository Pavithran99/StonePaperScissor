import cv2


cam=cv2.VideoCapture(0)
count = 0
while True:
    hasframe,frame_c = cam.read()
    # print(frame_c.shape)
    frame=cv2.cvtColor(frame_c,cv2.COLOR_BGR2GRAY)
    _,th=cv2.threshold(frame,200,225,cv2.THRESH_BINARY)
    crp = th[10:300,10:300]
    cv2.rectangle(frame_c,(10,10),(300,300),(0,255,0),1)
    cv2.imshow('input',crp)
    cv2.imshow('image',frame_c)
    k = cv2.waitKey(33)
    if k == 13:
        break
    elif k == 32:
        pass
cv2.destroyAllWindows()
cam.release()
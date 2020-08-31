import cv2


cam=cv2.VideoCapture(0)
des_path = '/home/pavithran/StonePaperScissor/dataset/train/nohand/'
count = 0
while True:
    hasframe,frame = cam.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,th=cv2.threshold(frame,140,225,cv2.THRESH_BINARY)
    crp = th[10:300,10:300]
    cv2.rectangle(frame,(10,10),(300,300),(0,255,0),1)
    cv2.imshow('input',crp)
    # cv2.imshow('image',th)
    k = cv2.waitKey(33)
    if k == 13:
        break
    elif k == 32:
        cv2.imwrite(des_path + str(count) + '.jpg', crp)
        count = count + 1
        print('Saved {}'.format(count))
cv2.destroyAllWindows()
cam.release()
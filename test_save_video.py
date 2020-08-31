import cv2





cam = cv2.VideoCapture('/home/pavithran/StonePaperScissor/18_08_2020-16_46_14.avi')
_,frame = cam.read()
frame_width = int(cam.get(3))
frame_height = int(cam.get(4))
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
while True:
	hasframe,frame = cam.read()
	if hasframe == False:
		break	
	out.write(frame)
out.release()
cam.release()






sudo ssh -i ml_instance_01.pem ubuntu@15.207.111.99



    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))


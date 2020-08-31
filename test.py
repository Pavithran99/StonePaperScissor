from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2

def prediction(img,im_size = 128):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (im_size, im_size))
    res = img_to_array(im)
    res = preprocess_input(res)
    res = np.expand_dims(res, axis=0)
    prediction = model.predict(res)[0]
    return prediction


model = load_model('/home/pavithran/StonePaperScissor/weights/hand_model.model')
path = '/home/pavithran/StonePaperScissor/sps/train/paper/6.jpg'
image = cv2.imread(path)
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
output = prediction(image)
index =  np.argmax(output)
print(output)
print('output_index:', index)
if index == 1:
    print('Paper')
elif index == 2:
    print('Scissor')
elif index == 3:
    print('Stone')
elif index == 4:
    print('No hand')
else:
    print('Error')
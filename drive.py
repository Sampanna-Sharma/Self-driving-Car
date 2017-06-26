from keras.models import load_model
import grabscreen
from gettrainingdata import imageprocessing
import numpy as np
from sendkey import PressKey
from sendkey import ReleaseKey
import time
import cv2
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 150, 300, 1
model = load_model("model-001.h5")
W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

for j in range(5):
    print(j + 1)
    time.sleep(1)
while True:
    image1 = grabscreen.grab_screen([100, 350, 700, 650])
    image1 = imageprocessing(image1)
    image1 = cv2.resize(image1, (300, 150))
    #new_img = imageprocessing(new_img)
    new1 = image1
    image1 = np.array(image1, dtype=np.uint8)
    image1 = image1.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    image1 = image1.astype('float32')
    image1 /= 255

    output = (model.predict(image1,batch_size=1))
    #output = [x for x in output[0]]
    print(output)
    #output = [round(x) for x in output[0]]

    cv2.imshow("window1", new1)
    if cv2.waitKey(255) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    '''if output[0][0] > output[0][1] and output[0][0] > output[0][2]:
        PressKey(A)
        ReleaseKey(W)
        ReleaseKey(D)
        print('A')
    elif output[0][2] > output[0][1] and output[0][2] > output[0][0]:
        PressKey(D)
        ReleaseKey(A)
        ReleaseKey(W)
        print('D')
    else:
        PressKey(W)
        ReleaseKey(A)
        ReleaseKey(D)
        print('W')'''


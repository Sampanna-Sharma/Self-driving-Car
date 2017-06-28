import grabscreen
import csv
import os
import time
import cv2
from getkeys import key_check
import numpy as np
from PIL import Image

def imageprocessing(image):
    processed_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #processed_image = cv2.Canny(processed_image,threshold1=200,threshold2=200)
    return processed_image

def writecsv(o1):
    with open('log.csv','a',newline='') as fp:
        writer = csv.writer(fp,delimiter=',')
        writer.writerow(o1)

def getkey():
    key = key_check()
    output = [0,0,0]
    #AWD

    if 'A' in key:
        output[0] = 1
    elif 'D' in key:
        output[2] = 1
    else:
        output[1] = 1

    return output




if __name__ == "__main__":
    try:
        os.mkdir("Images")
    except:
        pass
    i = 1
    #dataarray = []
    for j in range(5):
        print(j + 1)
        time.sleep(1)
    while True:
        image1 = grabscreen.grab_screen([0, 350, 800, 650])
        image1 = imageprocessing(image1)
        image1 = cv2.resize(image1, (400, 150))
        image_name = str(i)+ '.png'
        full_name_image = os.path.join('Images', image_name)
        keypressed = getkey()
        writecsv([full_name_image,keypressed])
        cv2.imwrite(full_name_image,image1)
        print(i)
        i += 1


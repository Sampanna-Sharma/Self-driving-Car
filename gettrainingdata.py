import grabscreen
import csv
import os
import time
import cv2
from msvcrt import getch
from msvcrt import kbhit

def imageprocessing(image):
    processed_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    processed_image = cv2.Canny(processed_image,threshold1=500,threshold2=500)
    return processed_image

def writecsv(o1):
    with open('log.csv','a',newline='') as fp:
        writer = csv.writer(fp,delimiter=',')
        writer.writerow(o1)

def getkey():
    if kbhit():
        c = ord(getch())
    else:
        c = ord('w')
    return c


try:
    os.mkdir("Images")
except:
    pass

i = 1
while True:
    t1 = time.time()
    image = grabscreen.grab_screen([100, 100, 600, 600])
    image = imageprocessing(image)
    name = str(i) + '.png'
    full_name = os.path.join('Images',name)
    keypressed = getkey()
    writecsv([full_name,keypressed])
    cv2.imwrite(full_name,image)
    print(i)
    i += 1
    if cv2.waitKey(255) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

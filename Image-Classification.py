import cv2
import numpy as np
from pynput import keyboard

frameWidth = 320
frameHeight = 240
cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)

#img = cv2.imread('cup.jpg')
classNames= []
classFile = ['Ball','Bottle','Glasses','Smartphone']

weightsPath = 'simple_frozen_graph.pb'

net = cv2.dnn.readNetFromTensorflow(weightsPath)
### key event
def detect():
    print('Detecting')
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (224, 224), swapRB = True, crop = False )
    blob_image = blob.reshape(blob.shape[2] * blob.shape[1], blob.shape[3], 1) 
    net.setInput(blob)
    out = net.forward()
    print(out)
    print(classFile[np.argmax(out[0], axis=0)])

def on_activate_s():
    print('checking')

listener = keyboard.GlobalHotKeys({
        'c': detect,
        's': on_activate_s})

listener.start()


###
while True:
    success, img = cap.read()
    cv2.imshow("Camera", img)
    cv2.waitKey(1)
    



'''
success, img = cap.read()
blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (224, 224), swapRB = True, crop = False )
blob_image = blob.reshape(blob.shape[2] * blob.shape[1], blob.shape[3], 1) 
net.setInput(blob)
out = net.forward()
print(out)
print(classFile[np.argmax(out[0], axis=0)])
cv2.putText(img,classFile[np.argmax(out[0], axis=0)],(150,180),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
cv2.imshow("Camera", img)
cv2.waitKey(1)
'''


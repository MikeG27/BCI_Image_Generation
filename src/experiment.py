import time

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

X = x_train[0:1200]

X.shape

import cv2

cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

for i in range(len(X)):
    if i == 0 :
        time.sleep(5)
        
    cv2.imshow('frame',X[i])
    #cv2.resizeWindow('frame', 1200,1200)
    time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



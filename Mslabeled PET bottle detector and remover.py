from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import numpy as np
import picamera

lower_yellow = np.array([20,100,100])
upper_yellow = np.array([30,255,255])

lower_green = np.array([36,50,1])
upper_green = np.array([89,255,255])

lower_red = np.array([0,70,50])
upper_red = np.array([10,255,255])

lower_w = np.array([0,0,168])
upper_w = np.array([0,0,255])

lower_grey = np.array([0,0,71])
upper_grey = np.array([90,255,50])

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (340, 220)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(340, 220))
    
# allow the camera to warmup
time.sleep(0.1)
    
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    img = frame.array
    
    #converting BGR to HSV
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #create the Mask
    mask1 = cv2.inRange(imgHSV,lower_yellow,upper_yellow)
    mask2 = cv2.inRange(imgHSV,lower_green,upper_green)
    mask3 = cv2.inRange(imgHSV,lower_red,upper_red)
    mask4 = cv2.inRange(img1,lower_w,upper_w)
    mask5 = cv2.inRange(img1,lower_grey,upper_grey)
    
    mask = mask1 + mask2 + mask3 + mask4 + mask5

    results = cv2.bitwise_and(img, img,mask = mask)

    #applying median blur to make the image more clearer and to remove noise
    median = cv2.medianBlur(mask,9)
    blur = cv2.GaussianBlur(mask,(15,15),0)
    outputt = median + blur

    #contour drawing
    contours,h=cv2.findContours(outputt.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2:]
    #cv2.drawContours(img,conts,-1,(255,0,0),2)

    #rectangular contour drawing
    for i in range(len(contours)):
        x,y,w,h=cv2.boundingRect(contours[i])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
        print("bounding_box(x,y,w,h):",x,y,w,h)

    #detecting the mislabelled bottle
        if (w==102 and h<=174 and h>=172):
                print ("Correct Label")
        else:
                print ("Mislabelled bottle detected")
    
    cv2.imshow("mask",mask)
    cv2.imshow("cam",img)
    cv2.imshow("convolve_output",outputt)
    
    cv2.waitKey(10)

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    

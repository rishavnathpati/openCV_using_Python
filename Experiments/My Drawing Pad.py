import cv2
import matplotlib.pyplot as plt 
import numpy as np  

################# VARIABLES ####################################
# True when mouse DOWN, False when UP
drawing=False
ix=-1
iy=-1

#################### FUNCTION ##################################
def drawShapes(event,x,y,flags,param):
    global ix,iy,drawing
    
    # For circles, L-filled circle, R-line circle
    # if event == cv2.EVENT_LBUTTONDOWN:
    #     cv2.circle(img,(x,y),20,(0,100,100),-1)
    # if event== cv2.EVENT_RBUTTONDOWN:
    #     cv2.circle(img,(x,y),20,(100,10,100),2)

    if event == cv2.EVENT_RBUTTONDOWN:
        drawing=True
        ix,iy=x,y
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(ix,iy),1,(100,100,100),1)
    elif event==cv2.EVENT_RBUTTONUP:
        drawing=False
        cv2.circle(img,(ix,iy),1,(100,100,100),1)
    
    # For rectangles with mouse drag

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            pass
            #cv2.rectangle(img,(ix,iy),(x,y),(100,100,100),-1)
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.rectangle(img,(ix,iy),(x,y),(100,100,100),-1)

###################### SHOWING IMAGE #########################
img=np.zeros((512,512,3))
cv2.namedWindow(winname="My Drawing")
cv2.setMouseCallback("My Drawing",drawShapes)

while True:
    cv2.imshow("My Drawing",img)
    if(cv2.waitKey(20) & 0xFF == 27):
        break

cv2.destroyAllWindows()
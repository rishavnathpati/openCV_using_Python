import time
import cv2
import numpy as np

path = r'D:/Study/Python/openCV/College/res/'

img = cv2.imread(path+'five_shapes.png')
scale_percent = 50 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

start_time = time.time()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,50)
# print(lines,lines[1][0])
for i in range(lines.shape[0]):
# for rho,theta in lines[i]:
    rho,theta = lines[i][0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

print("--- %s seconds ---" % (time.time() - start_time))
print(img.shape)
print(lines.shape)

# cv2.imshow('houghlines3.jpg',img)
# cv2.waitKey(0)
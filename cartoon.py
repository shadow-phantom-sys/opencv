import cv2
from cv2 import adaptiveThreshold
import numpy as np

img=cv2.imread("satya2.jpeg")

def cartoonize(img,k):
    data=np.float32(img).reshape(-8,3)                              #converted to float value        #reshape image using numpy without changing any value of pixels
    print("shape of input data",img.shape)                                                      #print (no of pixels in x and y) and 
    print("shape o resize data",data.shape)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0)                          #image is segmented using kmeans clustering
    _,label,center=cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)                
    center=np.uint8(center)
    print(center)
    result=center[label.flatten()]
    result=result.reshape(img.shape)
    cv2.imshow("result",result)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                                                   #Converts the img to gray image from bgr to gray
   
    #Adaptive thresholding is the method where the threshold value is calculated for smaller regions and therefore, there will be different threshold values for different regions.  
    edges=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,9)  #Adaptive threshold function for finding edges 
    cv2.imshow("edges",edges)                                                                   #Line_value=9 Blurr_value=8
    
    
    blurred = cv2.medianBlur(result, 3)                                                         #to smoothen the edges
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)                                     # Combine the result and edges to get final cartoon effect
    cv2.imshow("output", cartoon)                                                               #show output
    cv2.imwrite("output3.jpeg",cartoon)                                                         #save output
cartoonize(img,8)

cv2.waitKey(0)
cv2.destroyAllWindows

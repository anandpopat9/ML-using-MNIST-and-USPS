#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:20:41 2016

@author: anandpopat
"""

import numpy as np
import cv2
import os



img_a = cv2.imread('/Users/anandpopat/desktop/ml/proj3/USPSdata/0001a.png',0)
resized_image_a = cv2.resize(img_a, (28, 28)) 
norm_image_a = cv2.normalize(resized_image_a, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
data_a=np.array(norm_image_a,dtype=float)
test_usps_x=data_a.flatten()


test_usps_y=np.array(0)


for x in range(0,10):
    

    


    for root, dirs, files in os.walk("/Users/anandpopat/desktop/ml/proj3/USPSdata/Numerals/"+str(x)+"/"):
        print files


    for file1 in files:
        img_a = cv2.imread('/Users/anandpopat/desktop/ml/proj3/USPSdata/Numerals/'+str(x)+'/'+str(file1),0)
        resized_image_a = cv2.resize(img_a, (28, 28)) 
        norm_image_a = cv2.normalize(resized_image_a, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        data_a=np.array(norm_image_a,dtype=float)
        flattened_a=data_a.flatten()
        test_usps_x=np.vstack((test_usps_x,flattened_a))
        test_usps_y1=np.array(x)
        test_usps_y=np.vstack((test_usps_y,test_usps_y1))
        
np.save('test_usps_x',test_usps_x)
np.save('test_usps_y',test_usps_y)



'''
count = 0
img_a = cv2.imread('/Users/anandpopat/desktop/ml/proj3/USPSdata/Numerals/0/0001a.png',0)
resized_image_a = cv2.resize(img_a, (28, 28)) 
norm_image_a = cv2.normalize(resized_image_a, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
data_a=np.array(norm_image_a,dtype=float)
flattened_a=data_a.flatten()
 
    

img_b = cv2.imread('/Users/anandpopat/desktop/ml/proj3/USPSdata/Numerals/0/0001b.png',0)
resized_image_b = cv2.resize(img_b, (28, 28)) 
norm_image_b = cv2.normalize(resized_image_b, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
data_b=np.array(norm_image_b,dtype=float)
flattened_b=data_b.flatten()



img_c = cv2.imread('/Users/anandpopat/desktop/ml/proj3/USPSdata/Numerals/0/0001c.png',0)
resized_image_c = cv2.resize(img_c, (28, 28)) 
norm_image_c = cv2.normalize(resized_image_c, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
data_c=np.array(norm_image_c,dtype=float)
flattened_c=data_c.flatten()

test_usps=np.vstack((flattened_a,flattened_b))
test_usps=np.vstack((test_usps,flattened_c))

test_usps1=test_usps

for x in range(2,10):
    img_a = cv2.imread('/Users/anandpopat/desktop/ml/proj3/USPSdata/Numerals/0/000'+str(x)+'a.png',0)
    resized_image_a = cv2.resize(img_a, (28, 28)) 
    norm_image_a = cv2.normalize(resized_image_a, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    data_a=np.array(norm_image_a,dtype=float)
    flattened_a=data_a.flatten()
 
    

    img_b = cv2.imread('/Users/anandpopat/desktop/ml/proj3/USPSdata/Numerals/0/000'+str(x)+'b.png',0)
    resized_image_b = cv2.resize(img_b, (28, 28)) 
    norm_image_b = cv2.normalize(resized_image_b, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    data_b=np.array(norm_image_b,dtype=float)
    flattened_b=data_b.flatten()



    img_c = cv2.imread('/Users/anandpopat/desktop/ml/proj3/USPSdata/Numerals/0/000'+str(x)+'c.png',0)
    resized_image_c = cv2.resize(img_c, (28, 28)) 
    norm_image_c = cv2.normalize(resized_image_c, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    data_c=np.array(norm_image_c,dtype=float)
    flattened_c=data_c.flatten()



    
    test_usps=np.vstack((flattened_a,flattened_b))
    test_usps=np.vstack((test_usps,flattened_c))

    test_usps1=np.vstack((test_usps1,test_usps))
    
for x in range(10,100):
    img_a = cv2.imread('/Users/anandpopat/desktop/ml/proj3/USPSdata/Numerals/0/00'+str(x)+'a.png',0)
    resized_image_a = cv2.resize(img_a, (28, 28)) 
    norm_image_a = cv2.normalize(resized_image_a, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    data_a=np.array(norm_image_a,dtype=float)
    flattened_a=data_a.flatten()
 
    

    img_b = cv2.imread('/Users/anandpopat/desktop/ml/proj3/USPSdata/Numerals/0/00'+str(x)+'b.png',0)
    resized_image_b = cv2.resize(img_b, (28, 28)) 
    norm_image_b = cv2.normalize(resized_image_b, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    data_b=np.array(norm_image_b,dtype=float)
    flattened_b=data_b.flatten()



    img_c = cv2.imread('/Users/anandpopat/desktop/ml/proj3/USPSdata/Numerals/0/00'+str(x)+'c.png',0)
    resized_image_c = cv2.resize(img_c, (28, 28)) 
    norm_image_c = cv2.normalize(resized_image_c, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    data_c=np.array(norm_image_c,dtype=float)
    flattened_c=data_c.flatten()



    
    test_usps=np.vstack((flattened_a,flattened_b))
    test_usps=np.vstack((test_usps,flattened_c))

    test_usps1=np.vstack((test_usps1,test_usps))
    count = count + 1
'''
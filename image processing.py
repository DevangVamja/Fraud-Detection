# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:17:59 2020

@author: Devang Patel
"""

import cv2
#import numpy as np
import sys
import matplotlib.pyplot as plt

img = cv2.imread('dev2.png',cv2.IMREAD_COLOR)
histr = cv2.calcHist([img],[2],None,[256],[0,256])
plt.plot(histr)
plt.xlim([0,256])
#plt.ylim([0,100])
plt.show()
#blur=cv2.blur(img,(7,7))
#cv2.imshow('blur image',blur)
#cv2.imshow('dev',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#sys.exit()
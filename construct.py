import cv2 as cv
import numpy as np

# n:n大小测地膨胀  f:标记图像 b：结构元素 g：模板图像
#测地膨胀
def D_g(f,b,g):  
    return np.min((cv.dilate(f,b),g),axis=0)

#膨胀重建
def R_g_D(f,b,g):
    img = f
    while True:
        new = D_g(img,b,g)
        if (new==img).all():
            return img
        img = new

# 重建开操作
def O_R(n,f,b,conn = cv.getStructuringElement(cv.MORPH_RECT,(3,3))):
    erosion=cv.erode(f,b,iterations=n)
    return R_g_D(erosion,conn,f)
    
 # 重建闭操作
def C_R(n,f,b,conn = cv.getStructuringElement(cv.MORPH_RECT,(3,3))):
    dilation=cv.dilate(f,b,iterations=n)
    return 255-R_g_D(255-dilation,conn,255-f)

def region_max(dist_transform,h):
    dist_transform_marker = dist_transform - h
    dist_transform_OC = R_g_D(dist_transform_marker,cv.getStructuringElement(cv.MORPH_RECT,(5,5)),dist_transform)
    dist_transform_max = dist_transform - dist_transform_OC
    return dist_transform_max

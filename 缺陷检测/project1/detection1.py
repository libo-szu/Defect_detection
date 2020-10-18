import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def detection(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_roi=img.copy()
    img_roi=cv2.GaussianBlur(img_roi,(7,7),1)
    img_roi[img_roi<120]=0
    img_roi[img_roi>=120]=255
    xs,ys=np.nonzero(img_roi)
    min_x=np.min(xs)
    min_y=np.min(ys)
    max_x=np.max(xs)
    max_y=np.max(ys)
    img_crop_roi=img[min_x:max_x,min_y:max_y]
    img_crop_roi=cv2.GaussianBlur(img_crop_roi,(5,5),1)

    img_crop_roi[img_crop_roi<100]=0
    img_crop_roi[img_crop_roi>=100]=255
    img_copy=np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    img_copy[min_x:max_x,min_y:max_y]=img_crop_roi
    cnts,_ = cv2.findContours(img_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for item in cnts:
        item=np.array(item)
        xmin_roi=min(item[:,0,1])
        xmax_roi=max(item[:,0,1])
        ymin_roi=min(item[:,0,0])
        ymax_roi=max(item[:,0,0])
        if((ymax_roi-ymin_roi)*(xmax_roi-xmin_roi)<50*50):

            img[img>0]=0
            img[xmin_roi:xmax_roi,ymin_roi:ymax_roi]=255-img_copy[xmin_roi:xmax_roi,ymin_roi:ymax_roi]
    return img


if __name__=="__main__":
    path="1.jpg"
    img=cv2.imread(path)
    roi=detection(img)
    img[roi>0]=[255,0,0]
    plt.imshow(img)
    plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt


def detection(img):
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img_gray=cv2.GaussianBlur(img_gray,(3,3),1)
    ret,thresh_img=cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierachy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    mask = np.zeros((img_gray.shape[0],img_gray.shape[1]))
    cv2.fillPoly(mask, [sorted_contours[0]], 255)
    thresh_img[mask==0] = 0
    # thresh_img=cv2.resize(thresh_img,(int(thresh_img.shape[1]),int(thresh_img.shape[0]*10)))
    # thresh_img[thresh_img>=128]=255
    # thresh_img[thresh_img<128]=0
    contours2, hierachy2 = cv2.findContours(thresh_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE )

    # draw_all_contours = cv2.drawContours(img, contours2, -1, (0, 0, 255), 2)
    roi_bboxs=[]
    for i,contour in enumerate(contours2):
        contour=np.array(contour)
        try:
            if(hierachy2[i][0][3]!=-1  and len(contour)>10):
                minx=np.min(contour[:,0,0])
                maxx=np.max(contour[:,0,0])
                miny=np.min(contour[:,0,1])
                maxy=np.max(contour[:,0,1])
                roi_bboxs.append([minx,miny,maxx,maxy])
        except:
            if(len(contour)>10 ):
                minx=np.min(contour[:,0,1])
                maxx=np.max(contour[:,0,1])
                miny=np.min(contour[:,0,0])
                maxy=np.max(contour[:,0,0])

                roi_bboxs.append([minx,miny,maxx,maxy])
    for bbox in roi_bboxs:
        xmin,ymin,xmax,ymax=bbox
        ret, thresh_roi = cv2.threshold(img_gray[xmin:xmax,ymin:ymax], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img[xmin:xmax,ymin:ymax,0][thresh_roi==0]=255

    return img

if __name__=="__main__":
    path="2.jpg"
    img=cv2.imread(path)
    new_img=detection(img)
    plt.imshow(new_img)
    plt.show()
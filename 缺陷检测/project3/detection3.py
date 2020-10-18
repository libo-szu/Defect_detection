import cv2
import numpy as np
import matplotlib.pyplot as plt

def detection(img):
    kernel = np.ones((3, 3), np.uint8)
    NpKernel = np.uint8(np.ones((5, 5)))
    # for i in range(5):
    #     NpKernel[2, i] = 1
    #     NpKernel[i, 2] = 1
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_blur=cv2.GaussianBlur(img_gray,(7,7),1)
    ret,thresh_img=cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts,hes=cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    sorted_cnts=sorted(cnts,key=lambda c:cv2.contourArea(c),reverse=True)
    rois=[]
    cnts_roi=[]
    xy_center=[]
    for cnt in sorted_cnts[:3]:
        mask=np.zeros_like(img_gray)
        cv2.fillPoly(mask,[cnt],255)
        mask1=cv2.resize(mask,(mask.shape[1]//4,mask.shape[0]//4))
        circles1 = cv2.HoughCircles(mask1, cv2.HOUGH_GRADIENT,1,1,param1=80,param2=30,minRadius=15,maxRadius=500)
        if(type(circles1)!=np.ndarray):
            rois.append(mask)
            cnts_roi.append(cnt)
            ymin=np.min(np.array(cnt)[:,0,1])
            ymax=np.max(np.array(cnt)[:,0,1])
            xy_center.append((ymin+ymax)//2)

    # print(xy_center)
    for i,roi in enumerate(rois):
        closed_img = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
        if(i==0 and (xy_center[0]<xy_center[1])):
            xmin = np.min(np.array(cnts_roi[i])[:, 0, 0])
            ymin_y=np.max(np.array(cnts_roi[i])[:, 0, 1][np.where(np.array(cnts_roi[i])[:, 0, 0]==xmin)])
            closed_img[ymin_y:,:]=0
        
        else:
            ymax = np.max(np.array(cnts_roi[i])[:, 0, 1])
            xmax_y=np.max(np.array(cnts_roi[i])[:, 0, 0][np.where(np.array(cnts_roi[i])[:, 0, 1]==ymax)])

            xmin= np.min(np.array(cnts_roi[i])[:, 0, 0])
            ymin= np.min(np.array(cnts_roi[i])[:, 0, 1])

            xmax= np.max(np.array(cnts_roi[i])[:, 0, 0])
            mask_=np.zeros_like(closed_img)
            cv2.fillPoly(mask_,np.array([[[xmin,ymin],[xmax_y,ymax],[xmax,ymin]]]),(255,255,255))
            closed_img[mask_==0]=0
        cnts3, hes3 = cv2.findContours(closed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        hull = cv2.convexHull(cnts3[0], returnPoints=False)
        defects  = cv2.convexityDefects(cnts3[0],hull)
        list_roi_out=[]
        list_roi_in=[]
        for j in range(defects.shape[0]):
            s, e, f, d = defects[j, 0]
            start = tuple(cnts3[0][s][0])
            end = tuple(cnts3[0][e][0])
            far = tuple(cnts3[0][f][0])
            list_roi_out.append(list(start))
            list_roi_out.append(list(end))
            list_roi_in.append(list(far))
        mask_1 = np.zeros_like(closed_img)
        cv2.fillPoly(mask_1, np.array([list_roi_out]), (255, 255, 255))


        mask_2 = np.zeros_like(closed_img)
        cv2.fillPoly(mask_2, np.array([list_roi_in]), (255, 255, 255))
        mask_1[mask_2==mask_1]=0
        mask_1[thresh_img==mask_1]=0
        mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, NpKernel, iterations=1)
        img[mask_1>0]=[255,0,0]
    return img

if  __name__=="__main__":
    path="3.jpg"
    img=cv2.imread(path)
    new_img=detection(img)
    plt.imshow(new_img)
    plt.show()

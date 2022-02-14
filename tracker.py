from mimetypes import init
import numpy as np
import cv2

class Tracker():
    """
    kernel_size: size of square kernel.
    """
    def __init__(self, init_img, iou_thres, kernel_size=5, seek_window=5, stride=1):
        self.kernel_size = kernel_size
        self.seek_window = seek_window
        self.stride = stride
        self.iou_thres = iou_thres
        self.last_img = init_img
        self.tracking = [self.track_item([266,659,116,60]),
                        self.track_item([640,52,22,20])]

        if (seek_window-1)%stride != 0:
            raise ValueError(
                'wrong stride : (seek_window-1)mod(stride) should be 0')

    class track_item():
        def __init__(self, bbox):
            self.bbox = bbox
            self.corners = [
                [bbox[0],bbox[1]],
                [bbox[0]+bbox[2],bbox[1]],
                [bbox[0],bbox[1]+bbox[3]],
                [bbox[0]+bbox[2],bbox[1]+bbox[3]]]

    def create_kernels(self):
        ROI_size = self.kernel_size+self.seek_window-1
        p = ROI_size//2
        nk = (self.seek_window-1)//self.stride + 1
        board = np.zeros((nk*nk,ROI_size*len(self.tracking),ROI_size*4))
        last_img_pad = np.pad(self.last_img,p,mode='edge')
        for i, t_item in enumerate(self.tracking):
            for j, c in enumerate(t_item.corners):
                board[0,i*ROI_size:i*ROI_size+self.kernel_size,
                    j*ROI_size:j*ROI_size+self.kernel_size] = \
                        last_img_pad[c[1]+p-self.kernel_size//2:c[1]+p+self.kernel_size//2+1,
                            c[0]+p-self.kernel_size//2:c[0]+p+self.kernel_size//2+1]
        for i in range(nk):
            for j in range(nk):
                board[i*nk+j,i*self.stride:board.shape[1]-ROI_size+i*self.stride+self.kernel_size,
                    j*self.stride:board.shape[2]-ROI_size+j*self.stride+self.kernel_size] = \
                        board[0,:-ROI_size+self.kernel_size,:-ROI_size+self.kernel_size]

        return board

    def create_crnt_img_mosaic(self, img):
        ROI_size = self.kernel_size+self.seek_window-1
        p = ROI_size//2
        img_pad = np.pad(img,p,mode='edge')
        board = np.zeros((ROI_size*len(self.tracking),ROI_size*4))
        for i, t_item in enumerate(self.tracking):
            for j, c in enumerate(t_item.corners):
                board[i*ROI_size:(i+1)*ROI_size,
                    j*ROI_size:(j+1)*ROI_size] = \
                        img_pad[c[1]+p-ROI_size//2:c[1]+p+ROI_size//2+1,
                            c[0]+p-ROI_size//2:c[0]+p+ROI_size//2+1]
            
        return board

    def getIoU(self, pred, label):
        area_label = label[2] * label[3]
        area_predict = pred[2] * pred[3]
        area_of_overlap = max(0,min(label[0]+label[2]*0.5, pred[0]+pred[2]*0.5)\
                        - max(label[0]-label[2]*0.5, pred[0]-pred[2]*0.5))\
                        * \
                        max(0,min(label[1]+label[3]*0.5, pred[1]+pred[3]*0.5) \
                        - max(label[1]-label[3]*0.5, pred[1]-pred[3]*0.5))
        area_of_union = area_label + area_predict - area_of_overlap
        return area_of_overlap / area_of_union


    def track(self, img, new_bboxes=[]):
        kernels = tr.create_kernels()
        mosaic = tr.create_crnt_img_mosaic(img)
        km = ((kernels-mosaic)**2)*(kernels != 0)
        km = np.transpose(km,(1,2,0))
        km_avg = cv2.resize(km,(4,len(self.tracking)))

        print(km_avg)
        # # remove those that are already being tracked
        # for t in self.tracking:
        #     for b in new_bboxes:
        #         if self.getIoU(t, b) > self.iou_thres:
        #             new_bboxes.remove(b)
        #             break
        # # append new bboxes
        # self.tracking += new_bboxes


if __name__ == '__main__':
    '266 659 116 60' '640 52 22 20'

    init_img = cv2.imread('0-101.jpg')
    img_gr = cv2.cvtColor(init_img,cv2.COLOR_BGR2GRAY)

    tr = Tracker(img_gr,0.5,5,5,2)
    tr.track(img_gr)
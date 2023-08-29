"""
Demo script.
"""

import os,sys
import argparse
import cv2
import torch
import glob
import numpy as np


from predictor import Predictor
from utils.file_io import write_pfm

TEST_FRMS = "./output/frms/"
OUT_DIR = "./output/frms/"

#-----
class InstrNet(object):
    def __init__(self) -> None:

        self.cfg_pth_path_ = './pretrained_instr/models/pretrained_model.pth'
        self.cfg_viz_ = False
        self.cfg_save_ = False
        #----
        parser = argparse.ArgumentParser()
        parser.add_argument('--focal-length', type=float, default=1390.0277099609375/(2208/640))  # ZED intrinsics per default
        parser.add_argument('--baseline', type=float, default=0.12)  # ZED intrinsics per default
        parser.add_argument('--aux-modality', type=str, default='disp', choices=['depth', 'disp'])
        parser.add_argument('--alpha', type=float, default=0.4)
        args = parser.parse_args()
        self.args_ = args

    #----
    def init(self):
        self.frm_idx_ = 0
        args = self.args_
        # load net
        net = Predictor(state_dict_path=self.cfg_pth_path_, focal_length=args.focal_length, baseline=args.baseline, return_depth=True if args.aux_modality == 'depth' else False)
        self.net_ = net

        #-----------

        #----
        if self.cfg_save_:
            wdir = OUT_DIR
            os.makedirs(os.path.join(wdir), exist_ok=True)
            os.makedirs(os.path.join(wdir, 'disp'), exist_ok=True)
            os.makedirs(os.path.join(wdir, 'disp_vis'), exist_ok=True)
            os.makedirs(os.path.join(wdir, 'segm'), exist_ok=True)
            os.makedirs(os.path.join(wdir, 'segm_vis'), exist_ok=True)

        return True
    
    #----
    def run(self, imL, imR):
        net = self.net_
        args = self.args_

        self.frm_idx_ += 1
        fi = self.frm_idx_

        with torch.no_grad():
            pred_segmap, pred_disp = net.predict(imL, imR)

        print("run pred done")

        #----
        sz = (imL.shape[1], imR.shape[0])
        print("sz=", sz)
        left1 = cv2.resize(imL, (640, 480), interpolation=cv2.INTER_LINEAR)
        left_overlay = net.colorize_preds(torch.from_numpy(pred_segmap).unsqueeze(0), rgb=left1, alpha=args.alpha)
        im_segmv = cv2.resize(left_overlay, sz)
        print("pred_disp range:", (pred_disp.min(), pred_disp.max()))

        print("pred_disp shape:", pred_disp.shape)
        im_dispv = pred_disp.astype(np.uint8)
        im_dispv = cv2.resize(im_dispv, sz)

        if self.cfg_viz_:
            cv2.imshow('segm overlay',  im_segmv)
            cv2.imshow("Disparity vis", im_dispv)
            cv2.waitKey(1)
        #------
        #print("pred_segmap dim:", pred_segmap.shape)
        #-----

        if self.cfg_save_:
            print("instr save frm ", str(fi))
            cv2.imwrite(OUT_DIR + "segm/" + str(fi)+ '.png', pred_segmap)
            cv2.imwrite(OUT_DIR + "segm_vis/" + str(fi)+ '.png', im_segmv)
            write_pfm(OUT_DIR + "disp/" + str(fi)+ '.pfm', pred_disp)
            cv2.imwrite(OUT_DIR + "disp_vis/" + str(fi)+ '.png', im_dispv)
        return


#-----
def test_imgs():
    ins = InstrNet()
    ins.init()
    ins.cfg_viz_ = True

    #----
    for i in range(1,100):
        sfL = TEST_FRMS + "left_rctf/" + str(i) +".png"
        sfR = TEST_FRMS + "right_rctf/" + str(i) +".png"

        imL = cv2.imread(sfL)
        imR = cv2.imread(sfR)
        if imL is None:
            break    
        #----
        ins.run(imL, imR)

    #---
    print("done")    
    return

#------
if __name__ == '__main__':
    test_imgs()

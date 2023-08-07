"""
Demo script.
"""

import os
import argparse
import cv2
import torch
import glob
import numpy as np
from predictor import Predictor

DFLT_IMGS_DIR = './rundir/frms/'

class Camera:
    def __init__(self, *args, **kwargs):
        self.idx_ = 0
        self.imgs_dir_ = kwargs["imgs_dir"]
        sdirL = self.imgs_dir_ + 'left/'
        sdirR = self.imgs_dir_ + 'right/'

        print('search imgs in '+sdirL)
        imLs = glob.glob(sdirL+'*.png')
        print("found imLs : ", len(imLs))

        self.sfImgs_ = [] 
        for s in imLs:
            s0 = s.split('/')[-1]
            self.sfImgs_.append(s0)
            print("  ", s0)

        return
    #---
    def getNum(self):
        return len(self.sfImgs_) 

    #---
    def get_stereo(self):
        if self.idx_ >= len(self.sfImgs_) :
            return None, None

        print("get_stereo() idx=", self.idx_)

        sf = self.sfImgs_[self.idx_]
        self.idx_ += 1
        sfL = self.imgs_dir_ + "left/" + sf
        sfR = self.imgs_dir_ + "right/" + sf
        print("load img L/R:", sfL, ", ", sfR)
        imL = cv2.imread(sfL)
        imR = cv2.imread(sfR)

        return imL, imR

def demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict', type=str, default='./pretrained_instr/models/pretrained_model.pth')
    parser.add_argument('--focal-length', type=float, default=1390.0277099609375/(2208/640))  # ZED intrinsics per default
    parser.add_argument('--baseline', type=float, default=0.12)  # ZED intrinsics per default
    parser.add_argument('--viz', default=False, action='store_true')
    parser.add_argument('--imgs-dir', type=str, default=DFLT_IMGS_DIR)
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--save-dir', type=str, default='./rundir/output')
    parser.add_argument('--aux-modality', type=str, default='depth', choices=['depth', 'disp'])
    parser.add_argument('--alpha', type=float, default=0.4)
    args = parser.parse_args()

    if args.save:
        print(f"Saving images to {args.save_dir}")
        os.makedirs(os.path.join(args.save_dir), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'left'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'right'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'pred'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'overlay'), exist_ok=True)

    # load net
    net = Predictor(state_dict_path=args.state_dict, focal_length=args.focal_length, baseline=args.baseline, return_depth=True if args.aux_modality == 'depth' else False)

    # init zed
    cam = Camera(imgs_dir=DFLT_IMGS_DIR)
    N = cam.getNum()
    print("cam imgs: N=",N)

    ctr = 0
    # main forward loop
    for i in range(N):
        print("getting img i=",i)
        left, right = cam.get_stereo()
        if left is None:
            break

        with torch.no_grad():
            pred_segmap, pred_depth = net.predict(left, right)

        if args.viz:
            left = cv2.resize(left, (640, 480), interpolation=cv2.INTER_LINEAR)
            left_overlay = net.colorize_preds(torch.from_numpy(pred_segmap).unsqueeze(0), rgb=left, alpha=args.alpha)
            cv2.imshow('left', cv2.resize(left.copy(), (640, 480), interpolation=cv2.INTER_LINEAR))
            cv2.imshow('right', cv2.resize(right.copy(), (640, 480), interpolation=cv2.INTER_LINEAR))
            cv2.imshow('pred', left_overlay)
            cv2.imshow(args.aux_modality, pred_depth / pred_depth.max())
            cv2.waitKey(1)

        if args.save:
            cv2.imwrite(os.path.join(args.save_dir, 'left', str(ctr).zfill(6) + '.png'), left)
            cv2.imwrite(os.path.join(args.save_dir, 'right', str(ctr).zfill(6) + '.png'), right)
            np.save(os.path.join(args.save_dir, 'depth', str(ctr).zfill(6) + '.npy'), pred_depth)
            cv2.imwrite(os.path.join(args.save_dir, 'segmap', str(ctr).zfill(6) + '.png'), pred_segmap)
            left_overlay = net.colorize_preds(torch.from_numpy(pred_segmap).unsqueeze(0), rgb=left, alpha=args.alpha)
            cv2.imwrite(os.path.join(args.save_dir, 'overlay', str(ctr).zfill(6) + '.png'), left_overlay)

            ctr += 1

#------
def test1():
    cam = Camera(imgs_dir=DFLT_IMGS_DIR)
    N = cam.getNum()
    for i in range(N):
        imL, imR = cam.get_stereo()


    return

#------
if __name__ == '__main__':
    demo()
#    test1()

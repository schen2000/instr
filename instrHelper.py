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
from utils.file_io import write_pfm

TEST_FRMS = "./output/frms/"
#-----
class InstrNet(object):
    def __init__(self) -> None:

        self.cfg_pth_path_ = './pretrained_instr/models/pretrained_model.pth'
        self.cfg_viz_ = False
        #----
        parser = argparse.ArgumentParser()
 #       parser.add_argument('--state-dict', type=str, default='./pretrained_instr/models/pretrained_model.pth')
        parser.add_argument('--focal-length', type=float, default=1390.0277099609375/(2208/640))  # ZED intrinsics per default
        parser.add_argument('--baseline', type=float, default=0.12)  # ZED intrinsics per default
#        parser.add_argument('--viz', default=False, action='store_true')
#        parser.add_argument('--imgs-dir', type=str, default=DFLT_IMGS_DIR)
        parser.add_argument('--save', default=False, action='store_true')
        parser.add_argument('--save-dir', type=str, default='./rundir/output')
        parser.add_argument('--aux-modality', type=str, default='depth', choices=['depth', 'disp'])
        parser.add_argument('--alpha', type=float, default=0.4)
        args = parser.parse_args()
        self.args_ = args

        #----
        if args.save:
            print(f"Saving images to {args.save_dir}")
            os.makedirs(os.path.join(args.save_dir), exist_ok=True)
            os.makedirs(os.path.join(args.save_dir, 'depth'), exist_ok=True)
            os.makedirs(os.path.join(args.save_dir, 'pred'), exist_ok=True)
            os.makedirs(os.path.join(args.save_dir, 'segmap'), exist_ok=True)
            os.makedirs(os.path.join(args.save_dir, 'overlay'), exist_ok=True)
        return 

    #----
    def init(self):
        args = self.args_
        # load net
        net = Predictor(state_dict_path=self.cfg_pth_path_, focal_length=args.focal_length, baseline=args.baseline, return_depth=True if args.aux_modality == 'depth' else False)
        self.net_ = net
        return True
    
    #----
    def run(self, imL, imR):
        net = self.net_
        args = self.args_
        with torch.no_grad():
            pred_segmap, pred_depth = net.predict(imL, imR)

        print("run pred done")

        #----
        if self.cfg_viz_:
            left = cv2.resize(imL, (640, 480), interpolation=cv2.INTER_LINEAR)
            left_overlay = net.colorize_preds(torch.from_numpy(pred_segmap).unsqueeze(0), rgb=left, alpha=args.alpha)
            cv2.imshow('pred', left_overlay)
            cv2.imshow(args.aux_modality, pred_depth / pred_depth.max())
            cv2.waitKey(1)
        #------
        #print("pred_segmap dim:", pred_segmap.shape)
        #-----

        if args.save:
            print("saving output...")
            #cv2.imwrite(os.path.join(args.save_dir, 'left', str(ctr).zfill(6) + '.png'), left)
            #cv2.imwrite(os.path.join(args.save_dir, 'right', str(ctr).zfill(6) + '.png'), right)
            np.save(os.path.join(args.save_dir, 'depth', str(ctr).zfill(6) + '.npy'), pred_depth)
            cv2.imwrite(os.path.join(args.save_dir, 'segmap', str(ctr).zfill(6) + '.png'), pred_segmap)
            left_overlay = net.colorize_preds(torch.from_numpy(pred_segmap).unsqueeze(0), rgb=left, alpha=args.alpha)
            cv2.imwrite(os.path.join(args.save_dir, 'overlay', str(ctr).zfill(6) + '.png'), left_overlay)

            #---- depth
            #sfd = os.path.join(args.save_dir, 'depth', sf.replace('.png', '.pfm'))
            #write_pfm(sfd, pred_depth)
            #ctr += 1
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

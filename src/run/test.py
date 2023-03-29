"""
File: test.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: test script
"""
from pathlib import Path

import argparse
import os
import cv2
import numpy as np
import torch
from imutils import paths

import sys
sys.path.append("..") # Add parent directory to system path

from core.trainers.filter_trainer import LitModel


def get_output(ckpt_path, model, gpu=True):
    data = cv2.imread(img_path) / 255.
    data = np.transpose(data, [2, 0, 1])
    data = torch.tensor(data).unsqueeze(0)
    data = data.float()

    if gpu:
        output = model(data.cuda()).squeeze().cpu().numpy()
    else:
        output = model(data).squeeze().numpy()

    output = np.transpose(output, [1, 2, 0])
    output = np.clip(output, 0, 1)
    output = output * 255
    output = output.astype(np.uint8)

    return output


if __name__ == "__main__":
    ckpt_path = './ckpt/Photographic-Style/epoch=138.ckpt'
    # ckpt_path = './ckpt/L0-smoothing/_ckpt_epoch_79.ckpt'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='Multiscale-Tone',
        help=('Photographic-Style | L0-smoothing | Pencil | Multiscale-Tone'))
    
    args = parser.parse_args()
    
    if args.model_name == "Photographic-Style":
        ckpt_path = "./ckpt/Photographic-Style/epoch=138.ckpt"
    elif args.model_name == "L0-smoothing":
        ckpt_path = "./ckpt/L0-smoothing/_ckpt_epoch_79.ckpt"
    elif args.model_name == "Pencil":
        ckpt_path = "./ckpt/Pencil/epoch=138.ckpt"
    elif args.model_name == "Multiscale-Tone":
        ckpt_path = "./ckpt/Multiscale-Tone/epoch=178.ckpt"
    else:
        ckpt_path = "./ckpt/Multiscale-Tone/epoch=178.ckpt"
        
    os.makedirs(args.output, exist_ok=True)
    
    imgs_dir = args.input
    out_dir = args.output
    gpu = True

    if gpu:
        model = LitModel.load_from_checkpoint(ckpt_path).cuda()
    else:
        model = LitModel.load_from_checkpoint(ckpt_path,
                                              map_location=torch.device('cpu'))
    model.eval()
    model.freeze()

    imgs_path = paths.list_images(imgs_dir)
    for img_path in imgs_path:
        out_name = Path(img_path).stem + '_out.png'
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = str(Path(out_dir).joinpath(out_name))
        out = get_output(ckpt_path, model)
        cv2.imwrite(out_path, out)

# coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt


def read_img(image_path, gray_scale=False):
    image_src = cv2.imread(image_path)
    if gray_scale:
        image_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    else:
        # image_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
        image_rgb = image_src
    return image_rgb


def mirror_img(image_path, gray_scale=False, debug=False):
    """ 图像镜像翻转 """
    image_rgb = read_img(image_path, gray_scale)
    image_fliplr = np.fliplr(image_rgb)
    #image_flipud = np.flipud(image_rgb)
    if debug:
        cv2.imwrite('./image_fliplr.jpg', image_fliplr)  # 左右翻转
        #cv2.imwrite('./image_flipud.jpg', image_flipud)  # 上下翻转
    return image_fliplr#, image_flipud


img_path = './img/1.jpg'
mirror_img(img_path, gray_scale=False, debug=True)

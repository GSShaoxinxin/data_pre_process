#!/usr/bin/python

'''
    OpenCV Image Alignment  Example
    
    Copyright 2015 by Satya Mallick <spmallick@learnopencv.com>
    
'''

import cv2
import numpy as np
import tifffile as tiff


def read_img_for_ecc(img_file_path):
    if img_file_path.endswith(".JPG"):
        tmp = cv2.imread(img_file_path)

        img = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

        img = img.astype(dtype=np.float32)
        img = img * 256
    elif img_file_path.endswith(".TIF"):
        img = tiff.imread(img_file_path)
        img = img.astype(dtype=np.float32)
    return img


def bias_xy_use_ecc(img1_file_path, img2_file_path):
    im1 = read_img_for_ecc(img1_file_path)
    im2 = read_img_for_ecc(img2_file_path)
    sz = im1.shape
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000;

    # Specify the threshold of the increment 指定增量的阈值
    # in the correlation coefficient between two iterations 两次迭代之间的相关系数
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1, im2, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    print("hello")

    im2_aligned = im2_aligned.astype(dtype = np.uint16)
    # Show final results
    #cv2.imshow("Image 1", im1)
    #cv2.imshow("Image 2", im2)
    cv2.imshow("Aligned Image 2", im2_aligned)
    cv2.waitKey(0)
    #x y 位置偏移
    return warp_matrix[0,2], warp_matrix[1,2],





if __name__ == '__main__':
    im1_file_path =r"E:\appData\pycharmdata\DataPreProcess\image_registration\images\DJI_1200.JPG"
    im2_file_path =r"E:\appData\pycharmdata\DataPreProcess\image_registration\images\DJI_1201.TIF"
    bias_xy_use_ecc(im1_file_path, im2_file_path)
#大疆精灵4多光谱无人机图像配准
#功能：配准多光谱图像
#参考资料：大疆多光谱图像处理指南 https://dl.djicdn.com/downloads/p4-multispectral/20200717/P4_Multispectral_Image_Processing_Guide_CHS.pdf
#输出将会在同级目录新建output_idx后缀文件夹
import cv2
import os
import yaml
import glob
import re
import tifffile as tiff
from PIL import Image
import numpy as np
import math
#文件名称中序号代表不同波段
RGB_0=0
BLUE_1 = 1
GREEN_2 =2
RED_3 = 3
RE_4 =4
NIR_5 =5
#BIAS_X=[4.75000, -7.34375, -2.90625, -4.65625, -2.93750, 0.00000]
#BIAS_Y=[16.93750, -0.21875, -2.15625, 6.25000, 5.31250, 0.00000]

def parse_args():
    with open('split4.yaml', 'r', encoding='utf-8') as y:
        cfg = yaml.full_load(y)
        src_path = cfg['src_path']
        w = cfg['w']
        h = cfg['h']
        return src_path, w, h #dsc_path


def create_output_bro_dir(src_path):
    dsc_path=''
    for i in range(50):
        dsc_path = src_path + '_split_' + str(i)
        if (not os.path.exists(dsc_path)):  # 如果目标文件夹不存在，则新建
            os.mkdir(dsc_path)
            break;
    return dsc_path
def copy_create_dir(src_root_path,dsc_root_path):
    for dirpath, dirnames, filenames in os.walk(src_root_path, topdown=True):
        for dir in dirnames:
            src = os.path.join(dirpath, dir)
            path = src.replace(src_root_path, dsc_root_path)
            if (os.path.isdir(src) and not os.path.exists(path)):  # 是目录则复制目录
                os.mkdir(path)  # 处理嵌套文件夹

def read_img_for_ecc(img_file_path):
    print("文件"+img_file_path)
    if img_file_path.endswith(".JPG"):
        tmp = cv2.imread(img_file_path)

        img = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

        img = img.astype(dtype=np.float32)
        img = img * 256
    elif img_file_path.endswith(".TIF"):
        img = tiff.imread(img_file_path)
        img = img.astype(dtype=np.float32)
    return img



def gauss_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)


def get_bias_one(img1_file_path, img2_file_path):
    im1 = read_img_for_ecc(img1_file_path)

    im2 = read_img_for_ecc(img2_file_path)
    im1 = gauss_blur(im1)
    im2 = gauss_blur(im2)
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
    try:
        (cc, warp_matrix) = cv2.findTransformECC(im1, im2, warp_matrix, warp_mode, criteria)
    except cv2.error:
        print("不收敛 im1："+img1_file_path+"img2:"+img2_file_path+"\n")
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
    #cv2.imshow("Aligned Image 2", im2_aligned)
    #cv2.waitKey(0)
    #x y 位置偏移
    return warp_matrix[0, 2], warp_matrix[1, 2],



def get_bias_list_XY(dirpath,rgbname):
    rgb_file_path = os.path.join(dirpath,rgbname)
    bias_X = []
    bias_Y = []
    rgb_file_name = os.path.splitext(rgbname)[-1]
    m5_path = os.path.join(dirpath,mltspcttxt_from_rgb(rgbname, 5))
    # 每一张都与5光谱进行操作
    for i in range(5):
        if (0 == i):
            tmp_biasx, tmp_biasy = get_bias_one(m5_path, rgb_file_path)
        else:
            tmp_biasx, tmp_biasy = get_bias_one(m5_path, mltspcttxt_from_rgb(rgb_file_path, i))
        bias_X.append(tmp_biasx)
        bias_Y.append(tmp_biasy)
    bias_X.append(0) #5光谱自己对自己的偏移是0
    bias_Y.append(0)
    return bias_X,bias_Y


def split4(src_root_path,dsc_root_path,w,h):
    for dirpath, dirnames, filenames in os.walk(src_root_path, topdown=True):
        #RGB_filenames = [item for item in filenames if item.endswith('.JPG')]
        #TIF_filenames = [item for item in filenames if item.endswith('.TIF')]
        #PNG_filenames = [item for item in filenames if item.endswith('.png')]
        for filename in filenames:

            path0 = os.path.join(dirpath, filename.split(".")[0]).replace(src_root_path,dsc_root_path) + "_0"+  os.path.splitext(filename)[-1]
            path1 = os.path.join(dirpath, filename.split(".")[0]).replace(src_root_path,dsc_root_path) + "_1"+  os.path.splitext(filename)[-1]
            path2 = os.path.join(dirpath, filename.split(".")[0]).replace(src_root_path,dsc_root_path) + "_2"+  os.path.splitext(filename)[-1]
            path3 = os.path.join(dirpath, filename.split(".")[0]).replace(src_root_path,dsc_root_path) + "_3"+  os.path.splitext(filename)[-1]
            if(filename.endswith(".JPG")):
                img = cv2.imread(os.path.join(dirpath,filename))
                [h, w] = img.shape[:2]
                print(path0, (h, w))
                l1img = img[:int(h / 2), :int(w / 2), :]
                l2img = img[int(h / 2 + 1):, :int(w / 2), :]
                r1img = img[:int(h / 2), int(w / 2 + 1):, :]
                r2img = img[int(h / 2 + 1):, int(w / 2 + 1):, :]
                cv2.imwrite(path0, l1img)
                cv2.imwrite(path1, l2img)
                cv2.imwrite(path2, r1img)
                cv2.imwrite(path3, r2img)
            elif(filename.endswith(".png")):
                img = cv2.imread(os.path.join(dirpath, filename),cv2.IMREAD_GRAYSCALE)
                [h, w] = img.shape[:2]
                print(path0, (h, w))
                l1img = img[:int(h / 2), :int(w / 2), ]
                l2img = img[int(h / 2 + 1):, :int(w / 2), ]
                r1img = img[:int(h / 2), int(w / 2 + 1):, ]
                r2img = img[int(h / 2 + 1):, int(w / 2 + 1):, ]
                cv2.imwrite(path0, l1img)
                cv2.imwrite(path1, l2img)
                cv2.imwrite(path2, r1img)
                cv2.imwrite(path3, r2img)
            elif(filename.endswith(".TIF")):
                img = tiff.imread(os.path.join(dirpath, filename))
                [h, w] = img.shape[:2]
                print(path0, (h, w))
                l1img = img[:int(h / 2), :int(w / 2), ]
                l2img = img[int(h / 2 + 1):, :int(w / 2), ]
                r1img = img[:int(h / 2), int(w / 2 + 1):, ]
                r2img = img[int(h / 2 + 1):, int(w / 2 + 1):, ]
                tiff.imwrite(path0, l1img)
                tiff.imwrite(path1, l2img)
                tiff.imwrite(path2, r1img)
                tiff.imwrite(path3, r2img)





def main():
    src_root_path, w, h = parse_args()
    dsc_root_path = create_output_bro_dir(src_root_path)#创建兄弟目录作为输出目录
    copy_create_dir(src_root_path,dsc_root_path)
    #按名称读取文件
    split4(src_root_path,dsc_root_path,w,h)





if __name__ == '__main__':
    main()
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
#文件名称中序号代表不同波段
RGB_0=0
BLUE_1 = 1
GREEN_2 =2
RED_3 = 3
RE_4 =4
NIR_5 =5
BIAS_X=[4.75000, -7.34375, -2.90625, -4.65625, -2.93750, 0.00000]
BIAS_Y=[16.93750, -0.21875, -2.15625, 6.25000, 5.31250, 0.00000]

def parse_args():
    with open('image_ecc_registration.yaml', 'r', encoding='utf-8') as y:
        cfg = yaml.full_load(y)
        src_path = cfg['src_path']
        return src_path #dsc_path

def cmpt_overlap_in_base(bias_x, bias_y, w, h):#计算在基准坐标下获得的重叠区域，以波段5为基准坐标
    # 构建长方形
    rects = getRects(bias_x, bias_y, w, h)
    (ltx, lty, rbx, rby) = rects[5]
    for i in range(len(bias_x)-1):    #算出重叠区
        (ltx1, lty1, rbx1, rby1) = rects[i]
        ltx = max(ltx, ltx1)
        lty = max(lty, lty1)
        rbx = min(rbx, rbx1)
        rby = min(rby, rby1)
    return (ltx, lty, rbx, rby)

def getRects(x_array,y_array,w,h):
    ret=[]
    for i in range(len(x_array)):
        ret.append((-x_array[i],-y_array[i],-x_array[i]+w,-y_array[i]+h))
    return ret

def get_coordinate_in_bias(bias_x,bias_y,ltx,lty,rbx,rby):
    corr_arr=[]
    for i in range(len(bias_x)):
        corr_arr.append((ltx+bias_x[i],lty+bias_y[i],rbx+bias_x[i],rby+bias_y[i]))
    return corr_arr
def mltspctname_from_rgbname(rgb_name, index):#根据可见光图的名称生成对应index波段的多光谱图像的名称
    id_begin = 0
    id_end = 0
    matches = re.finditer("\d{4}", rgb_name)
    for item in matches:  # 如何取迭代对象的第一个元素，写的不够好
        (id_begin, id_end) = item.span()
        break;
        # 返回的是下标
    return rgb_name[:id_begin + 3]+str(index)+".TIF"


def create_output_bro_dir(src_path):
    dsc_path=''
    for i in range(50):
        dsc_path = src_path + '_output_' + str(i)
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
def xy_registration(src_root_path,dsc_root_path):
    for dirpath, dirnames, filenames in os.walk(src_root_path, topdown=True):
        RGB_filenames = [item for item in filenames if item.endswith('.JPG')]
        for rgb_name in RGB_filenames:
            img = Image.open(os.path.join(dirpath, rgb_name))  # cv2.imread(rgb_path,cv2.IMREAD_UNCHANGED)
            width = img.width
            height = img.height
            (ltx, lty, rbx, rby) = cmpt_overlap_in_base(BIAS_X, BIAS_Y, width, height)
            coordination = get_coordinate_in_bias(BIAS_X, BIAS_Y, ltx, lty, rbx, rby)  # 得到重叠部分在每个坐标中的像素值
            # 构造出对应的多光谱图像名称并读取
            for i in range(len(coordination)):
                (ltx, lty, rbx, rby) = coordination[i]
                ltx = round(ltx)
                lty = round(lty)
                rbx = round(rbx)
                rby = round(rby)
                if (0 == i):
                    filename = rgb_name
                    a = Image.open(os.path.join(dirpath, filename))
                    b = a.crop((ltx, lty, rbx, rby))
                    # 保存图像
                    b.save(os.path.join(dirpath.replace(src_root_path,dsc_root_path), filename))

                else:
                    filename = mltspctname_from_rgbname(rgb_name, i)
                    # 1.打开文件，进行裁剪
                    a = tiff.imread(os.path.join(dirpath, filename))
                    b = a[lty:rby, ltx:rbx]
                    tiff.imwrite(os.path.join(dirpath.replace(src_root_path,dsc_root_path), filename), b)


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




def main():
    src_root_path = parse_args()
    dsc_root_path = create_output_bro_dir(src_root_path)#创建兄弟目录作为输出目录
    copy_create_dir(src_root_path,dsc_root_path)
    #按名称读取文件
    xy_registration(src_root_path,dsc_root_path)
    #ecc_registration(src_root_path,dsc_root_path)




if __name__ == '__main__':
    main()
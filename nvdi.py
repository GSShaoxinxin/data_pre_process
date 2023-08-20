#对多光谱无人机的原始数据做一些必要的处理，计算反射率不成功
'''参考资料： https://www.cnblogs.com/ludwig1860/p/14965019.html '''
import os
import yaml
import numpy as np
import shutil
import exiftool
import tifffile as tiff
import re
import matplotlib.pylab as plt
import matplotlib
matplotlib.use('TkAgg')
import cv2 as cv
def parse_args():
    with open('nvdi.yaml', 'r', encoding='utf-8') as y:
        cfg = yaml.full_load(y)
        src_dir = cfg['src_dir']

        return src_dir#,exiftool_path#, filenames_path, name_pre







def get_new_name(filename,delta):

    new_name=''
    #if(filename.endswit)
    filename_4num = re.search("\d{4}", filename).group()
    filename_bone = filename_4num[:-1]
    filename_last_num = filename_4num[-1]
    new_num =int(filename_last_num) + delta
    if(0 == new_num):
        post = ".JPG"
    else:
        post = ".TIF"
    new_name = filename.replace(filename_4num,filename_bone + str(new_num))
    new_name = os.path.splitext(new_name)[0]+post


    return new_name







def cmpt(Process_dir,dsc_dir):
    for dirpath, dirnames, filenames in os.walk(Process_dir, topdown=True):
        namelist = [os.path.join(dirpath,item) for item in filenames if item.endswith('.JPG')]
        for filename in namelist:
            r_name = get_new_name(filename, 3)
            nir_name = get_new_name(filename, 4)
            r_f = tiff.imread(os.path.join(dirpath, r_name))
            nir_f = tiff.imread(os.path.join(dirpath, nir_name))
            nvdi = (r_f-nir_f)/(r_f+nir_f)
            #处理脏数据
            fill_mean = np.mean(nvdi[np.isfinite(nvdi)])
            nvdi[np.isinf(nvdi)] = fill_mean
            nvdi[np.isnan(nvdi)] = fill_mean
            #放缩到0-1再乘65535 输出tiff
            nvdi = (nvdi - np.amin(nvdi)) / (np.amax(nvdi) - np.amin(nvdi))
            nvdi=(65535*nvdi).astype(np.uint16)

            tiff.imwrite(os.path.join(dirpath, get_new_name(filename, 7)).replace(Process_dir,dsc_dir),nvdi)
            print("hello")


def create_output_bro_dir(src_path):
    dsc_path=''
    for i in range(50):
        dsc_path = src_path + '_nvdi_' + str(i)
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



def main():
    src_dir = parse_args()
    dsc_root_path = create_output_bro_dir(src_dir)  # 创建兄弟目录作为输出目录
    copy_create_dir(src_dir, dsc_root_path)
    # 按名称读取文件
    cmpt(src_dir,dsc_root_path)

if __name__ == '__main__':
   main()
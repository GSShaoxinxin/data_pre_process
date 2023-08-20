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
    with open('cmpt_reflectance.yaml', 'r', encoding='utf-8') as y:
        cfg = yaml.full_load(y)
        src_dir = cfg['src_dir']
        exiftool_path = cfg['exiftool_path']
        return src_dir,exiftool_path#, filenames_path, name_pre



# Numpy function
def euclidean_distance(image,ucntr,vcntr):
    center = np.array([ucntr, vcntr])
    distances = np.linalg.norm(np.indices(image.shape) - center[:,None,None] + 0.5, axis = 0)

    return distances


def black_degree(dn,meta):
    u_cntr, v_cntr = meta['XMP:VignettingCenter']
    k_list = meta['XMP:VignettingPolynomial']

    r_matrix = euclidean_distance(dn, u_cntr, v_cntr)
    #   DN3=DN2*(k5*r^6+k4*r^5+k3*r^4+k2*r^3+k1*r^2+k0*r+1)

    p = np.poly1d([k_list[5], k_list[4],k_list[3],k_list[2],k_list[1],k_list[0],1])
    r_2 =p(r_matrix)
    dn1=np.multiply(dn, r_2)
    return dn1

def get_new_name(filename,delta):

    new_name=''
    #if(filename.endswit)
    filename_4num = re.search("\d{4}", filename).group()
    filename_bone = filename_4num[:-1]
    filename_last_num = filename_4num[-1]
    new_name = filename.replace(filename_4num,filename_bone + str(int(filename_last_num) + delta))

    return new_name




def calibrate_img(src_dir,dsc_dir, meta): # one_meta是exiftool执行后的返回值

    filename = meta["File:FileName"]#os.path.splitext(src)[-1]

    #new_name = get_new_name(filename,1)
    file_path = os.path.join(src_dir,filename)
    if filename.endswith(".JPG"):
       # img = tiff.imread(0)
        shutil.copy(file_path,os.path.join(dsc_dir,filename))
        print('hello')
        return
    elif(filename.endswith(".TIF")):
        #dn_0 = tiff.imread(file_path)
        #dn = dn_0/np.linalg .norm(dn_0)
        dn = tiff.imread(file_path)
        # 暗电流校正
        #dn1_0 = dn-meta["XMP:BlackCurrent"]
        #dn1 = dn1_0/np.linalg .norm(dn1_0)
        dn1=dn-meta["XMP:BlackCurrent"]
        # 曝光时间与增益校正
        #dn1=dn1/np.linalg.norm(dn1)#进行一次归一化

        # 暗角校正
        dn2 = black_degree(dn1, meta)
        dn3 = dn2/(meta['XMP:SensorGain']*meta['XMP:ExposureTime']/1e6)
        dn3 = (dn3 - np.amin(dn3)) / (np.amax(dn3) - np.amin(dn3))
        #dn3 = dn3 / np.linalg.norm(dn3)
        #dn5 = dn3[dn3>0.5]=dn3 #让影子全黑，然后扩展树木
        dn4 = 65535*dn3
        #dn4 = np.mat(dn4)
        #dn5 = 65535*dn3
        #dn5 = dn5.astype(np.float32)
        #hist1 = cv.calcHist([dn5],[0],None,[65536],[0,65535])
        #plt.plot(hist1)
        #plt.show()
        #直方图均衡化

        tiff.imwrite(os.path.join(dsc_dir,filename),dn4.astype(np.uint16))

    else:
        print("没有实现这种后缀")
        return



def cmpt(Process_dir,dsc_dir):
    for dirpath, dirnames, filenames in os.walk(Process_dir, topdown=True):
        namelist = [os.path.join(dirpath,item) for item in filenames if item.endswith('.JPG') or item.endswith('.TIF')]
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(namelist)
            for d in metadata:
                print("{:20.20} {:20.20}".format(d["SourceFile"],
                                                 d["EXIF:DateTimeOriginal"]))
                calibrate_img(dirpath,dirpath.replace(Process_dir,dsc_dir),d)


def create_output_bro_dir(src_path):
    dsc_path=''
    for i in range(50):
        dsc_path = src_path + '_reflect_' + str(i)
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
    src_dir,exiftool_path = parse_args()
    dsc_root_path = create_output_bro_dir(src_dir)  # 创建兄弟目录作为输出目录
    copy_create_dir(src_dir, dsc_root_path)
    # 按名称读取文件
    cmpt(src_dir,dsc_root_path)

if __name__ == '__main__':
   main()
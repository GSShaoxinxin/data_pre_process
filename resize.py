#缩放成正方形,并将结果存储在同级目录下，目录命名为多加一个数字后缀 尽量用英文目录
import os
import cv2
import yaml
import shutil
import tifffile as tiff

from skimage.transform import resize
def get_min_size(Process_dir):
    min_w=3000
    min_h=3000
    for dirpath, dirnames, filenames in os.walk(Process_dir, topdown=True):
        for filename in filenames:
            src_name = os.path.join(dirpath, filename)
            if (src_name.endswith(".JPG") or src_name.endswith(".png")):
                img = cv2.imread(src_name)
                min_w = min(min_w, img.shape[1])
                min_h = min(min_h, img.shape[0])
            elif (src_name.endswith(".TIF") or src_name.endswith(".tif")):
                img = tiff.imread(src_name)
                min_w = min(min_w, img.shape[1])
                min_h = min(min_h, img.shape[0])
            else:
                print("未知类型")
            if(min_w <1300):
                print("hello")


    return min_w,min_h

def main():

    with open('resize.yaml', 'r', encoding='utf-8') as y:
        cfg = yaml.full_load(y)
        Process_dir = cfg['Process_dir']
        max_dir_num = cfg['max_dir_num']
        out_size_w = cfg['out_size_w']
        out_size_h = cfg['out_size_h']
    father_path = os.path.abspath(os.path.join(Process_dir, os.path.pardir))
    me = os.path.split(Process_dir)[-1]
    bro_path = ''
    bro =''
    #在Process_dir 同级目录下创建目标文件夹
    i=1
    for i in range(1,max_dir_num):#range是左闭右开区间
        bro = me+str(i)
        bro_path = os.path.join(father_path,bro)
        if(not os.path.exists(bro_path)):
            os.mkdir(bro_path)
            break;
    if(50==i):
        print("文件夹命名已达到最大值，删除一些后可再次运行")
        return;

    if(out_size_w<=0 or out_size_h<=0):
        out_size_w,out_size_h=get_min_size(Process_dir)
        print("宽:"+str(out_size_w)+",高:"+str(out_size_h)+"\n")


    #迭代 复制每一级子目录
    for dirpath, dirnames, filenames in os.walk(Process_dir, topdown=True):
        path=''
        for dir in dirnames:
            src = os.path.join(dirpath,dir)
            path = src.replace(Process_dir,bro_path)
            if( os.path.isdir(src) and not os.path.exists(path)):#是目录则复制目录
                os.mkdir(path)

        for filename in filenames:
            src_name = os.path.join(dirpath, filename)
            dsc_name = os.path.join(dirpath.replace(Process_dir, bro_path), filename)
            #print(dsc_name)
            #shutil.copyfile(src_name, dsc_name)
            if(filename.endswith('.JPG')):
                img_array = cv2.imread(src_name, cv2.IMREAD_COLOR)
                resize_img = cv2.resize(img_array, (out_size_w, out_size_h), interpolation=cv2.INTER_CUBIC)
            #print("successfully resize " + filename)
                cv2.imwrite(dsc_name, resize_img)
            elif(filename.endswith('.png')):
                img_array = cv2.imread(src_name, cv2.IMREAD_GRAYSCALE)
                resize_img = cv2.resize(img_array, (out_size_w, out_size_h), interpolation=cv2.INTER_CUBIC)#cv2 imread 0是高度，1是宽度。resize 先宽度，后高度
            #print("successfully resize " + filename)
                cv2.imwrite(dsc_name, resize_img)
            elif(filename.endswith('.TIF')):
                '''img = tiff.imread(src_name)
                img = resize(img, (out_size_w, out_size_h),
                             anti_aliasing=True)  # resize_shape 代表要更改至的长宽大小'''
                img_array = tiff.imread(src_name)
                #resize_img = tiff.resize(img_array, (out_size_w, out_size_h), interpolation=cv2.INTER_CUBIC)
                #resized_data = resize(img_array, (out_size_w, out_size_h, 1))
                resized_data = img_array[:out_size_h, :out_size_w]
                tiff.imwrite(dsc_name, resized_data, planarconfig='CONTIG')


if __name__ == '__main__':
    main()
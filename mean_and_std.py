#计算数据集的均值和方差

import yaml
import re
import numpy as np
import cv2
import random
import os
import tifffile as tiff
def parse_args():
    with open('mean_and_std.yaml', 'r', encoding='utf-8') as y:
        cfg = yaml.full_load(y)
        Process_dir = cfg['Process_dir']
        rgb_or_tif = cfg['rgb_or_tif']
        return Process_dir,os.path.join(Process_dir,'filenames.txt'),rgb_or_tif
def path_to_file():
    Process_dir,txt_path,rgb_or_tif = parse_args()
    if('rgb' == rgb_or_tif):
        re_pttrn = '.jpg|.png|.JPG|.PNG'
    elif('tif' == rgb_or_tif):
        re_pttrn = '.tif|.TIF'
    with open(txt_path, "w") as f: #|w|打开文件只用于写入。若文件存在则打开该文件，并从开头开始编辑，即原有内容会删除。若文件不存在则自动创建新文件。|
        for dirpath, dirnames, filenames in os.walk(Process_dir, topdown=True):
            #ground_truth 文件夹不要写入
            if(re.search('ground_truth',dirpath)):
                continue
            for filename in filenames:
                if re.search(re_pttrn, filename) :
                        f.write(os.path.join(dirpath, filename) + "\n")


def compute_mean_std():
    #path_to_file()
    Process_dir, txt_path,rgb_or_tif = parse_args()
    means = [0, 0, 0]
    stdevs = [0, 0, 0]
    index = 1
    num_imgs = 0
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        # random.shuffle(lines)
        print(lines)
        for line in lines:
            print(line)
            print('{}/{}'.format(index, len(lines)))
            index += 1
            a = os.path.join(line)
            # print(a[:-1])
            num_imgs += 1
            if('rgb' == rgb_or_tif):
                img = cv2.imread(a[:-1])
                img = np.asarray(img)
                img = img.astype(np.float32) / 255.
                for i in range(3):
                    means[i] += img[:, :, i].mean()
                    stdevs[i] += img[:, :, i].std()
            elif ('tif' == rgb_or_tif):
                img = tiff.imread(a[:-1])
                img = img/65535
                means[0] += img.mean()
                stdevs[0] += img.std()
            else:
                print('error')
                return
            #print(img)

    print(num_imgs)
    means.reverse()
    stdevs.reverse()

    means = np.asarray(means) / num_imgs
    stdevs = np.asarray(stdevs) / num_imgs

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))


if __name__ == '__main__':
    path_to_file()
    compute_mean_std()
#根据txt文本将指定文件拷贝到指定文件夹，并修改名为日期+原命名
import os
import yaml
import numpy as np
import shutil
def parse_args():
    with open('selected_copy.yaml', 'r', encoding='utf-8') as y:
        cfg = yaml.full_load(y)
        src_dir = cfg['src_dir']
        dsc_dir = cfg['dsc_dir']
        filenames_path = cfg['filenames_path']
        name_pre = cfg['name_pre']
        name_post = cfg['name_post']

        return src_dir, dsc_dir, filenames_path, name_pre, name_post


if __name__ == '__main__':
    src_dir, dsc_dir, filenames_path, name_pre, name_post = parse_args()
    matrix = np.loadtxt(filenames_path, dtype = str).tolist()
    if(not os.path.exists(dsc_dir)):
        os.mkdir(dsc_dir)
    for i in range(len(matrix)):
        print(matrix[i])
        tmp_src_filepath = os.path.join(src_dir,'DJI_'+str(matrix[i])+'.JPG')
        tmp_dsc_filepath = os.path.join(dsc_dir,name_pre+'DJI_'+str(matrix[i])+name_post)
        shutil.copyfile(tmp_src_filepath, tmp_dsc_filepath)

    print(type(matrix))
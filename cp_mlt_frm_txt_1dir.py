#根据txt文本将可见光和多光谱拷贝到指定文件夹，
import os
import yaml
import numpy as np
import shutil
def parse_args():
    with open('cp_mlt_frm_txt_1dir.yaml', 'r', encoding='utf-8') as y:
        cfg = yaml.full_load(y)
        src_dir = cfg['src_dir']
        dsc_dir = cfg['dsc_dir']
        filenames_path = cfg['filenames_path']
        name_pre = cfg['name_pre']


        return src_dir, dsc_dir, filenames_path, name_pre


if __name__ == '__main__':
    src_dir, dsc_dir, filenames_path, name_pre= parse_args()
    if(not os.path.exists(dsc_dir)):
        os.mkdir(dsc_dir)
    matrix = np.loadtxt(filenames_path, dtype = str).tolist()

    for i in range(len(matrix)):
        print(matrix[i])
        bone = str(matrix[i])[:-1]
        for j in range (6):
            if(0==j):
                file_name = 'DJI_' + bone +str(j)+'.JPG'
            else:
                file_name = 'DJI_' + bone +str(j) +'.TIF'
            tmp_src_filepath = os.path.join(src_dir,file_name )
            tmp_dsc_filepath = os.path.join(dsc_dir,name_pre+file_name)
            shutil.copyfile(tmp_src_filepath, tmp_dsc_filepath)

    print(type(matrix))
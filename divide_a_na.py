import os
import numpy as np
import yaml
import shutil
def parse_args():
    with open('divide_a_na.yaml', 'r', encoding='utf-8') as y:
        cfg = yaml.full_load(y)
        src_dir = cfg['src_dir']
        txt_path = cfg['txt_path']
        return src_dir, txt_path
def main():
    src_dir, txt_path = parse_args()
    defect_dir = os.path.join(src_dir, 'defect')
    if (not os.path.exists(defect_dir)):
        os.mkdir(defect_dir)
    num=0
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = 'DJI_'+line.strip()+'.JPG'
            src_filr_path = os.path.join(src_dir, line)
            tar_path = os.path.join(defect_dir,line)
            #移动JPG
            if(os.path.isfile(src_filr_path)):
                os.rename(src_filr_path, tar_path)
                #shutil.copy(src_filr_path, tar_path)
                #移动多光谱
                prefix= line.split('.')[-2][:-1]
                for i in range(1,6):
                    m_name=prefix+str(i)+'.TIF'
                    # print(file)
                    tar_path = os.path.join(defect_dir,m_name)
                    # print(target_dir+tar_file.split("\\")[-1])
                    if os.path.isfile(tar_path):  # 判断目标文件夹是否已存在该文件
                        print("已经存在该文件")
                    else:
                        print("正在移动第{}个文件：{}".format(num + 1, tar_path))
                        num +=1
                        #shutil.copy(os.path.join(src_dir,m_name),tar_path)
                        os.rename(os.path.join(src_dir,m_name),tar_path)
            print(line.strip())



    #for i in range( matrix.)
if __name__ == '__main__':
    main()


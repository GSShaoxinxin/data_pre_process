'''给没有标注内容的图像创建对应的TXT文件'''
import yaml
import os
import glob
def parse_args():
    with open('create_blank_label_txt.yaml', 'r', encoding='utf-8') as y:
        cfg = yaml.full_load(y)
        src_path = cfg['src_path']

        return src_path


def main():
    src_path= parse_args()
    rgb_path_list = glob.glob(os.path.join(src_path, '*.JPG'))
    for rgb_path in rgb_path_list:
        txt_path = os.path.splitext(rgb_path)
        if (not os.path.exists(txt_path)):
            print('create')
if __name__ == '__main__':
    main()
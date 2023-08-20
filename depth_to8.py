import os
import yaml
import cv2
def parse_args():
    with open('depth_to8.yaml', 'r', encoding='utf-8') as y:
        cfg = yaml.full_load(y)
        src_path = cfg['src_path']
        return src_path
def main():
    src_path = parse_args()
    dsc_path = src_path + str(1)
    if(not os.path.exists(dsc_path)):
        os.mkdir(dsc_path)

    f_n = os.listdir(src_path)
    print(f_n)
    for n in f_n:
        imdir = os.path.join(src_path , n)
        print(n)
        img = cv2.imread(imdir)

        cropped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(dsc_path, n), cropped)
    print('finish')
if __name__ == '__main__':
    main()

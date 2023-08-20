# -*- coding:utf8 -*-
#功能实现：将labelme标注后文件转换成异常检测类型文件
#前置要求：将pycharm解释器切到安装有labelme的conda环境下
#         图像和标注信息json文件在一个文件件下,已经修改labelme生成的标签是白色的
#step1 通过labelme 自带的提取函数将内容为标注信息的json文件转成黑白二值图，白色代表异常定位区域
#step2 从step1中生成文件夹中拷贝图像并重命名，整理在同一目录下
import yaml
import os
import shutil

def parse_args():
    with open('prepare_ground_truth.yaml', 'r', encoding='utf-8') as y:
        cfg = yaml.full_load(y)
        json_path = cfg['json_path']
        return json_path


def json2png():
    json_path = parse_args()
    #  获取文件夹内的文件名
    FileNameList = os.listdir(json_path)
    #  激活labelme环境
    os.system("activate labelimg")#在pycharm中执行是要手动切到该环境下
    for i in range(len(FileNameList)):
        #  判断当前文件是否为json文件
        if(os.path.splitext(FileNameList[i])[1] == ".json"):
            json_file = os.path.join(json_path,FileNameList[i])
            #json_file = json_path + "\\" + FileNameList[i]
            #  将该json文件转为png
            os.system("labelme_json_to_dataset " + json_file)


def collectpngs():
    json_path = parse_args()
    #  获取文件夹内的文件名
    FileNameList = os.listdir(json_path)
    dsc_path = os.path.join(json_path,"ground_truth") #存放收集的ground_truth文件夹
    i=1
    tmp_path = dsc_path
    while(os.path.exists(tmp_path)):
        tmp_path=dsc_path+str(i)
        i = i+1
    dsc_path = tmp_path
    os.mkdir(dsc_path)
    NewFileName = 1
    for i in range(len(FileNameList)):
        #  判断当前文件是否为json文件
        if (os.path.splitext(FileNameList[i])[1] == ".json"):

            #  复制label文件
            jpg_file_name = FileNameList[i].split(".", 1)[0]
            label_file = os.path.join(json_path,jpg_file_name+"_json",'label.png')#上一步处理后生成的文件夹下的label.png
            new_label_file =os.path.join(dsc_path ,str(jpg_file_name)+"_mask.png")
            shutil.copyfile(label_file, new_label_file)

            #  文件序列名+1
            NewFileName = NewFileName + 1


def main():
    json2png()
    collectpngs()


if __name__ == '__main__':
    main()
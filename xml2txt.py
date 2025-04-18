import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random
from shutil import copyfile
from PIL import Image

# 只要改下面的CLASSES和PATH就可以了，其他的不用改，这个脚本会自动划分数据集，生成YOLO格式的标签文件【淘宝店铺： 深度学习YOLO数据集】
# 店铺地址：https://d94jo8owngi1qee4sewlr0fjnbvmy1x.taobao.com/?spm=a1z10.1-c-s.0.0.22a4b130DE2kTn

# 分类名称  这里改成数据集的分类名称，一定要改！！！请查看数据集目录下的txt文件
CLASSES = ["cabbage","potato","onion","egg","garlic","carrot","bell_pepper","beet","tomato","eggplant","zucchini","cucumber"]
# 数据集目录 这里改成数据集的根目录，根目录下有两个文件夹Annotations和JPEGImages，一定要改！！！
PATH = "/home/aaronbao/train_dateset_for_vegetables/"
# 训练集占比80% 训练集:验证集=8:2 这里划分数据集 不用改
TRAIN_RATIO = 80


def clear_hidden_files(path):
    dir_list = os.listdir(path)
    for i in dir_list:
        abspath = os.path.join(os.path.abspath(path), i)
        if os.path.isfile(abspath):
            if i.startswith("._"):
                os.remove(abspath)
        else:
            clear_hidden_files(abspath)


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    # Assuming the image format is jpg
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    img = Image.open(image_path)
    w, h = img.size
    in_file = open(PATH+'/Annotations/%s.xml' % image_id, encoding='utf-8')
    out_file = open(PATH+'/YOLOLabels/%s.txt' %
                    image_id, 'w', encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)
    difficult = 0
    for obj in root.iter('object'):
        if obj.find('difficult'):
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in CLASSES or int(difficult) == 1:
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " +
                       " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()


wd = os.getcwd()
wd = os.getcwd()

work_sapce_dir = os.path.join(wd, PATH+"/")

annotation_dir = os.path.join(work_sapce_dir, "Annotations/")
if not os.path.isdir(annotation_dir):
    os.mkdir(annotation_dir)
clear_hidden_files(annotation_dir)
image_dir = os.path.join(work_sapce_dir, "JPEGImages/")
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
clear_hidden_files(image_dir)
yolo_labels_dir = os.path.join(work_sapce_dir, "YOLOLabels/")
if not os.path.isdir(yolo_labels_dir):
    os.mkdir(yolo_labels_dir)
clear_hidden_files(yolo_labels_dir)

yolov5_train_dir = os.path.join(work_sapce_dir, "train/")
if not os.path.isdir(yolov5_train_dir):
    os.mkdir(yolov5_train_dir)
clear_hidden_files(yolov5_train_dir)
yolov5_images_train_dir = os.path.join(yolov5_train_dir, "images/")
if not os.path.isdir(yolov5_images_train_dir):
    os.mkdir(yolov5_images_train_dir)
clear_hidden_files(yolov5_images_train_dir)
yolov5_labels_train_dir = os.path.join(yolov5_train_dir, "labels/")
if not os.path.isdir(yolov5_labels_train_dir):
    os.mkdir(yolov5_labels_train_dir)
clear_hidden_files(yolov5_labels_train_dir)

yolov5_test_dir = os.path.join(work_sapce_dir, "val/")
if not os.path.isdir(yolov5_test_dir):
    os.mkdir(yolov5_test_dir)
clear_hidden_files(yolov5_test_dir)
yolov5_images_test_dir = os.path.join(yolov5_test_dir, "images/")
if not os.path.isdir(yolov5_images_test_dir):
    os.mkdir(yolov5_images_test_dir)
clear_hidden_files(yolov5_images_test_dir)
yolov5_labels_test_dir = os.path.join(yolov5_test_dir, "labels/")
if not os.path.isdir(yolov5_labels_test_dir):
    os.mkdir(yolov5_labels_test_dir)
clear_hidden_files(yolov5_labels_test_dir)


train_file = open(os.path.join(wd, "yolov5_train.txt"), 'w', encoding='utf-8')
test_file = open(os.path.join(wd, "yolov5_valid.txt"), 'w', encoding='utf-8')
train_file.close()
test_file.close()
train_file = open(os.path.join(wd, "yolov5_train.txt"), 'a', encoding='utf-8')
test_file = open(os.path.join(wd, "yolov5_valid.txt"), 'a', encoding='utf-8')
list_imgs = os.listdir(image_dir)  # list image files
prob = random.randint(1, 100)
print("数据集: %d个" % len(list_imgs))
for i in range(0, len(list_imgs)):
    path = os.path.join(image_dir, list_imgs[i])
    if os.path.isfile(path):
        image_path = image_dir + list_imgs[i]
        voc_path = list_imgs[i]
        (nameWithoutExtention, extention) = os.path.splitext(
            os.path.basename(image_path))
        (voc_nameWithoutExtention, voc_extention) = os.path.splitext(
            os.path.basename(voc_path))
        annotation_name = nameWithoutExtention + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_name)
        label_name = nameWithoutExtention + '.txt'
        label_path = os.path.join(yolo_labels_dir, label_name)
    prob = random.randint(1, 100)
    print("Probability: %d" % prob, i, list_imgs[i])
    if (prob < TRAIN_RATIO):
        # train dataset
        if os.path.exists(annotation_path):
            train_file.write(image_path + '\n')
            convert_annotation(nameWithoutExtention)  # convert label
            copyfile(image_path, yolov5_images_train_dir + voc_path)
            copyfile(label_path, yolov5_labels_train_dir + label_name)
    else:
        # test dataset
        if os.path.exists(annotation_path):
            test_file.write(image_path + '\n')
            convert_annotation(nameWithoutExtention)  # convert label
            copyfile(image_path, yolov5_images_test_dir + voc_path)
            copyfile(label_path, yolov5_labels_test_dir + label_name)
train_file.close()
test_file.close()

import logging
import os
import time
import xml.etree.ElementTree as ET
import json
 
def get_file_list(path, type=".xml"):
    file_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext == type:
                file_names.append(filename)
    return file_names
 
 
def xml_to_coco(ann_path, class_names):
    """
    convert xml annotations to coco_api
    :param ann_path:
    :return:
    """
    class_name_dict = dict(zip(class_names, range(1, len(class_names)+1)))
    print("loading annotations into memory...")
    tic = time.time()
    ann_file_names = get_file_list(ann_path, type=".xml")
    logging.info("Found {} annotation files.".format(len(ann_file_names)))
    image_info = []
    categories = []
    annotations = []
    for idx, supercat in enumerate(class_names):
        categories.append(
            {"supercategory": supercat, "id": idx + 1, "name": supercat}
        )
    ann_id = 1
    for idx, xml_name in enumerate(ann_file_names):
        tree = ET.parse(os.path.join(ann_path, xml_name))
        root = tree.getroot()
        file_name = root.find("filename").text
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)
        info = {
            "file_name": file_name,
            "height": height,
            "width": width,
            "id": idx + 1,
        }
        image_info.append(info)
        for _object in root.findall("object"):
            category = _object.find("name").text
            if category not in class_names:
                print(
                    "WARNING! {} is not in class_names! "
                    "Pass this box annotation.".format(category)
                )
                continue
 
            cat_id = class_name_dict[category]
 
            xmin = int(_object.find("bndbox").find("xmin").text)
            ymin = int(_object.find("bndbox").find("ymin").text)
            xmax = int(_object.find("bndbox").find("xmax").text)
            ymax = int(_object.find("bndbox").find("ymax").text)
            w = xmax - xmin
            h = ymax - ymin
            if w < 0 or h < 0:
                print(
                    "WARNING! Find error data in file {}! Box w and "
                    "h should > 0. Pass this box annotation.".format(xml_name)
                )
                continue
            coco_box = [max(xmin, 0), max(ymin, 0), min(w, width), min(h, height)]
            ann = {
                "image_id": idx + 1,  # 此处的图片序号，从1开始计算，该bbox所在的图片序号
                "bbox": coco_box,
                "category_id": cat_id,
                "iscrowd": 0,
                "id": ann_id,  # ann_id指的是bbox的序号
                "area": coco_box[2] * coco_box[3],
            }
            annotations.append(ann)  # annotation是一个list，包含了所有bbox的相关信息
            ann_id += 1
 
    coco_dict = {
        "images": image_info,
        "categories": categories,  # categories，从1开始计算
        "annotations": annotations,
    }
 
    print("Done (t={:0.2f}s)".format(time.time() - tic))
 
    return coco_dict
 
 
xml_folder = "/home/aaronbao/train_dateset_for_vegetables/Annotations"
class_name = ["cabbage","potato","onion","egg","garlic","carrot","bell_pepper","beet","tomato","eggplant","zucchini","cucumber"]
 
coco_dict = xml_to_coco(xml_folder, class_name)
 
"""
保存为json文件
"""
json_file = "/home/aaronbao/train_dateset_for_vegetables/L_All.json"
json_fp = open(json_file, 'w')
json_str = json.dumps(coco_dict)
json_fp.write(json_str)
json_fp.close()
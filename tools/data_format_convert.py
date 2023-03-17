import os
import os.path as osp
import json
import cv2 as cv
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse


def read_txt(filepath):
    print("==========> Read .txt file......")
    with open(filepath, "r", encoding="utf8") as fp:
        data_list = fp.readlines()

    return data_list


def write_json(data, filepath):
    print(f"==========> Write .json file, save to: {filepath}")
    with open(filepath, "w", encoding="utf8") as fp:
        json.dump(data, fp, indent=2)


def image_info(img_path):
    img = cv.imread(img_path)
    img_h, img_w = img.shape[:-1]

    return [img_w, img_h]


def attribute_map(attr_list):
    attr_name_pos_list = ["large_pose", "exaggerate_expression", "extreme_illumination", "make-up", "occlusion", "blur"]
    attr_name_neg_list = ["normal_pose", "normal_expression", "normal_illumination", "no_make-up", "no_occlusion", "clear"]
    new_name_list = []
    for index, pos_name, neg_name in zip(attr_list, attr_name_pos_list, attr_name_neg_list):
        if index == 0:
            new_name_list.append(neg_name)
        else:
            new_name_list.append(pos_name)
    
    return new_name_list


def labelme_formated(data_list, imgshape):
    print(f"==========> The face number of the img: {len(data_list)}")
    shapes = []
    for data in tqdm(data_list):
        for point in data[0]:
            shape = {
                "label": "face",
                "points": point,
                "shape_type": "point"
            }
            shapes.append(shape)
        attribute = attribute_map(data[-1])
        shape = {
                "label": "face",
                "points": data[1],
                "attribute": attribute,
                "shape_type": "rectangle",
        }
        shapes.append()
    labelme = {
        "shapes": shapes,
        "imageWidth": imgshape[0],
        "imageHeight": imgshape[1]
    }
    
    return labelme


def list2dict(txt_list):
    data_info = {}
    print("==========> convert txt list to dict:\n")
    for txt in tqdm(txt_list):
        txt_list = txt.rstrip("\n").split(" ")
        points = np.array(txt_list[0:195]).astype(float).reshape([98, 1, 2]).tolist()
        bbox = np.array(txt_list[195:199]).astype(float).reshape([2, 2]).tolist()
        attribute = np.array(txt_list[199:205]).astype(int).reshape([6, 1]).tolist()
        img_name = txt_list[-1]
        if not img_name in data_info:
            data_info[img_name] = [[points, bbox, attribute]]
        else:
            data_info[img_name].append([points, bbox, attribute])

    return data_info


def txt2labelme(txtfile, img_dir, save_dir):
    """
        # .txt contents
        
        ## format

        coordinates of 98 landmarks (196) + coordinates of upper left corner and lower right corner of detection rectangle (4) + attributes annotations (6) + image name (1)
        x0 y0 ... x97 y97 x_min_rect y_min_rect x_max_rect y_max_rect pose expression illumination make-up occlusion blur image_name


        ## Attached the mappings between attribute names and label values

        1. pose:
            - normal pose->0
            - large pose->1
        2. expression:
            - normal expression->0
            - exaggerate expression->1
        3. illumination:
            - normal illumination->0
            - extreme illumination->1
        4. make-up:
            - no make-up->0
            - make-up->1
        5. occlusion:
            - no occlusion->0
            - occlusion->1
        6. blur:
            - clear->0
            - blur->1
        
        # .json contents

    """
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    txt_list = read_txt(txtfile)
    data_info = list2dict(txt_list)
    for imgname, data in data_info.items():
        img_path = osp.join(img_dir, imgname)
        img_w, img_h = image_info(img_path)
        print(f"==========> Convert: {imgname}")
        labelme_json = labelme_formated(data, imgshape=[img_w, img_h])
        save_path = osp.join(save_dir, imgname.replace(".jpg", ".json").replace(".png", ".json"))
        write_json(labelme_json, save_path)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-f', "--txtfile", default="/dataset/xcyuan/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt", type=str, help="")
    parser.add_argument('-i', "--img_dir", default="/dataset/xcyuan/WFLW/WFLW_images", type=str, help="")
    parser.add_argument('-s', "--save_dir", default="/dataset/xcyuan/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test_json/test", type=str, help="")
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    pass

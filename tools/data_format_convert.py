#-*- coding:utf-8 -*-
import os
import os.path as osp
import shutil
import json
from glob import glob
import cv2 as cv
import numpy as np
from tqdm import tqdm
import argparse


def read_txt(filepath):
    print(f"==========> Read .txt file: {filepath}")
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
    
    new_name_str = ";".join(new_name_list)
    
    return new_name_str


def wflw_labelme_formated(data_list, imgshape):
    print(f"==========> The face number of the img: {len(data_list)}")
    shapes = []
    for data in data_list:
        attribute = attribute_map(data[-1])
        shape_rect = {
                "label": "face",
                "points": data[1],
                "attribute": attribute,
                "shape_type": "rectangle",
        }
        shapes.append(shape_rect)
        shape_points = {
            "cheek": [],
            "left_eye": [],
            "right_eye": [],
            "left_brow": [],
            "right_brow": [],
            "nose": [],
            "outer_mouth": [],
            "inner_mouth": [],
            "right_viewpoint": [],
            "left_viewpoint": [],
        }
        for point_index, point in enumerate(data[0]):
                    # cheek
            if point_index in range(0, 33):
                shape_points["cheek"].append(point)
            # right_brow
            elif point_index in range(33, 42):
                shape_points["right_brow"].append(point)
            # left_brow
            elif point_index in range(42, 51):
                shape_points["left_brow"].append(point)
            # nose
            elif point_index in range(51, 60):
                shape_points["nose"].append(point)
            # right_eye
            elif point_index in range(60, 68):
                shape_points["right_eye"].append(point)
            # left_eye
            elif point_index in range(68, 76):
                shape_points["left_eye"].append(point)
            # outer_mouth
            elif point_index in range(76, 88):
                shape_points["outer_mouth"].append(point)
            # inner_mouth
            elif point_index in range(88, 96):
                shape_points["inner_mouth"].append(point)
            # right_viewpoint
            elif point_index == 96:
                shape_points["right_viewpoint"].append(point)
            # left_viewpoint
            elif point_index == 97:
                shape_points["left_viewpoint"].append(point)
        for attr, points in shape_points.items():
            shapes.append({
                "label": "face",
                "points": points,
                "shape_type": "point",
                "attribute": attr,
            })
    labelme = {
        "shapes": shapes,
        "imageWidth": imgshape[0],
        "imageHeight": imgshape[1],
    }
    
    return labelme


def helen_labelme_formated(data, imgshape):
    shapes = []
    shape_points = {
        "cheek": [],
        "left_eye": [],
        "right_eye": [],
        "left_brow": [],
        "right_brow": [],
        "nose": [],
        "outer_mouth": [],
        "inner_mouth": [],
    }
    for point_index, point in enumerate(data):
        # cheek
        if point_index in range(0, 41):
            shape_points["cheek"].append(point)
        # nose
        elif point_index in range(41, 58):
            shape_points["nose"].append(point)
        # outer_mouth
        elif point_index in range(58, 86):
            shape_points["outer_mouth"].append(point)
        # inner_mouth
        elif point_index in range(86, 114):
            shape_points["inner_mouth"].append(point)
        # left_eye
        elif point_index in range(114, 134):
            shape_points["left_eye"].append(point)
        # right_eye
        elif point_index in range(134, 154):
            shape_points["right_eye"].append(point)
        # left_brow
        elif point_index in range(154, 174):
            shape_points["left_brow"].append(point)
        # right_brow
        elif point_index in range(174, 194):
            shape_points["right_brow"].append(point)
    for attr, points in shape_points.items():
        shapes.append({
            "label": "face",
            "points": points,
            "shape_type": "point",
            "attribute": attr,
        })
    labelme = {
        "imageWidth": imgshape[0],
        "imageHeight": imgshape[1],
        "shapes": shapes,
    }

    return labelme


def wflw_list2dict(txt_list):
    data_info = {}
    print("==========> convert txt list to dict:\n")
    for txt in tqdm(txt_list):
        txt_list = txt.rstrip("\n").split(" ")
        points = np.array(txt_list[0:196]).astype(float).reshape([98, 2]).tolist()
        bbox = np.array(txt_list[196:200]).astype(float).reshape([2, 2]).tolist()
        attribute = np.array(txt_list[200:206]).astype(int).reshape([6, 1]).tolist()
        img_name = txt_list[206]
        if not img_name in data_info:
            data_info[img_name] = [[points, bbox, attribute]]
        else:
            data_info[img_name].append([points, bbox, attribute])

    return data_info


def helen_list2dict(txt_path_list):
    data_info = {}
    for txt_path in txt_path_list:
        txt_content = read_txt(txt_path)
        points = []
        for index, txt_row in enumerate(txt_content):
            if index == 0:
                filename = txt_row.rstrip("\n")+".jpg"
            else:
                point = [float(i) for i in txt_row.rstrip("\n").split(" , ")]
                points.append(point)
        data_info[filename] = points
    
    return data_info


# WFLW 数据集
def wflw_txt2labelme(txt_dir_list, img_dir, save_dir):
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
    if osp.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    txt_list = []
    for txt_path in txt_dir_list:
        txt_list += read_txt(txt_path)
    data_info = wflw_list2dict(txt_list)
    cnt = 0
    for imgname, data in data_info.items():
        cnt += 1
        img_path = osp.join(img_dir, imgname)
        img_w, img_h = image_info(img_path)
        print(f"==========> Convert ({cnt} / {len(data_info)}) : {imgname}")
        labelme_json = wflw_labelme_formated(data, imgshape=[img_w, img_h])
        save_path = osp.join(save_dir, osp.basename(imgname).replace(".jpg", ".json").replace(".png", ".json"))
        write_json(labelme_json, save_path)


# helen 数据集
def helen_txt2labelme(txt_dir_list, img_dir, save_dir):
    if osp.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    txt_path_list = []
    for txt_dir in txt_dir_list:
        txt_path_list += glob(f"{txt_dir}/*.txt")
    data_info = helen_list2dict(txt_path_list)
    cnt = 0
    for imgname, data in data_info.items():
        cnt += 1
        img_path = osp.join(img_dir, imgname)
        img_w, img_h = image_info(img_path)
        print(f"==========> Convert ({cnt} / {len(data_info)}) : {imgname}")
        labelme_json = helen_labelme_formated(data, imgshape=[img_w, img_h])
        save_path = osp.join(save_dir, osp.basename(imgname).replace(".jpg", ".json").replace(".png", ".json"))
        write_json(labelme_json, save_path)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-t', "--txt_list", nargs="+", type=str, help="")
    parser.add_argument('-i', "--img_dir", default="/dataset/xcyuan/WFLW/WFLW_images", type=str, help="")
    parser.add_argument('-s', "--save_dir", default="/dataset/xcyuan/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test_json_other", type=str, help="")
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    opts = get_args()
    
    txt_list = opts.txt_list
    img_dir = opts.img_dir
    save_dir = opts.save_dir

    wflw_txt2labelme(txt_list, img_dir, save_dir)
    # helen_txt2labelme(txt_list, img_dir, save_dir)


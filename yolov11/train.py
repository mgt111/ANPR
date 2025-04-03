import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as xet
import cv2
import shutil
import re
import warnings
import torch
from ultralytics import YOLO

# 忽略警告信息
warnings.filterwarnings('ignore')

def find_the_number(name_of_file):
    """
    从文件名中提取第一个数字序列并返回为整数。
    如果未找到数字，则返回0。

    参数:
    name_of_file (str): 要搜索数字的输入字符串。

    返回:
    int: 在输入字符串中找到的第一个数字序列，如果未找到则返回0。
    """
    match = re.search(r'(\d+)', name_of_file)
    if match:
        return int(match.group(0))
    else:
        return 0

def make_split_folder_in_yolo_format(split_name, split_df):
    """
    为数据集分割（train/val/test）创建YOLO格式的文件夹结构。

    参数:
    split_name (str): 分割的名称（如 'train', 'val', 'test'）。
    split_df (pd.DataFrame): 包含分割数据的DataFrame。

    该函数将在 'datasets/car_license_plate_new/{split_name}' 下创建 'labels' 和 'images' 子目录，
    并以YOLO格式保存相应的标签和图像。
    """
    labels_path = os.path.join('datasets', 'car_license_plate_new', split_name, 'labels')
    images_path = os.path.join('datasets', 'car_license_plate_new', split_name, 'images')

    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    for _, row in split_df.iterrows():
        img_name, img_extension = os.path.splitext(os.path.basename(row['img_path']))

        # 计算YOLO格式的边界框坐标
        x_center = (row['xmin'] + row['xmax']) / 2 / row['img_w']
        y_center = (row['ymin'] + row['ymax']) / 2 / row['img_h']
        width = (row['xmax'] - row['xmin']) / row['img_w']
        height = (row['ymax'] - row['ymin']) / row['img_h']

        # 保存YOLO格式的标签
        label_path = os.path.join(labels_path, f'{img_name}.txt')
        with open(label_path, 'w') as file:
            file.write(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")

        # 将图像复制到images目录
        shutil.copy(row['img_path'], os.path.join(images_path, img_name + img_extension))

    print(f"Created '{images_path}' and '{labels_path}'")

if __name__ == '__main__':
    # 检查CUDA是否可用
    print(f'{torch.cuda.is_available() = }')
    print(f'{torch.cuda.device_count() = }')

    # 数据集路径
    path = "../dataset"

    # 初始化字典以存储标签信息
    labels_dict = dict(
        img_path=[],
        xmin=[],
        xmax=[],
        ymin=[],
        ymax=[],
        img_w=[],
        img_h=[]
    )

    # 获取所有XML文件
    xml_files = glob(f"{path}/annotations/*.xml")
    print(xml_files[:12])

    # 解析XML文件并提取信息
    for file_name in sorted(xml_files, key=find_the_number):
        info = xet.parse(file_name)
        root = info.getroot()
        member_object = root.find('object')
        labels_info = member_object.find('bndbox')
        xmin = int(labels_info.find('xmin').text)
        xmax = int(labels_info.find('xmax').text)
        ymin = int(labels_info.find('ymin').text)
        ymax = int(labels_info.find('ymax').text)

        img_name = root.find('filename').text
        img_path = os.path.join(path, 'images', img_name)

        labels_dict['img_path'].append(img_path)
        labels_dict['xmin'].append(xmin)
        labels_dict['xmax'].append(xmax)
        labels_dict['ymin'].append(ymin)
        labels_dict['ymax'].append(ymax)

        height, width, _ = cv2.imread(img_path).shape
        labels_dict['img_w'].append(width)
        labels_dict['img_h'].append(height)

    # 将字典转换为DataFrame
    alldata = pd.DataFrame(labels_dict)

    # 将数据分割为训练集和测试集
    train, test = train_test_split(alldata, test_size=1 / 10, random_state=42)

    # 将训练集进一步分割为训练集和验证集
    train, val = train_test_split(train, train_size=8 / 9, random_state=42)

    # 打印每个集合的样本数量
    print(f'''
          len(train) = {len(train)}
          len(val) = {len(val)}
          len(test) = {len(test)}
    ''')

    # 如果存在旧的datasets文件夹，则删除
    if os.path.exists('datasets'):
        shutil.rmtree('datasets')

    # 创建YOLO格式的文件夹结构
    make_split_folder_in_yolo_format("train", train)
    make_split_folder_in_yolo_format("val", val)
    make_split_folder_in_yolo_format("test", test)

    # 创建datasets.yaml文件
    datasets_yaml = '''
    path: car_license_plate_new

    train: train/images
    val: val/images
    test: test/images

    # number of classes
    nc: 1

    # class names
    names: ['license_plate']
    '''

    with open('datasets.yaml', 'w') as file:
        file.write(datasets_yaml)

    print(os.getcwd())

    # 加载YOLO模型并开始训练
    model = YOLO('yolo11n.pt')
    model.train(
        data='datasets.yaml',  # 数据集配置文件路径
        epochs=100,  # 训练轮数
        batch=32,  # 批量大小
        device='cuda',  # 使用GPU进行训练
        imgsz=320,  # 训练图像大小（宽度和高度）
        cache=True  # 缓存图像以加快训练速度
    )

    # 找到最近的训练日志目录
    log_dir = max(glob('runs/detect/train*'), key=find_the_number)

    # 从CSV文件加载训练结果
    results = pd.read_csv(os.path.join(log_dir, 'results.csv'))
    results.columns = results.columns.str.strip()  # 去除列名中的前后空格

    # 提取epochs和准确率指标
    epochs = results.index + 1  # Epochs从0开始，所以加1
    mAP_0_5 = results['metrics/mAP50(B)']  # IoU=0.5时的平均精度
    mAP_0_5_0_95 = results['metrics/mAP50-95(B)']  # IoU=0.5:0.95时的平均精度

    # 绘制准确率随epochs的变化
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, mAP_0_5, label='mAP@0.5')
    plt.plot(epochs, mAP_0_5_0_95, label='mAP@0.5:0.95')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
import argparse
from tqdm import tqdm
import shutil
 
def parse_args():
    parser = argparse.ArgumentParser(description='Convert VOC XML to COCO JSON')
    parser.add_argument('--data_dir', default='VOCdevkit/VOC2012', help='VOC数据集根目录')
    parser.add_argument('--output_dir', default='VOCdevkit/VOC2012_coco', help='输出目录')
    return parser.parse_args()
 
def convert_voc_to_coco(data_dir, output_dir):
    """将VOC格式转换为COCO格式"""
    
    # VOC类别（标准20类）
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    
    # 定义类别映射
    categories = []
    for i, class_name in enumerate(VOC_CLASSES, 1):
        categories.append({
            'id': i,
            'name': class_name,
            'supercategory': 'none'
        })
    
    # 处理每个分割集
    for split in ['train', 'val', 'test']:
        print(f"\n处理 {split} 集...")
        
        # 读取对应的文件列表
        list_file = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(list_file):
            print(f"警告：找不到 {list_file}，跳过 {split} 集")
            continue
            
        with open(list_file, 'r') as f:
            file_names = [line.strip() for line in f.readlines()]
        
        # 初始化COCO格式
        coco_data = {
            'info': {
                'description': f'VOC2012 {split} Dataset',
                'version': '1.0',
                'year': 2012,
                'contributor': 'Converted from VOC format',
                'date_created': '2024'
            },
            'licenses': [{
                'id': 1,
                'name': 'Unknown',
                'url': ''
            }],
            'categories': categories,
            'images': [],
            'annotations': []
        }
        
        image_id = 1
        annotation_id = 1
        
        # 处理每个文件
        for file_name in tqdm(file_names, desc=f"转换 {split}"):
            # XML文件路径
            xml_path = os.path.join(data_dir, 'Annotations', split, f"{file_name}.xml")
            
            if not os.path.exists(xml_path):
                print(f"警告：找不到 {xml_path}，跳过")
                continue
                
            # 解析XML
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 获取图片信息
            size_elem = root.find('size')
            if size_elem is None:
                continue
                
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
            
            # 图片信息
            image_info = {
                'id': image_id,
                'file_name': f"{file_name}.jpg",
                'width': width,
                'height': height,
                'license': 1,
                'flickr_url': '',
                'coco_url': '',
                'date_captured': '2024'
            }
            coco_data['images'].append(image_info)
            
            # 处理每个物体标注
            for obj in root.findall('object'):
                name = obj.find('name').text
                
                # 检查是否为VOC标准类别
                if name not in VOC_CLASSES:
                    continue
                    
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue
                    
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # 计算COCO格式的bbox [x, y, width, height]
                bbox = [
                    xmin,
                    ymin,
                    xmax - xmin,
                    ymax - ymin
                ]
                
                # 检查bbox有效性
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                
                # 标注信息
                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': VOC_CLASSES.index(name) + 1,
                    'bbox': bbox,
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 0,
                    'segmentation': []  # VOC没有分割标注
                }
                coco_data['annotations'].append(annotation)
                annotation_id += 1
            
            image_id += 1
        
        # 保存JSON文件
        output_json = os.path.join(output_dir, 'annotations', f'instances_{split}.json')
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存: {output_json}")
        print(f"  图片数: {len(coco_data['images'])}")
        print(f"  标注数: {len(coco_data['annotations'])}")
        
        # 复制图片到新目录（保持目录结构）
        if split in ['train', 'val', 'test']:
            img_src_dir = os.path.join(data_dir, 'JPEGImages', split)
            img_dst_dir = os.path.join(output_dir, split)
            
            if os.path.exists(img_src_dir):
                os.makedirs(img_dst_dir, exist_ok=True)
                
                # 复制图片文件
                for file_name in tqdm(file_names, desc=f"复制 {split} 图片"):
                    img_src = os.path.join(img_src_dir, f"{file_name}.jpg")
                    img_dst = os.path.join(img_dst_dir, f"{file_name}.jpg")
                    
                    if os.path.exists(img_src):
                        shutil.copy2(img_src, img_dst)
                    else:
                        # 尝试其他图片格式
                        for ext in ['.png', '.jpeg', '.bmp']:
                            img_src = os.path.join(img_src_dir, f"{file_name}{ext}")
                            if os.path.exists(img_src):
                                shutil.copy2(img_src, img_dst)
                                break
                
                print(f"已复制图片到: {img_dst_dir}")
    
    # 创建数据集元信息文件
    meta_info = {
        'dataset': 'VOC2012 COCO Format',
        'num_classes': len(VOC_CLASSES),
        'classes': VOC_CLASSES,
        'splits': {
            'train': {
                'json': 'annotations/instances_train.json',
                'images': 'train'
            },
            'val': {
                'json': 'annotations/instances_val.json',
                'images': 'val'
            },
            'test': {
                'json': 'annotations/instances_test.json',
                'images': 'test'
            }
        }
    }
    
    meta_file = os.path.join(output_dir, 'dataset_info.json')
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 转换完成！")
    print(f"输出目录: {output_dir}")
    print(f"目录结构:")
    print(f"{output_dir}/")
    print(f"├── annotations/")
    print(f"│   ├── instances_train.json")
    print(f"│   ├── instances_val.json")
    print(f"│   └── instances_test.json")
    print(f"├── train/          # 训练集图片")
    print(f"├── val/            # 验证集图片")
    print(f"├── test/           # 测试集图片")
    print(f"└── dataset_info.json")
    
    return output_dir
 
def main():
    args = parse_args()
    convert_voc_to_coco(args.data_dir, args.output_dir)
 
if __name__ == '__main__':
    main()
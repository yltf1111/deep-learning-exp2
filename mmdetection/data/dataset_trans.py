import os
import random
import shutil
 
def split_dataset(dataset_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    划分数据集为训练集、验证集、测试集，保持原始目录结构
    """
    
    random.seed(seed)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "比例之和必须为1"
    
    # 根据你的实际目录结构修改路径
    images_dir = os.path.join(dataset_dir, "JPEGImages")
    annotations_dir = os.path.join(dataset_dir, "Annotations")
    
    if not os.path.exists(images_dir):
        print(f"错误：找不到图片目录 {images_dir}")
        return
    if not os.path.exists(annotations_dir):
        print(f"错误：找不到标签目录 {annotations_dir}")
        return
    
    # 获取图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(images_dir):
        file_path = os.path.join(images_dir, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    if len(image_files) == 0:
        print("错误：未找到图片文件！")
        return
    
    # 验证图片和标签对应关系
    valid_pairs = []
    missing_annotations = []
    
    for img_file in image_files:
        img_name, img_ext = os.path.splitext(img_file)
        
        # 支持常见的标签文件格式
        label_extensions = ['.txt', '.json', '.xml', '.csv']
        label_file = None
        
        for ext in label_extensions:
            possible_label = img_name + ext
            label_path = os.path.join(annotations_dir, possible_label)
            if os.path.exists(label_path):
                label_file = possible_label
                break
        
        if label_file:
            valid_pairs.append((img_file, label_file))
        else:
            missing_annotations.append(img_file)
    
    print(f"成功配对 {len(valid_pairs)} 个图片-标签对")
    
    if len(missing_annotations) > 0:
        print(f"警告：{len(missing_annotations)} 个图片没有对应的标签文件")
        if len(missing_annotations) < 20:
            for img in missing_annotations:
                print(f"  - {img}")
    
    if len(valid_pairs) == 0:
        print("错误：未找到任何有效的图片-标签对！")
        return
    
    # 随机打乱并划分
    random.shuffle(valid_pairs)
    total = len(valid_pairs)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = valid_pairs[:train_size]
    val_data = valid_pairs[train_size:train_size + val_size]
    test_data = valid_pairs[train_size + val_size:]
    
    print(f"\n划分完成：")
    print(f"训练集: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"验证集: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"测试集: {len(test_data)} ({len(test_data)/total*100:.1f}%)")
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        split_images_dir = os.path.join(dataset_dir, "JPEGImages", split)
        split_annotations_dir = os.path.join(dataset_dir, "Annotations", split)
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_annotations_dir, exist_ok=True)
    
    # 复制文件
    def copy_split_files(data_list, split_name):
        images_split_dir = os.path.join(dataset_dir, "JPEGImages", split_name)
        annotations_split_dir = os.path.join(dataset_dir, "Annotations", split_name)
        
        for img_file, label_file in data_list:
            shutil.copy2(os.path.join(images_dir, img_file), 
                        os.path.join(images_split_dir, img_file))
            shutil.copy2(os.path.join(annotations_dir, label_file), 
                        os.path.join(annotations_split_dir, label_file))
    
    print("\n正在复制文件...")
    copy_split_files(train_data, 'train')
    copy_split_files(val_data, 'val')
    copy_split_files(test_data, 'test')
    
    # 保存划分信息
    info_file = os.path.join(dataset_dir, "split_info.txt")
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"数据集路径: {dataset_dir}\n")
        f.write(f"划分比例: {train_ratio*100:.0f}% : {val_ratio*100:.0f}% : {test_ratio*100:.0f}%\n")
        f.write(f"随机种子: {seed}\n")
        f.write(f"总样本数: {total}\n")
        f.write(f"训练集: {len(train_data)}\n")
        f.write(f"验证集: {len(val_data)}\n")
        f.write(f"测试集: {len(test_data)}\n")
    
    # 保存文件列表
    for split_name, data_list in [('train', train_data), ('val', val_data), ('test', test_data)]:
        list_file = os.path.join(dataset_dir, f"{split_name}.txt")
        with open(list_file, 'w', encoding='utf-8') as f:
            for img_file, _ in data_list:
                img_name = os.path.splitext(img_file)[0]
                f.write(f"{img_name}\n")
    
    print(f"\n✅ 划分完成！")
    print(f"输出目录结构:")
    print(f"{dataset_dir}/")
    print(f"├── JPEGImages/train/")
    print(f"├── JPEGImages/val/")
    print(f"├── JPEGImages/test/")
    print(f"├── Annotations/train/")
    print(f"├── Annotations/val/")
    print(f"├── Annotations/test/")
    print(f"├── split_info.txt")
    print(f"├── train.txt")
    print(f"├── val.txt")
    print(f"└── test.txt")
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'total': total
    }
 
 
# 主程序
if __name__ == "__main__":
    # 修改这里为你的数据集路径
    dataset_path = "/mnt/c/yltf/深度学习/实验2/mmdetection/data/VOCdevkit/VOC2012"
    
    if not os.path.exists(dataset_path):
        print(f"错误：路径 '{dataset_path}' 不存在！")
        print(f"当前工作目录: {os.getcwd()}")
    else:
        result = split_dataset(
            dataset_path, 
            train_ratio=0.7, 
            val_ratio=0.2, 
            test_ratio=0.1,
            seed=42
        )
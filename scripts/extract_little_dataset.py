import os
import pandas as pd
import shutil
import numpy as np
from collections import Counter

# 读取注释文件
annotations_path = 'data/annotations.csv'
data = pd.read_csv(annotations_path)

# 查看分类数量和分布
categories = data['category'].unique()
categories.sort()
print(f"总共有 {len(categories)} 个不同的分类")
print(f"分类编号范围: {categories.min()} - {categories.max()}")

# 确保目标目录存在，并清空目标目录
target_dir = 'data_little'
images_dir = 'data_little/images'

# 清空目标目录（如果存在）
if os.path.exists(images_dir):
    print(f"清空目标目录: {images_dir}")
    for file in os.listdir(images_dir):
        file_path = os.path.join(images_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    # 创建目录
    os.makedirs(images_dir, exist_ok=True)

# 首先验证文件是否存在，只保留存在的文件的数据
valid_data = []
for _, row in data.iterrows():
    file_path = os.path.join('data/images', row['file_name'])
    if os.path.exists(file_path):
        valid_data.append(row)

valid_data = pd.DataFrame(valid_data)
print(f"原始数据集中有效图片数量: {len(valid_data)}")

# 每个分类至少选择一张图片
selected_files = []
selected_annotations = []

# 为每个分类选择至少一张有效图片
for category in categories:
    # 获取该分类的所有有效样本
    cat_samples = valid_data[valid_data['category'] == category]
    
    # 随机选择一张图片
    if len(cat_samples) > 0:
        selected_sample = cat_samples.sample(1)
        selected_files.append(selected_sample['file_name'].values[0])
        selected_annotations.append(selected_sample)

print(f"已为每个分类选择了一张图片，共 {len(selected_files)} 张")

# 计算还需要多少张图片才能达到128张
remaining_count = 128 - len(selected_files)
print(f"还需要选择 {remaining_count} 张图片")

# 为剩余的图片随机选择（按分类比例分配）
if remaining_count > 0:
    # 计算每个分类的图片数量比例
    category_counts = Counter(valid_data['category'])
    total_valid = len(valid_data)
    category_ratio = {cat: count / total_valid for cat, count in category_counts.items()}
    
    # 按比例分配剩余图片数量
    allocated_counts = {}
    for cat, ratio in category_ratio.items():
        # 按比例分配，至少0张
        allocated_counts[cat] = max(0, int(remaining_count * ratio))
    
    # 确保总数不超过remaining_count
    total_allocated = sum(allocated_counts.values())
    if total_allocated < remaining_count:
        # 有剩余配额，逐个分配给数量最多的类别
        diff = remaining_count - total_allocated
        sorted_cats = sorted(category_ratio.items(), key=lambda x: x[1], reverse=True)
        for i in range(diff):
            cat = sorted_cats[i % len(sorted_cats)][0]
            allocated_counts[cat] += 1
    
    # 随机选择额外的图片
    for category, count in allocated_counts.items():
        if count <= 0:
            continue
            
        # 获取该分类的所有有效样本（排除已选择的）
        cat_samples = valid_data[valid_data['category'] == category]
        cat_samples = cat_samples[~cat_samples['file_name'].isin(selected_files)]
        
        # 如果有足够的样本，随机选择指定数量
        select_count = min(count, len(cat_samples))
        if select_count > 0:
            additional_samples = cat_samples.sample(select_count)
            selected_files.extend(additional_samples['file_name'].values)
            for _, sample in additional_samples.iterrows():
                selected_annotations.append(pd.DataFrame([sample]))

# 确保选择了足够的图片
while len(selected_files) < 128 and len(valid_data) > len(selected_files):
    # 从未选择的有效图片中随机选择
    remaining_samples = valid_data[~valid_data['file_name'].isin(selected_files)]
    if len(remaining_samples) == 0:
        break
        
    # 随机选择额外的图片
    additional_count = min(128 - len(selected_files), len(remaining_samples))
    additional_samples = remaining_samples.sample(additional_count)
    
    selected_files.extend(additional_samples['file_name'].values)
    for _, sample in additional_samples.iterrows():
        selected_annotations.append(pd.DataFrame([sample]))

print(f"最终选择了 {len(selected_files)} 张图片")

# 复制选定的图片到目标目录
for file_name in selected_files:
    src_path = os.path.join('data/images', file_name)
    dst_path = os.path.join(images_dir, file_name)
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"复制文件: {file_name}")
    else:
        print(f"警告: 找不到文件 {src_path}")

# 创建新的注释文件
selected_df = pd.concat(selected_annotations, ignore_index=True)
selected_df.to_csv(os.path.join(target_dir, 'annotations.csv'), index=False)

print("小数据集创建完成！")
print(f"图片保存在: {images_dir}")
print(f"注释文件保存在: {os.path.join(target_dir, 'annotations.csv')}")

# 统计每个分类的图片数量
category_distribution = Counter(selected_df['category'])
print("\n每个分类的图片数量:")
for category in sorted(category_distribution.keys()):
    print(f"分类 {category}: {category_distribution[category]} 张图片")

# 验证文件数量是否匹配
actual_files = os.listdir(images_dir)
print(f"\n实际图片文件数量: {len(actual_files)}")
print(f"选择的图片数量: {len(selected_files)}")
print(f"注释文件中的记录数量: {len(selected_df)}") 
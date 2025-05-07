import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import BaseDataset, create_dataloaders, get_transforms

def test_base_dataset():
    """测试基础数据集加载功能"""
    data_dir = os.path.join('data', 'images')
    anno_file = os.path.join('data', 'annotations.csv')
    
    # 创建基础数据集
    dataset = BaseDataset(data_dir, anno_file)
    
    print(f"数据集大小: {len(dataset)}")
    
    # 获取一个样本
    idx = 0
    image, target = dataset[idx]
    
    print(f"图像大小: {image.size}")
    print(f"边界框: {target['boxes']}")
    print(f"标签: {target['labels']}")
    
    # 可视化图像和边界框
    draw = ImageDraw.Draw(image)
    box = target['boxes'][0].numpy()
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(image))
    plt.title(f"类别: {target['labels'][0].item()}")
    plt.axis("off")
    plt.savefig(os.path.join('tests', 'sample_image.png'))
    plt.close()
    
    print(f"样本图像已保存到 tests/sample_image.png")
    
    return True

def test_data_loaders():
    """测试数据加载器功能"""
    data_dir = os.path.join('data', 'images')
    anno_file = os.path.join('data', 'annotations.csv')
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        anno_file=anno_file,
        batch_size=4
    )
    
    print(f"训练数据批次数: {len(train_loader)}")
    print(f"验证数据批次数: {len(val_loader)}")
    
    # 获取一个批次
    images, targets = next(iter(train_loader))
    
    print(f"批次图像形状: {images.shape}")
    print(f"批次目标数量: {len(targets)}")
    
    # 可视化批次
    batch_size = images.shape[0]
    fig, axs = plt.subplots(1, batch_size, figsize=(15, 5))
    
    # 反归一化函数
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for i in range(batch_size):
        # 反归一化
        img = images[i] * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        if batch_size > 1:
            ax = axs[i]
        else:
            ax = axs
        
        ax.imshow(img)
        ax.set_title(f"类别: {targets[i]['labels'][0].item()}")
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join('tests', 'batch_images.png'))
    plt.close()
    
    print(f"批次图像已保存到 tests/batch_images.png")
    
    return True

if __name__ == "__main__":
    # 创建测试目录
    os.makedirs('tests', exist_ok=True)
    
    # 运行测试
    print("测试基础数据集...")
    test_result_1 = test_base_dataset()
    
    print("\n测试数据加载器...")
    test_result_2 = test_data_loaders()
    
    if test_result_1 and test_result_2:
        print("\n所有测试通过!")
    else:
        print("\n测试失败!") 
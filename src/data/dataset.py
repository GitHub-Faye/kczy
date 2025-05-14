import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Union, Any

class BaseDataset(Dataset):
    """
    基础数据加载类，用于加载图像文件和标注信息
    
    参数:
        data_dir (str): 数据目录路径
        anno_file (str): 标注文件路径
        transform (Optional[transforms.Compose]): 图像转换操作
        target_transform (Optional[transforms.Compose]): 标注转换操作
    """
    
    def __init__(
        self,
        data_dir: str,
        anno_file: str,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.data_dir = data_dir
        self.anno_file = anno_file
        self.transform = transform
        self.target_transform = target_transform
        
        # 加载标注文件
        self.annotations = self._load_annotations()
        
        # 验证数据目录
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录 {self.data_dir} 不存在")
        
        # 检查文件存在性（可选）
        self.verify_files_exist = False
        if self.verify_files_exist:
            self._verify_images_exist()
        
    def _load_annotations(self) -> pd.DataFrame:
        """
        加载标注文件
        
        返回:
            pd.DataFrame: 包含标注信息的DataFrame
        """
        if not os.path.exists(self.anno_file):
            raise FileNotFoundError(f"标注文件 {self.anno_file} 不存在")
        
        # 使用pandas加载CSV文件
        df = pd.read_csv(self.anno_file)
        return df
    
    def _verify_images_exist(self) -> None:
        """
        验证所有图像文件是否存在
        
        注意: 这可能会在数据集较大时花费较长时间
        """
        print(f"正在验证图像文件是否存在...数据集大小: {len(self.annotations)}")
        missing_files = []
        
        for idx in range(len(self.annotations)):
            img_name = self.annotations.iloc[idx, 0]
            img_path = os.path.join(self.data_dir, img_name)
            if not os.path.exists(img_path):
                missing_files.append(img_name)
        
        if missing_files:
            print(f"警告: 发现{len(missing_files)}个缺失的图像文件:")
            for file in missing_files[:10]:
                print(f"  - {file}")
            if len(missing_files) > 10:
                print(f"  ... 还有 {len(missing_files) - 10} 个文件未显示")
    
    def __len__(self) -> int:
        """
        返回数据集大小
        
        返回:
            int: 数据集中样本数量
        """
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        获取指定索引的样本
        
        参数:
            idx (int): 索引
            
        返回:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: 
                包含图像张量和标注信息的元组
        """
        try:
            # 获取指定索引的图像文件名
            img_name = self.annotations.iloc[idx, 0]
            img_path = os.path.join(self.data_dir, img_name)
            
            # 检查图像文件是否存在
            if not os.path.exists(img_path):
                # 提供详细错误信息，包括数据目录和文件名
                error_msg = (
                    f"图像文件 {img_path} 不存在\n"
                    f"数据目录: {self.data_dir}\n"
                    f"图像文件名: {img_name}\n"
                    f"索引: {idx}\n"
                    f"请检查:\n"
                    f"1. 数据目录路径是否正确\n"
                    f"2. 标注文件中的图像文件名是否正确\n"
                    f"3. 图像文件是否已放置在正确位置"
                )
                raise FileNotFoundError(error_msg)
            
            # 加载图像
            image = Image.open(img_path).convert("RGB")
            
            # 获取边界框坐标和类别
            x1 = self.annotations.iloc[idx, 3]
            y1 = self.annotations.iloc[idx, 4]
            x2 = self.annotations.iloc[idx, 5]
            y2 = self.annotations.iloc[idx, 6]
            category = self.annotations.iloc[idx, 7]
            
            # 创建边界框和标签
            bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
            label = torch.tensor(category, dtype=torch.long)
            
            # 创建目标字典
            target = {
                'boxes': bbox.unsqueeze(0),  # 添加批次维度
                'labels': label.unsqueeze(0),  # 添加批次维度
                'image_id': torch.tensor([idx]),
                'area': torch.tensor([(x2 - x1) * (y2 - y1)]),
                'iscrowd': torch.zeros((1,), dtype=torch.int64),
            }
            
            # 应用转换
            if self.transform:
                image = self.transform(image)
            
            if self.target_transform:
                target = self.target_transform(target)
            
            return image, target
            
        except Exception as e:
            # 捕获所有异常并添加更多上下文信息
            if isinstance(e, FileNotFoundError):
                # 已经处理过的文件不存在错误
                raise
            else:
                # 其他类型的错误，添加更多上下文
                error_msg = (
                    f"处理索引 {idx} 的样本时出错:\n"
                    f"原始错误: {str(e)}\n"
                    f"数据目录: {self.data_dir}\n"
                    f"尝试访问的列: {self.annotations.columns.tolist()}\n"
                    f"样本数量: {len(self.annotations)}"
                )
                raise type(e)(error_msg) from e 
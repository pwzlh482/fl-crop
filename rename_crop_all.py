#!/usr/bin/env python3
"""
对 dataset/crop/ 目录下的所有类别图片进行按序重命名
类别: corn, rice, weed, wheat
命名格式: 类别_序号.扩展名 (如 corn_001.jpg, weed_001.jpeg)
"""

import os
import sys
import shutil

def rename_sequential(base_dir):
    """
    对 base_dir 下的每个类别子目录进行按序重命名
    """
    categories = ['corn', 'rice', 'weed', 'wheat']
    supported_ext = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    for category in categories:
        category_dir = os.path.join(base_dir, category)
        if not os.path.exists(category_dir):
            print(f"警告: 目录不存在 {category_dir}")
            continue
        
        # 收集所有支持的图像文件
        files = []
        for f in os.listdir(category_dir):
            if f.lower().endswith(supported_ext):
                files.append(f)
        
        if len(files) == 0:
            print(f"类别 {category}: 没有找到图像文件")
            continue
        
        # 按文件名排序，确保一致顺序
        files.sort()
        
        print(f"类别 {category}: 找到 {len(files)} 个文件")
        
        # 重命名
        for idx, old_name in enumerate(files, start=1):
            # 获取扩展名（保留原大小写）
            ext = os.path.splitext(old_name)[1]
            # 生成新文件名，序号三位数，不足补零
            new_name = f"{category}_{idx:03d}{ext}"
            
            old_path = os.path.join(category_dir, old_name)
            new_path = os.path.join(category_dir, new_name)
            
            # 如果新文件名与旧文件名相同（可能已经是目标格式），跳过
            if old_name == new_name:
                continue
            
            # 如果目标文件已存在（可能因为之前的重命名），需要处理冲突
            if os.path.exists(new_path):
                # 临时重命名为一个临时名称，避免覆盖
                temp_path = os.path.join(category_dir, f"temp_{idx}_{old_name}")
                shutil.move(old_path, temp_path)
                old_path = temp_path
            
            shutil.move(old_path, new_path)
        
        print(f"类别 {category}: 重命名完成")

if __name__ == "__main__":
    # 假设脚本从项目根目录运行
    base_dir = "PFLlib-master/dataset/crop"
    if not os.path.exists(base_dir):
        # 如果在 PFLlib-master 目录外运行，尝试其他路径
        base_dir = "dataset/crop"
        if not os.path.exists(base_dir):
            print("错误: 找不到 dataset/crop 目录")
            print("请确保在项目根目录或 PFLlib-master 目录下运行脚本")
            sys.exit(1)
    
    rename_sequential(base_dir)
    print("所有类别重命名完成！")
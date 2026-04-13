import os
import cv2
import numpy as np

# 配置
RAW_ROOT = "crop_data/raw"  # 原始图片目录
PROCESSED_RAW_ROOT = "crop_data/processed_raw"  # 预处理后图片目录（用于标注）
TARGET_SIZE = (320, 320)  # 目标尺寸改为320*320

# 创建目录
for cls in ["wheat", "corn", "rice"]:
    os.makedirs(os.path.join(PROCESSED_RAW_ROOT, cls), exist_ok=True)

def resize_with_aspect_ratio(img, target_size):
    """
    等比例缩放图片，空白处填充黑色，最终输出target_size尺寸
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    # 计算缩放比例（保持宽高比）
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图片
    resized_img = cv2.resize(img, (new_w, new_h))
    
    # 创建目标尺寸的空白画布（黑色填充）
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # 计算居中放置的偏移量
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # 将缩放后的图片放到画布中央
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    
    return canvas

def batch_preprocess():
    for cls in ["wheat", "corn", "rice"]:
        raw_cls_path = os.path.join(RAW_ROOT, cls)
        processed_cls_path = os.path.join(PROCESSED_RAW_ROOT, cls)
        if not os.path.exists(raw_cls_path):
            print(f"Warning: {raw_cls_path} 不存在，跳过{cls}")
            continue
        
        for img_name in os.listdir(raw_cls_path):
            img_suffix = os.path.splitext(img_name)[1]
            if img_suffix not in (".jpg", ".png", ".jpeg"):
                continue
            # 读取原始图
            img_path = os.path.join(raw_cls_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skip: {img_path} 读取失败")
                continue
            # 预处理：转RGB + 去噪 + 等比例缩放至320*320
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.GaussianBlur(img, (3, 3), 0)  # 去噪
            img = resize_with_aspect_ratio(img, TARGET_SIZE)  # 等比例缩放并填充至320*320
            # 保存到processed_raw目录（用于标注）
            save_path = os.path.join(processed_cls_path, img_name)
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"{cls} 预处理完成，保存至 {processed_cls_path}")

if __name__ == "__main__":
    batch_preprocess()
    print("="*50)
    print("全部预处理完成！")
    print(f"预处理后图片位于：{PROCESSED_RAW_ROOT}")
    print("接下来可以在该目录下进行labelme标注")
    print("="*50)
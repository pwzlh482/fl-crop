import os
import json
import cv2
import numpy as np

# 配置参数
CLASSES = ["wheat", "corn", "rice"]
CLASS2ID = {cls: i for i, cls in enumerate(CLASSES)}
RAW_ROOT = "crop_data/processed_raw"
ANNOT_SAVE_ROOT = "crop_data/annotations"

# 创建保存目录
for cls in CLASSES:
    os.makedirs(os.path.join(ANNOT_SAVE_ROOT, cls), exist_ok=True)

def json2yolo(json_path, img_path):
    """Convert labelme json format to yolo txt format"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    yolo_annot = []

    for shape in data["shapes"]:
        cls_name = shape["label"]
        if cls_name not in CLASS2ID:
            continue
        cls_id = CLASS2ID[cls_name]
        x1, y1 = np.min(shape["points"], axis=0)
        x2, y2 = np.max(shape["points"], axis=0)
        
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        yolo_annot.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return yolo_annot

def batch_convert():
    """Batch convert all classes annotations"""
    for cls in CLASSES:
        raw_cls_path = os.path.join(RAW_ROOT, cls)
        annot_cls_save_path = os.path.join(ANNOT_SAVE_ROOT, cls)
        if not os.path.exists(raw_cls_path):
            print(f"Warning: {raw_cls_path} not exist, skip {cls}")
            continue

        for file_name in os.listdir(raw_cls_path):
            if not file_name.endswith(".json"):
                continue
            json_path = os.path.join(raw_cls_path, file_name)
            img_name = file_name.replace(".json", "")
            img_path = os.path.join(raw_cls_path, img_name + ".jpg")

            yolo_annot = json2yolo(json_path, img_path)
            if yolo_annot is None:
                print(f"Skip: {img_path} read failed")
                continue
            txt_path = os.path.join(annot_cls_save_path, f"{os.path.splitext(img_name)[0]}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_annot))
        print(f"{cls} annotation convert finished, save to {annot_cls_save_path}")
    
    print("="*50)
    print("All annotations convert finished!")
    print("Format: class_id cx cy bw bh (0-1 float)")
    print("="*50)

if __name__ == "__main__":
    batch_convert()

import os
import cv2
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from utils.dataset_utils import separate_data, save_file

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 配置参数
CLASSES = ["wheat", "corn", "rice"]
CLASS2ID = {cls: i for i, cls in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
NUM_CLIENTS = 10
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
ALPHA = 0.5

# 路径配置
RAW_ROOT = "../dataset/crop_data/processed_raw"
ANNOT_ROOT = "crop_data/annotations"
CLIENT_SAVE_ROOT = "crop_data/clients"
CONFIG_PATH = os.path.join(CLIENT_SAVE_ROOT, "crop_config.json")

    
def preprocess_image(img_path):
    """Image preprocessing: denoise + resize + normalize + augmentation"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    #img = cv2.resize(img, IMG_SIZE)
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    return img / 255.0

def load_data():
    """Load preprocessed images, class labels and detection annotations"""
    all_imgs = []
    all_cls_labels = []
    all_det_annots = []

    for cls in CLASSES:
        raw_cls_path = os.path.join(RAW_ROOT, cls)
        annot_cls_path = os.path.join(ANNOT_ROOT, cls)
        if not os.path.exists(raw_cls_path):
            print(f"Warning: {raw_cls_path} not exist, skip {cls}")
            continue

        for img_name in os.listdir(raw_cls_path):
            img_suffix = os.path.splitext(img_name)[1]
            if img_suffix not in (".jpg", ".png", ".jpeg"):
                continue
            img_path = os.path.join(raw_cls_path, img_name)
            img = preprocess_image(img_path)
            if img is None:
                print(f"Skip: {img_path} read failed")
                continue

            cls_id = CLASS2ID[cls]
            annot_txt_name = os.path.splitext(img_name)[0] + ".txt"
            annot_txt_path = os.path.join(annot_cls_path, annot_txt_name)
            annot_content = ""
            if os.path.exists(annot_txt_path):
                with open(annot_txt_path, "r", encoding="utf-8") as f:
                    annot_content = f.read().strip()

            all_imgs.append(img)
            all_cls_labels.append(cls_id)
            all_det_annots.append(annot_content)

    all_imgs = np.array(all_imgs, dtype=np.float32)
    all_cls_labels = np.array(all_cls_labels, dtype=np.int64)
    print(f"Data loaded: {len(all_imgs)} images, {NUM_CLASSES} classes")
    return all_imgs, all_cls_labels, all_det_annots

def generate_dataset():
    """Generate non-iid distributed dataset for federated learning"""
    imgs, cls_labels, det_annots = load_data()
    if len(imgs) == 0:
        raise Exception("No images loaded, check raw data directory")

    # Non-iid data partition
    client_data, statistic = separate_data(
        dataset=(imgs, cls_labels),
        num_clients=NUM_CLIENTS,
        num_classes=NUM_CLASSES,
        partition="noniid",
        alpha=ALPHA
    )
    print(f"Non-iid partition finished: {NUM_CLIENTS} clients, alpha={ALPHA}")

    # Match annotations for each client
    client_annots = []
    idx = 0
    for c in range(NUM_CLIENTS):
        c_img_num = len(client_data[c][0])
        client_annots.append(det_annots[idx:idx+c_img_num])
        idx += c_img_num

    # Split train/val/test set
    train_data, val_data, test_data = [], [], []
    train_annots, val_annots, test_annots = [], [], []
    for c in range(NUM_CLIENTS):
        c_imgs, c_labels = client_data[c]
        c_annots = client_annots[c]

        total = len(c_imgs)  # 新增：获取客户端总数据量

        # ===== 新增：第一步拆分前的判断（核心） =====
        if total < 2:
            # 数据量<2张，全部给训练集，临时集/验证集/测试集为空
            x_train = c_imgs
            y_train = c_labels
            ann_train = c_annots
            x_temp = y_temp = ann_temp = np.array([]) if total <1 else np.array([])
            x_val = y_val = x_test = y_test = np.array([])
            ann_val = ann_test = []
        else:
            # 数据量≥2张，才执行第一步拆分
            x_train, x_temp, y_train, y_temp = train_test_split(
                c_imgs, c_labels, train_size=TRAIN_RATIO, random_state=SEED
            )
            ann_train, ann_temp, _, _ = train_test_split(
                c_annots, c_labels, train_size=TRAIN_RATIO, random_state=SEED
            )
        
            # ===== 图片数据判断 =====
            if len(x_temp) < 2:
                # 临时集<2张，直接赋值，不拆分
                x_val = x_temp
                y_val = y_temp
                x_test = np.array([], dtype=np.float32)  # 指定类型，避免后续报错
                y_test = np.array([], dtype=np.int64)
            else:
                # 临时集≥2张，才执行拆分
                x_val, x_test, y_val, y_test = train_test_split(
                    x_temp, y_temp, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=SEED
                )
        
            # ===== 标注数据判断 =====
            if len(ann_temp) < 2:
                # 临时标注<2条，直接赋值
                ann_val = ann_temp
                ann_test = []
            else:
                # 临时标注≥2条，才执行拆分
                ann_val, ann_test, _, _ = train_test_split(
                    ann_temp, y_temp, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=SEED
                )
            
        train_data.append((x_train, y_train, ann_train))
        val_data.append((x_val, y_val, ann_val))
        test_data.append((x_test, y_test, ann_test))

    # Save dataset
    save_file(
        config_path=CONFIG_PATH,
        train_path=os.path.join(CLIENT_SAVE_ROOT, "train"),
        val_path=os.path.join(CLIENT_SAVE_ROOT, "val"),
        test_path=os.path.join(CLIENT_SAVE_ROOT, "test"),
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        statistic=statistic
    )

    print("="*80)
    print("Dataset generation finished!")
    print(f"Config: {NUM_CLASSES} classes | {NUM_CLIENTS} clients | 7:2:1 train/val/test")
    print(f"Non-iid alpha: {ALPHA}")
    print(f"Save path: {CLIENT_SAVE_ROOT}")
    print("="*80)



if __name__ == "__main__":
    os.makedirs(os.path.join(CLIENT_SAVE_ROOT, "train"), exist_ok=True)
    os.makedirs(os.path.join(CLIENT_SAVE_ROOT, "val"), exist_ok=True)
    os.makedirs(os.path.join(CLIENT_SAVE_ROOT, "test"), exist_ok=True)
    generate_dataset()
    
# ========== 定义cfg配置类==========
class Config:
    # 基础路径配置
    raw_root = "crop_data/raw"
    annot_root = "crop_data/annotations"
    client_save_root = "crop_data/clients"
    data_path = "crop_data/clients"
    config_path = "crop_data/clients/crop_config.json"
    save_dir = "saved_models"
    
    # 联邦学习参数
    data_name = "crop"
    model_name = "resnet18+yolov8n" #注意小写英文字母
    global_rounds = 40  # 全局训练轮数
    local_epochs = 4     # 客户端本地训练轮数
    local_learning_rate = 0.001
    batch_size = 32
    num_clients = 10     # 客户端数量
    device_id = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_gap = 5         # 评估间隔轮数

# 实例化cfg对象（main.py导入的就是这个）
cfg = Config()
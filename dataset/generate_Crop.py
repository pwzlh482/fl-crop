import os
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
from utils.dataset_utils import separate_data, save_file

# 固定随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 配置参数
CLASSES = ["wheat", "corn", "rice"]
CLASS2ID = {cls: i for i, cls in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
NUM_CLIENTS = 20
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
IMG_SIZE = (224, 224)
ALPHA = 0.5

# 路径配置
RAW_ROOT = "crop_data/raw"
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
    img = cv2.resize(img, IMG_SIZE)
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

        x_train, x_temp, y_train, y_temp = train_test_split(
            c_imgs, c_labels, train_size=TRAIN_RATIO, random_state=SEED
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=SEED
        )

        ann_train, ann_temp, _, _ = train_test_split(
            c_annots, c_labels, train_size=TRAIN_RATIO, random_state=SEED
        )
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
    print(f"Image size: {IMG_SIZE} | Non-iid alpha: {ALPHA}")
    print(f"Save path: {CLIENT_SAVE_ROOT}")
    print("="*80)

if __name__ == "__main__":
    os.makedirs(os.path.join(CLIENT_SAVE_ROOT, "train"), exist_ok=True)
    os.makedirs(os.path.join(CLIENT_SAVE_ROOT, "val"), exist_ok=True)
    os.makedirs(os.path.join(CLIENT_SAVE_ROOT, "test"), exist_ok=True)
    generate_dataset()
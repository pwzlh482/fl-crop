import os
import sys
import cv2
import torch
import numpy as np
from config import cfg

# ===================== 核心配置）=====================
CLASSES = ["wheat", "corn", "rice"]  # 必须和main.py训练时的类别顺序一致
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
IMG_SIZE = (320, 320)  # 和main.py训练时的输入尺寸一致
# 模型保存路径：main.py训练后模型的存放目录（根据实际路径调整）
MODEL_SAVE_DIR = cfg.save_dir if hasattr(cfg, 'save_dir') else "./saved_models"
# 待预测图像配置（在predict_images放predict.jpg）
PREDICT_IMG_DIR = "predict_images"
PREDICT_IMG_NAME = "predict.jpg"
PREDICT_IMG_PATH = os.path.join(PREDICT_IMG_DIR, PREDICT_IMG_NAME)
# ======================================================================

# 关键：添加main.py所在路径，确保能导入get_model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from main import get_model  # 复用main.py的模型构建函数
except ImportError:
    raise ImportError("请确保predict.py和main.py在同一目录下，且main.py中有get_model函数")

def create_predict_dir():
    """自动创建待预测图像目录（如果不存在）"""
    if not os.path.exists(PREDICT_IMG_DIR):
        os.makedirs(PREDICT_IMG_DIR)
        print(f"已创建待预测图像目录：{PREDICT_IMG_DIR}")
        print(f"请将需要预测的图像放入该目录，并重命名为：{PREDICT_IMG_NAME}")

def find_latest_model():
    """查找main.py训练生成的最新全局模型（适配.pt/.pth后缀）"""
    # 兼容main.py可能生成的模型文件名：global_model.pt、crop_ServerCrop_model_1.pt、resnet18_round*.pth等
    model_suffixes = (".pt", ".pth")
    model_files = []
    for f in os.listdir(MODEL_SAVE_DIR):
        if f.endswith(model_suffixes) and ("model" in f or "global" in f or "round" in f):
            model_files.append(os.path.join(MODEL_SAVE_DIR, f))
    
    if not model_files:
        raise FileNotFoundError(f"在 {MODEL_SAVE_DIR} 中未找到main.py训练的模型文件！")
    
    # 按文件修改时间排序，取最新的模型
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_model = model_files[0]
    print(f"找到main.py训练的最新模型：{latest_model}")
    return latest_model

def preprocess(img_path):
    """图像预处理（和main.py训练时的逻辑完全一致）"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"未在 {img_path} 找到图像，请检查文件是否存在/格式是否正确")
    
    img_ori = img.copy()  # 保留原始图像用于可视化
    # 预处理：BGR转RGB → 调整尺寸 → 归一化 → 转Tensor → 加batch维度
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  # 归一化到0-1
    img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to(cfg.device)
    return img_ori, img

def has_gui_environment():
    """判断当前环境是否有GUI（避免无显示器时卡窗口）"""
    # 检查DISPLAY环境变量（Linux/Mac），Windows默认有GUI
    if sys.platform == "win32":
        return True
    display = os.environ.get("DISPLAY")
    return display is not None and display != ""

def predict():
    # 1. 检查并创建预测目录
    create_predict_dir()
    
    # 2. 查找最新模型文件
    model_path = find_latest_model()
    
    # 3. 构建和main.py训练时一致的模型结构
    device = cfg.device if hasattr(cfg, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        model_name=cfg.model_name if hasattr(cfg, 'model_name') else "resnet18+yolov8n",
        num_classes=len(CLASSES),
        device=device
    ).to(device)
    
    # 4. 加载模型权重（strict=False兼容融合模型的参数匹配）
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()  # 切换到评估模式
    print("模型加载成功，开始预测...")
    
    # 5. 预处理图像
    img_ori, img = preprocess(PREDICT_IMG_PATH)
    
    # 6. 模型预测（无梯度计算，避免显存占用）
    with torch.no_grad():
        output = model(img)
        pred_prob = torch.softmax(output, dim=1)  # 转换为类别概率
        pred_cls_idx = torch.argmax(pred_prob, dim=1).item()  # 预测类别索引
        pred_cls_name = CLASSES[pred_cls_idx]  # 预测类别名称
        pred_conf = pred_prob[0][pred_cls_idx].item()  # 预测置信度
    
    # 7. 在原始图像上绘制预测结果
    label_text = f"Pred: {pred_cls_name} (Conf: {pred_conf:.2f})"
    cv2.putText(
        img_ori,
        label_text,
        (10, 30),  # 文字位置（左上角）
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,        # 字体大小
        COLORS[pred_cls_idx],  # 对应类别颜色
        2           # 文字粗细
    )
    
    # 8. 保存预测结果
    result_filename = f"predict_result_{os.path.splitext(PREDICT_IMG_NAME)[0]}.jpg"
    cv2.imwrite(result_filename, img_ori)
    print(f"\n预测结果：")
    print(f"  预测类别：{pred_cls_name}")
    print(f"  置信度：{pred_conf:.4f}")
    print(f"  各类别概率：")
    for idx, cls in enumerate(CLASSES):
        print(f"      {cls}：{pred_prob[0][idx].item():.4f}")
    print(f"\n预测结果已保存为：{os.path.abspath(result_filename)}")
    
    # 9. 显示结果（仅在有GUI环境时，且设置超时自动关闭）
    try:
        if has_gui_environment():
            cv2.imshow("Crop Recognition Result", img_ori)
            # 设置5秒超时（5000ms），超时后自动关闭窗口，避免程序挂起
            cv2.waitKey(5000)
            # 强制销毁所有窗口，释放资源
            #cv2.waitKey(1)
            #cv2.destroyAllWindows()
 
            
            
        else:
            print("无GUI环境，已跳过图像显示（请手动打开保存的result文件）")
    except Exception as e:
        print(f"图像显示失败：{str(e)}（请手动打开保存的{result_filename}文件）")
        # 确保即使显示出错，也销毁窗口，避免资源残留
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    # 直接运行即可，无需依赖main.py的训练逻辑
    try:
        predict()
        print("\n预测已结束，程序退出")
    except Exception as e:
        print(f"\n预测出错：{str(e)}")
        # 异常时也确保销毁所有OpenCV窗口
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        sys.exit(1)  # 异常退出，返回非0状态码
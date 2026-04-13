import h5py
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mtick
import math

# 1. 设置文件路径
file_path = "../results/MNIST_ResNet_FedProx_test__acc0.9777.h5"

if not os.path.exists(file_path):
    print(f"找不到文件: {file_path}")
else:
    # 自动从文件名提取信息
    base_name = os.path.basename(file_path)
    parts = base_name.split('_')
    dataset_name = parts[0] if len(parts) > 0 else "Dataset"
    algo_name = parts[1] if len(parts) > 1 else "Algorithm"
    model_name = "ResNet18" if parts[2] == "test" else parts[2]

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = hf['rs_test_acc'][:]
        rs_train_loss = hf['rs_train_loss'][:]

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=120)
    plt.title(f"{model_name} Training Performance ({dataset_name} - {algo_name}18)", 
              fontsize=14, pad=45, fontweight='bold')
    
    eval_gap = 1
    global_rounds = [i * eval_gap for i in range(len(rs_test_acc))]
    last_round = global_rounds[-1]
    
    # --- 左轴：Loss (红虚线 + 圆点) ---
    line1, = ax1.plot(global_rounds, rs_train_loss, color='#d62728', 
                     label='Train Loss', linewidth=1.5, linestyle='--', 
                     marker='o', markersize=4, alpha=0.9)
    
    # 【末尾标注：在点正上方】
    last_loss = rs_train_loss[-1]
    ax1.annotate(f'{last_loss:.4f}', xy=(last_round, last_loss), xytext=(0, 10),
                 textcoords="offset points", color='#d62728', fontweight='bold', 
                 va='bottom', ha='center')

    # 标签置顶对齐左轴
    ax1.text(0, 1.02, 'Loss', transform=ax1.transAxes, color='#d62728', 
             fontsize=11, fontweight='bold', va='bottom', ha='left')
    
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)

    # --- 右轴：Accuracy (蓝实线 + 方块) ---
    ax2 = ax1.twinx()
    line2, = ax2.plot(global_rounds, rs_test_acc, color='#1f77b4', 
                     label='Test Accuracy', linewidth=1.5, 
                     marker='s', markersize=5)
    
    # 【末尾标注：在点正下方】
    last_acc = rs_test_acc[-1]
    ax2.annotate(f'{last_acc*100:.2f}%', xy=(last_round, last_acc), xytext=(0, -15),
                 textcoords="offset points", color='#1f77b4', fontweight='bold', 
                 va='top', ha='center')

    # 标签置顶对齐右轴
    ax2.text(1, 1.02, 'Accuracy', transform=ax2.transAxes, color='#1f77b4', 
             fontsize=11, fontweight='bold', va='bottom', ha='right')
    
    ax2.tick_params(axis='y', labelcolor='#1f77b4')

    # --- y 轴范围动态计算 ---
    # Accuracy 强制从 0 开始
    acc_max = max(rs_test_acc)
    acc_upper = min(1.0, math.ceil(acc_max * 20) / 20 + 0.05) 
    ax2.set_ylim(0, acc_upper) 
    
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

    # --- 横轴与整体布局 ---
    ax1.set_xlabel('Global Training Rounds(Local Epochs=1)', fontsize=12)
    # 左右留微小余量
    ax1.set_xlim(global_rounds[0] - 2, last_round + 5)

    # 合并图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05), 
               ncol=2, frameon=True, fontsize=10, shadow=True)

    plt.tight_layout()
    # 自动命名保存
    save_name = f"{dataset_name}_{model_name}_final_standard.png"
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    print(f"--- 图像生成成功: {save_name} ---")
    plt.show()

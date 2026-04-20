#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习客户端协同训练平台 (V7 - 灵动渐变与逻辑优化版)
- 修复：采用 QFrame 底板彻底解决 Windows 下背景全白的问题，实现黄绿渐变
- 优化：整体色彩提亮（翠绿+亮黄），增加灵动感
- 优化：全局轮次与 μ值 位置互换，且算法选择 FedAvg 时自动禁用 μ值
- 优化：引入底部状态栏、帮助按钮，规范文字描述与按钮图标
"""

import sys
import os
import subprocess
import platform
import signal
import re
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QProgressBar, QGroupBox,
    QComboBox, QLineEdit, QFormLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QFileDialog, QAbstractItemView, QInputDialog,
    QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QPixmap

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# ============================================================
# 全局样式常量 — 灵动西农定制风格
# ============================================================
FONT_SIZE_LABEL = 13       
FONT_SIZE_INPUT = 13       
FONT_SIZE_GROUP = 14       
FONT_SIZE_BTN = 15         
FONT_SIZE_LOG = 13         

# 色彩提亮，显得更轻盈灵动
NWAFU_GREEN = "#2A7240"  # 清新翠绿
NWAFU_YELLOW = "#DCA728" # 明亮金黄

GLOBAL_STYLE = f"""
/* 确保所有的背景由底板控制，不再受系统主题干扰 */
QMainWindow {{ background-color: white; }}
QWidget#central_widget {{ background: transparent; }}

/* 核心底板：完美的黄绿渐变 */
QFrame#bg_frame {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #E8F5E9, stop:1 #FFFDE7);
}}

QGroupBox {{
    font-weight: bold; font-size: {FONT_SIZE_GROUP}px;
    border: 1px solid rgba(42, 114, 64, 0.3);
    border-radius: 8px;
    margin-top: 16px; padding-top: 15px; 
    background-color: rgba(255, 255, 255, 0.65); /* 半透明白底，透出底板渐变 */
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 15px;
    padding: 0 8px; color: {NWAFU_GREEN};
}}

/* 按钮通用样式 */
QPushButton {{
    background-color: {NWAFU_GREEN}; color: white;
    border: none; padding: 10px 24px;
    border-radius: 6px; font-weight: bold;
    font-size: {FONT_SIZE_BTN}px; min-height: 25px;
}}
QPushButton:hover {{ background-color: #215c32; }}
QPushButton:disabled {{ background-color: #a0b2a6; color: #f0f0f0; }}

/* 特定按钮样式 */
QPushButton#stop_btn {{ background-color: #8B0000; }}
QPushButton#stop_btn:hover {{ background-color: #A52A2A; }}

QPushButton#clear_btn {{ background-color: #7f8c8d; }}
QPushButton#clear_btn:hover {{ background-color: #95a5a6; }}

QPushButton#help_btn {{ background-color: #3498db; }}
QPushButton#help_btn:hover {{ background-color: #2980b9; }}

QPushButton#result_btn {{
    background-color: {NWAFU_YELLOW}; color: white;
    padding: 12px 30px; border-radius: 6px;
    font-size: {FONT_SIZE_BTN}px; font-weight: 900;
}}
QPushButton#result_btn:hover {{ background-color: #e3b642; }}

/* 下拉框样式 */
QComboBox {{
    padding: 5px 10px; border: 1px solid #A0B2A6;
    border-radius: 4px; background-color: white;
    font-size: {FONT_SIZE_INPUT}px; min-width: 130px;
    color: #2c3e50;
}}
QComboBox:hover {{ border: 1px solid {NWAFU_YELLOW}; }}
QComboBox:disabled {{ background-color: #e9ecef; color: #6c757d; }}
QComboBox QAbstractItemView {{
    background-color: white;
    selection-background-color: #e8f5e9;
    selection-color: {NWAFU_GREEN};
}}

QLineEdit {{
    padding: 5px 10px; border: 1px solid #A0B2A6;
    border-radius: 4px; background-color: white;
    font-size: {FONT_SIZE_INPUT}px; min-width: 90px;
}}
QLineEdit:focus {{ border: 1px solid {NWAFU_GREEN}; }}

QLabel {{ color: #2c3e50; font-size: {FONT_SIZE_LABEL}px; background: transparent; }}

/* 进度条 */
QProgressBar {{
    border: 1px solid rgba(42, 114, 64, 0.4);
    border-radius: 6px; text-align: center;
    background-color: rgba(255,255,255,0.8); color: #154326; font-weight: bold;
    height: 20px;
}}
QProgressBar::chunk {{
    background-color: {NWAFU_GREEN};
    border-radius: 4px;
}}

/* 表格头 */
QHeaderView::section {{
    background-color: {NWAFU_GREEN}; color: white;
    font-weight: bold; padding: 8px; border: 1px solid #215c32;
}}
QTableWidget {{ gridline-color: #d0d0d0; background: white; }}
"""


class TrainingThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, args, cwd):
        super().__init__()
        self.args = args
        self.cwd = cwd
        self.process = None
        self.running = True
        self.exit_code = -1

    def run(self):
        try:
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            kwargs = {
                'cwd': self.cwd, 'stdout': subprocess.PIPE, 'stderr': subprocess.STDOUT,
                'text': True, 'encoding': 'utf-8', 'errors': 'replace', 'bufsize': 1, 'env': env
            }
            if platform.system() == 'Windows':
                kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs['preexec_fn'] = os.setsid

            self.process = subprocess.Popen(self.args, **kwargs)
            for line in iter(self.process.stdout.readline, ''):
                if not self.running: break
                self.log_signal.emit(line.rstrip())
            self.process.wait()
            self.exit_code = self.process.returncode
        except Exception as e:
            self.log_signal.emit(f"[系统错误] {str(e)}")
            self.exit_code = 1
        finally:
            self.finished_signal.emit(self.exit_code)

    def stop(self):
        self.running = False
        if self.process:
            try:
                if platform.system() == 'Windows':
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.process.pid)], creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except Exception:
                self.process.kill()


# ============================================================
# 结果窗口
# ============================================================
class ResultWindow(QMainWindow):
    def __init__(self, rounds_data, acc_data, loss_data, total_rounds,
                 config_info, auc_info=None, parent=None):
        super().__init__(parent)
        self.rounds_data, self.acc_data, self.loss_data = rounds_data, acc_data, loss_data
        self.total_rounds, self.config_info, self.auc_info = total_rounds, config_info, auc_info or {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("训练结果分析 - 联邦学习平台")
        self.setGeometry(150, 80, 1100, 750)
        self.setStyleSheet(GLOBAL_STYLE)
        
        central = QWidget()
        central.setObjectName("central_widget")
        self.setCentralWidget(central)
        c_layout = QVBoxLayout(central)
        c_layout.setContentsMargins(0, 0, 0, 0)

        # 同样使用 bg_frame 确保背景渐变生效
        bg_frame = QFrame()
        bg_frame.setObjectName("bg_frame")
        c_layout.addWidget(bg_frame)

        main_layout = QVBoxLayout(bg_frame)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("训练结果综合评估")
        title.setStyleSheet(f"color: {NWAFU_GREEN}; font-size: 28px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        chart_row = QHBoxLayout()
        fig_group = QGroupBox("训练曲线")
        fig_layout = QVBoxLayout()
        self.fig, self.ax = plt.subplots(figsize=(7, 4), dpi=120)
        self.canvas = FigureCanvas(self.fig)

        self.ax.set_xlabel('Global Rounds', fontsize=11, labelpad=8)
        self.ax.set_ylabel('Accuracy (%)', fontsize=11, color='#2A7240')
        self.ax.grid(True, alpha=0.3)
        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel('Loss', fontsize=11, color='#DCA728')

        if self.rounds_data:
            self.ax.plot(self.rounds_data, self.acc_data, 'o-', color='#2A7240', label='Test Acc', linewidth=2, markersize=5)
            self.ax2.plot(self.rounds_data, self.loss_data, 's-', color='#DCA728', label='Train Loss', linewidth=2, markersize=5)
            total = max(self.total_rounds, max(self.rounds_data) + 1)
            self.ax.set_xlim(0, total)
            self.ax.set_ylim(0, max(100, max(self.acc_data) + 5) if self.acc_data else 100)
            self.ax2.set_ylim(0, max(5, max(self.loss_data) + 0.5) if self.loss_data else 5)
        else:
            self.ax.set_xlim(0, self.total_rounds)
            self.ax.set_ylim(0, 100)
            self.ax2.set_ylim(0, 5)

        self.ax.legend(loc='upper left', fontsize=10)
        self.ax2.legend(loc='upper right', fontsize=10)
        self.fig.tight_layout(pad=2.0)
        fig_layout.addWidget(self.canvas)
        fig_group.setLayout(fig_layout)
        chart_row.addWidget(fig_group, 3)

        stats_group = QGroupBox("性能指标")
        stats_layout = QVBoxLayout()
        self.stats_table = QTableWidget(0, 2)
        self.stats_table.setHorizontalHeaderLabels(["指标", "值"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._fill_stats_table()
        stats_layout.addWidget(self.stats_table)
        stats_group.setLayout(stats_layout)
        chart_row.addWidget(stats_group, 2)

        main_layout.addLayout(chart_row, stretch=4)

        config_group = QGroupBox("训练配置")
        cfg_layout = QVBoxLayout()
        config_text = QTextEdit()
        config_text.setReadOnly(True)
        config_text.setMaximumHeight(100)
        lines = [f"• {k}: {v}" for k, v in self.config_info.items()]
        config_text.setText('\n'.join(lines))
        config_text.setStyleSheet(f"background-color: white; color: {NWAFU_GREEN}; font-size: {FONT_SIZE_LABEL}px; border-radius: 4px; padding: 10px;")
        cfg_layout.addWidget(config_text)
        config_group.setLayout(cfg_layout)
        main_layout.addWidget(config_group, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("导出报告图片")
        save_btn.clicked.connect(self.export_figure)
        btn_row.addWidget(save_btn)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        btn_row.addStretch()
        main_layout.addLayout(btn_row)

    def _fill_stats_table(self):
        rows = []
        if self.acc_data:
            rows.append(("最终准确率", f"{self.acc_data[-1]:.2f}%"))
            best_idx = np.argmax(self.acc_data)
            rows.append(("最佳准确率", f"{self.acc_data[best_idx]:.2f}% (R{self.rounds_data[best_idx]})"))
        if self.auc_info:
            if 'avg_auc' in self.auc_info: rows.append(("平均测试 AUC", f"{self.auc_info['avg_auc']:.4f}"))
            if 'std_acc' in self.auc_info: rows.append(("Accuracy标准差", f"{self.auc_info['std_acc']:.4f}"))
            if 'std_auc' in self.auc_info: rows.append(("AUC标准差", f"{self.auc_info['std_auc']:.4f}"))
        if self.loss_data: rows.append(("最终损失", f"{self.loss_data[-1]:.4f}"))
        rows.extend([("已训练轮次", str(len(self.rounds_data))), ("设定总轮次", str(self.total_rounds))])

        self.stats_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.stats_table.setItem(i, 0, QTableWidgetItem(k))
            item = QTableWidgetItem(v)
            if "最佳" in k or "AUC" in k: item.setForeground(Qt.darkGreen)
            self.stats_table.setItem(i, 1, item)

    def export_figure(self):
        default_path = os.path.join(os.path.expanduser("~"), "Desktop", f"training_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        path, _ = QFileDialog.getSaveFileName(self, "保存图片", default_path, "PNG (*.png)")
        if path:
            self.fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
            QMessageBox.information(self, "成功", f"已保存至:\n{path}")


# ============================================================
# 分类窗口
# ============================================================
class ClassifyWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.loaded_model, self.loaded_dataset, self.class_names, self.image_path = None, None, None, None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("农作物智能分类系统")
        self.setGeometry(180, 100, 900, 650)
        self.setStyleSheet(GLOBAL_STYLE)

        central = QWidget()
        central.setObjectName("central_widget")
        self.setCentralWidget(central)
        c_layout = QVBoxLayout(central)
        c_layout.setContentsMargins(0, 0, 0, 0)

        bg_frame = QFrame()
        bg_frame.setObjectName("bg_frame")
        c_layout.addWidget(bg_frame)

        main_layout = QVBoxLayout(bg_frame)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("农作物智能分类与预测系统")
        title.setStyleSheet(f"color: {NWAFU_GREEN}; font-size: 28px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        top_bar = QGroupBox("1. 配置与上传")
        top_bar_layout = QHBoxLayout()
        load_model_btn = QPushButton("加载模型 (.pt)")
        load_model_btn.clicked.connect(self.on_load_model)
        top_bar_layout.addWidget(load_model_btn)
        
        self.model_status = QLabel("模型: 未加载")
        self.model_status.setStyleSheet("color: #8B0000; font-weight: bold;")
        top_bar_layout.addWidget(self.model_status)
        top_bar_layout.addSpacing(30)
        
        upload_btn = QPushButton("选择待测图片")
        upload_btn.clicked.connect(self.on_select_image)
        top_bar_layout.addWidget(upload_btn)
        
        self.image_status = QLabel("图片: 未选择")
        self.image_status.setStyleSheet("color: #8B0000; font-weight: bold;")
        top_bar_layout.addWidget(self.image_status)
        
        top_bar_layout.addStretch()
        top_bar.setLayout(top_bar_layout)
        main_layout.addWidget(top_bar)

        content_layout = QHBoxLayout()
        img_group = QGroupBox("2. 图像预览")
        img_layout = QVBoxLayout()
        self.result_img_label = QLabel("暂无图片")
        self.result_img_label.setAlignment(Qt.AlignCenter)
        self.result_img_label.setMinimumSize(400, 400)
        self.result_img_label.setStyleSheet("background-color: white; border: 1px dashed #A0B2A6; border-radius: 8px;")
        img_layout.addWidget(self.result_img_label)
        img_group.setLayout(img_layout)
        content_layout.addWidget(img_group, 1)

        res_group = QGroupBox("3. 分类推理与结果")
        res_layout = QVBoxLayout()
        classify_btn = QPushButton("开始预测分类")
        classify_btn.setObjectName("result_btn")
        classify_btn.clicked.connect(self.on_classify)
        res_layout.addWidget(classify_btn)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet(f"""
            background-color: white; border: 2px solid {NWAFU_YELLOW}; 
            border-radius: 6px; font-size: {FONT_SIZE_LABEL + 2}px; 
            color: {NWAFU_GREEN}; padding: 10px;
        """)
        self.result_text.setPlaceholderText("预测结果将在此处详细展示...")
        res_layout.addWidget(self.result_text)
        
        res_group.setLayout(res_layout)
        content_layout.addWidget(res_group, 1)
        main_layout.addLayout(content_layout, stretch=1)

    def on_select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.image_path = path
            pixmap = QPixmap(path).scaled(self.result_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.result_img_label.setPixmap(pixmap)
            self.image_status.setText(f"已选: {os.path.basename(path)}")
            self.image_status.setStyleSheet(f"color: {NWAFU_GREEN}; font-weight: bold;")
            self.result_text.setPlainText("图片已就绪，等待预测...")

    def on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PyTorch (*.pt *.pth)")
        if path:
            try:
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from model_predict import parse_model_info_from_path, load_pytorch_model
                dataset, model_str, acc = parse_model_info_from_path(path)
                if not dataset: dataset, _ = QInputDialog.getText(self, "输入", "请输入数据集名称 (如 MNIST, Cifar10):")
                if not model_str: model_str, _ = QInputDialog.getText(self, "输入", "请输入模型 (如 MobileNet):")
                
                num_classes = 10 if dataset in ['Cifar10', 'MNIST', 'Digit5'] else 6
                self.loaded_model = load_pytorch_model(path, model_str, dataset, num_classes, self.device)
                self.loaded_model_str, self.loaded_dataset = model_str, dataset
                self.class_names = ['airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] if dataset == 'Cifar10' else [str(i) for i in range(10)]
                
                self.model_status.setText(f"已加载: {model_str} ({dataset})")
                self.model_status.setStyleSheet(f"color: {NWAFU_GREEN}; font-weight: bold;")
                self.result_text.setPlainText(f"模型加载成功！\n\n网络: {model_str}\n数据集: {dataset}\n类型数: {num_classes}\n\n请点击上方开始预测。")
            except Exception as e:
                QMessageBox.critical(self, "加载失败", str(e))

    def on_classify(self):
        if not self.loaded_model or not self.image_path:
            return QMessageBox.warning(self, "警告", "模型或图片尚未加载！")
        try:
            image = Image.open(self.image_path).convert('L' if self.loaded_dataset == 'MNIST' else 'RGB')
            size, mean, std = ((28,28), (0.5,), (0.5,)) if self.loaded_dataset == 'MNIST' else ((32,32), (0.5,0.5,0.5), (0.5,0.5,0.5))
            transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean, std)])
            
            with torch.no_grad():
                out = self.loaded_model(transform(image).unsqueeze(0).to(self.device))
                prob = torch.nn.functional.softmax(out, dim=1)
                pred = torch.argmax(prob, dim=1).item()
                conf = prob[0][pred].item()
            
            cname = self.class_names[pred] if self.class_names else str(pred)
            self.result_text.setPlainText(
                f"预测完成！\n\n"
                f"识别结果:  {cname}  (ID: {pred})\n"
                f"置信度:  {conf:.2%}\n"
                f"-----------------------------\n"
                f"文件: {os.path.basename(self.image_path)}\n"
                f"模型: {self.loaded_model_str}\n"
                f"数据集: {self.loaded_dataset}"
            )
        except Exception as e:
            QMessageBox.critical(self, "分类失败", str(e))


# ============================================================
# 主窗口
# ============================================================
class FedProxApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.training_thread, self.result_win, self.classify_win = None, None, None
        self.current_round, self.total_rounds, self.current_acc, self.current_loss = 0, 100, 0.0, 0.0
        self.training_failed = False
        self.collected_rounds, self.collected_accs, self.collected_losses = [], [], []
        self.auc_info = {}

        self.init_ui()
        self._detect_device()

    def init_ui(self):
        self.setWindowTitle("联邦学习客户端协同训练平台")
        self.setGeometry(100, 100, 1200, 780)
        self.setStyleSheet(GLOBAL_STYLE)

        # 核心修改：利用独立 QFrame 绑定渐变背景，解决白底 Bug
        central = QWidget()
        central.setObjectName("central_widget")
        self.setCentralWidget(central)
        c_layout = QVBoxLayout(central)
        c_layout.setContentsMargins(0, 0, 0, 0)

        bg_frame = QFrame()
        bg_frame.setObjectName("bg_frame")
        c_layout.addWidget(bg_frame)

        main = QVBoxLayout(bg_frame)
        main.setSpacing(12)
        main.setContentsMargins(25, 20, 25, 20)

        # ===== 标题与校徽 =====
        title_layout = QHBoxLayout()
        title_layout.addStretch()
        self.logo_label = QLabel()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logo_paths = [os.path.join(script_dir, "nwafu_logo.png"), os.path.join(script_dir, "image_e9656f.png")]
        
        logo_loaded = False
        for path in logo_paths:
            if os.path.exists(path):
                pixmap = QPixmap(path).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.logo_label.setPixmap(pixmap)
                logo_loaded = True
                break
        if not logo_loaded:
            self.logo_label.setText("") # 不显示冗余文字
        
        title_layout.addWidget(self.logo_label)
        title_layout.addSpacing(15)

        # 主标题使用明确大字号
        title = QLabel("联邦学习客户端协同训练平台")
        title.setStyleSheet(f"color: {NWAFU_GREEN}; font-size: 32px; font-weight: bold; background: transparent;")
        title_layout.addWidget(title)
        title_layout.addStretch()
        main.addLayout(title_layout)

        # ===== 配置区 (逻辑调换) =====
        cfg_grp = QGroupBox("训练参数配置")
        cfg_lyt = QVBoxLayout()
        row1 = QHBoxLayout()
        
        left_form = QFormLayout()
        left_form.setSpacing(12)
        left_form.setLabelAlignment(Qt.AlignRight)

        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["FedProxV2", "FedProx", "FedAvg"])
        self.algo_combo.currentTextChanged.connect(self.on_algo_changed) # 联动机制
        left_form.addRow("聚合算法:", self.algo_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["MobileNet", "ResNet18", "CNN", "DNN"])
        left_form.addRow("网络模型:", self.model_combo)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Cifar10", "MNIST"])
        left_form.addRow("目标数据:", self.dataset_combo)

        row1.addLayout(left_form, 1)
        row1.addSpacing(40)

        right_form = QFormLayout()
        right_form.setSpacing(12)
        right_form.setLabelAlignment(Qt.AlignRight)

        self.clients_combo = QComboBox()
        self.clients_combo.setEditable(True)
        self.clients_combo.addItems(["1", "2", "5", "10"])
        self.clients_combo.setCurrentText("2")
        right_form.addRow("客户端数:", self.clients_combo)

        self.lr_combo = QComboBox()
        self.lr_combo.setEditable(True)
        self.lr_combo.addItems(["0.01", "0.05", "0.1", "0.001"])
        self.lr_combo.setCurrentText("0.05")
        right_form.addRow("本地学习率:", self.lr_combo)

        # [逻辑优化] 将全局轮次放到核心右侧区
        self.rounds_input = QLineEdit("100")
        self.rounds_input.setMinimumWidth(130)
        right_form.addRow("全局轮次:", self.rounds_input)

        row1.addLayout(right_form, 1)
        cfg_lyt.addLayout(row1)

        # 第二行辅助参数
        row2 = QHBoxLayout()
        row2.setSpacing(15)
        
        # [逻辑优化] 将 u 值下放，并加上名称
        row2.addWidget(QLabel("FedProx μ值:"))
        self.mu_combo = QComboBox()
        self.mu_combo.setEditable(True)
        self.mu_combo.addItems(["0", "0.01", "0.1", "1.0"])
        self.mu_combo.setCurrentText("0.1")
        self.mu_combo.setMinimumWidth(70)
        row2.addWidget(self.mu_combo)

        row2.addWidget(QLabel("运行设备:"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(80)
        row2.addWidget(self.device_combo)

        for label, widget, width, d_val in [
            ("Batch Size:", QLineEdit(), 60, "64"), 
            ("本地轮次:", QLineEdit(), 60, "3"), 
            ("参与率:", QLineEdit(), 60, "1")
        ]:
            row2.addWidget(QLabel(label))
            if isinstance(widget, QLineEdit): widget.setText(d_val)
            widget.setMinimumWidth(width)
            row2.addWidget(widget)
            if label == "Batch Size:": self.batch_input = widget
            elif label == "本地轮次:": self.local_epochs_input = widget
            elif label == "参与率:": self.join_ratio_input = widget

        row2.addStretch()
        cfg_lyt.addLayout(row2)
        cfg_grp.setLayout(cfg_lyt)
        main.addWidget(cfg_grp)

        # ===== 核心控制按钮 =====
        btn_row = QHBoxLayout()
        
        self.start_btn = QPushButton("开始训练")
        self.start_btn.clicked.connect(self.start_training)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        btn_row.addWidget(self.stop_btn)

        self.clear_btn = QPushButton("清空终端")
        self.clear_btn.setObjectName("clear_btn")
        self.clear_btn.clicked.connect(self.clear_log)
        btn_row.addWidget(self.clear_btn)

        # 新增帮助按钮
        self.help_btn = QPushButton("❓ 使用帮助")
        self.help_btn.setObjectName("help_btn")
        self.help_btn.clicked.connect(self.show_help)
        btn_row.addWidget(self.help_btn)

        btn_row.addStretch()

        # 分离按钮，增加小图标 ➦
        self.classify_btn = QPushButton("➦ 进入农作物分类系统")
        self.classify_btn.setObjectName("result_btn")
        self.classify_btn.clicked.connect(self.open_classify_window)
        btn_row.addWidget(self.classify_btn)
        
        main.addLayout(btn_row)

        # ===== 进度指示区 =====
        stat_grp = QGroupBox("训练状态")
        stat_lyt = QHBoxLayout()
        self.status_label = QLabel("状态: 已就绪")
        self.status_label.setStyleSheet(f"color: {NWAFU_GREEN}; font-weight:bold; font-size: 15px;")
        stat_lyt.addWidget(self.status_label)
        stat_lyt.addSpacing(30)
        
        self.round_label = QLabel("通信轮次: 0 / 0")
        self.acc_label = QLabel("当前准确率: ---%")
        self.loss_label = QLabel("训练损失: ---")
        for lbl in [self.round_label, self.acc_label, self.loss_label]:
            lbl.setStyleSheet("font-size: 15px; font-weight:bold; color: #333;")
            stat_lyt.addWidget(lbl)
            stat_lyt.addSpacing(20)
        stat_lyt.addStretch()
        stat_grp.setLayout(stat_lyt)
        main.addWidget(stat_grp)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("训练进度 %v%")
        main.addWidget(self.progress_bar)

        # ===== 终端日志区 =====
        log_grp = QGroupBox("运行终端 (Console)")
        log_lyt = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(250)
        self.log_text.setStyleSheet(f"""
            background-color: #0A1910; color: #F5E8C7; 
            padding: 10px; border-radius: 6px; font-family: 'Consolas', monospace;
            font-size: {FONT_SIZE_LOG}px; border: 1px solid rgba(220, 167, 40, 0.4);
        """)
        log_lyt.addWidget(self.log_text)
        log_grp.setLayout(log_lyt)
        main.addWidget(log_grp, stretch=1)

        # 新增：底部状态栏
        self.statusBar().showMessage(" 西北农林科技大学 | 联邦学习客户端协同训练平台 v1.0 | 仅供学术研究使用")
        self.statusBar().setStyleSheet(f"color: #555; font-size: 12px; background: transparent;")

        # 初始化调用联动
        self.on_algo_changed(self.algo_combo.currentText())

    def on_algo_changed(self, text):
        """联动功能：非 FedProx 算法不需要 μ 值"""
        if text == "FedAvg":
            self.mu_combo.setEnabled(False)
            self.mu_combo.setToolTip("FedAvg算法无需近端项 μ 值")
        else:
            self.mu_combo.setEnabled(True)
            self.mu_combo.setToolTip("")

    def show_help(self):
        """显示帮助说明"""
        help_text = (
            "【操作说明与参数释义】\n\n"
            "1. 聚合算法：\n"
            "   - FedAvg：经典的联邦平均算法。\n"
            "   - FedProx：增加近端项限制，适合非独立同分布(Non-IID)数据，需设置 μ 值。\n\n"
            "2. 核心参数：\n"
            "   - 全局轮次：服务器与客户端交换模型参数的总次数。\n"
            "   - 本地轮次(Epoch)：在每次通信前，客户端本地训练的完整遍历次数。\n"
            "   - 参与率：每轮抽取多少比例的客户端进行聚合（1即100%）。\n\n"
            "3. 农作物分类：\n"
            "   点击「进入农作物分类系统」，可加载训练产生的 .pt 模型对图像进行推理预测。"
        )
        QMessageBox.information(self, "平台使用说明", help_text)

    def _detect_device(self):
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        self.device_combo.clear()
        if has_cuda:
            self.device_combo.addItems(["cuda", "cpu"])
            self.device_combo.setCurrentText("cuda")
            self.append_log(f"[系统] CUDA 可用，默认设备: cuda | GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device_combo.addItem("cpu")
            self.append_log("[系统] CUDA 不可用，使用 CPU")

    def append_log(self, msg):
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        self.parse_training_info(msg)

    def parse_training_info(self, msg):
        try:
            flag = False
            if re.search(r'(?:Traceback|Error|Exception|failed)', msg, re.IGNORECASE): self.training_failed = True
            rm = re.search(r'Round\s*number:\s*(\d+)', msg, re.IGNORECASE)
            if rm:
                self.current_round = int(rm.group(1))
                self.round_label.setText(f"通信轮次: {self.current_round} / {self.total_rounds}")
                self.progress_bar.setValue(int((self.current_round / self.total_rounds) * 100))

            bm = re.search(r'(?:Best\s+accuracy)[^0-9]*(\d+\.\d+)', msg, re.IGNORECASE)
            if bm:
                self.current_acc = float(bm.group(1)) * (100 if float(bm.group(1)) <= 1.0 else 1)
                self.acc_label.setText(f"当前准确率: {self.current_acc:.2f}%")
                flag = True
            elif am := re.search(r'(?:Accuracy|acc|accuracy).*?(\d+\.\d+)', msg, re.IGNORECASE):
                val = float(am.group(1))
                if val <= 100:
                    self.current_acc = val * (100 if val <= 1.0 else 1)
                    self.acc_label.setText(f"当前准确率: {self.current_acc:.2f}%")
                    flag = True

            if lm := re.search(r'(?:loss|Loss).*?(\d+\.\d+)', msg, re.IGNORECASE):
                self.current_loss = float(lm.group(1))
                self.loss_label.setText(f"训练损失: {self.current_loss:.3f}")
                flag = True

            if m := re.search(r'(?:Averaged\s+Test\s+AUC|Avg.*AUC)[^0-9]*(\d+\.\d+)', msg, re.IGNORECASE): self.auc_info['avg_auc'] = float(m.group(1))
            if m := re.search(r'(?:Std\s+Test\s+Accuracy|Std.*Acc)[^0-9]*(\d+\.\d+)', msg, re.IGNORECASE): self.auc_info['std_acc'] = float(m.group(1))
            if m := re.search(r'(?:Std\s+Test\s+AUC|Std.*AUC)[^0-9]*(\d+\.\d+)', msg, re.IGNORECASE): self.auc_info['std_auc'] = float(m.group(1))

            if flag and self.current_round > 0:
                if not self.collected_rounds or self.collected_rounds[-1] != self.current_round:
                    self.collected_rounds.append(self.current_round)
                    self.collected_accs.append(self.current_acc)
                    self.collected_losses.append(self.current_loss)
                else:
                    self.collected_accs[-1], self.collected_losses[-1] = self.current_acc, self.current_loss
        except Exception: pass

    def start_training(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("状态: 训练运行中...")
        self.status_label.setStyleSheet("color: #e67e22; font-weight:bold; font-size: 15px;")
        self.training_failed, self.current_round, self.current_acc, self.current_loss = False, 0, 0.0, 0.0
        self.total_rounds = int(self.rounds_input.text())
        self.progress_bar.setValue(0)
        self.collected_rounds.clear(); self.collected_accs.clear(); self.collected_losses.clear()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        args = [
            sys.executable, os.path.join(script_dir, "main_v2.py"),
            "-algo", self.algo_combo.currentText(), "-m", self.model_combo.currentText(),
            "-data", self.dataset_combo.currentText(), "-gr", self.rounds_input.text(),
            "-nc", self.clients_combo.currentText(), "-lr", self.lr_combo.currentText(),
            "-lbs", self.batch_input.text(), "-dev", self.device_combo.currentText(),
            "-ls", self.local_epochs_input.text(), "-jr", self.join_ratio_input.text(),
            "-eg", "1", "-mu", self.mu_combo.currentText(), "-fs", "0",
        ]

        self.config_info = {
            "聚合算法": self.algo_combo.currentText(), "网络模型": self.model_combo.currentText(),
            "数据集": self.dataset_combo.currentText(), "全局通信轮次": self.rounds_input.text(),
            "参与客户端": self.clients_combo.currentText(), "本地学习率": self.lr_combo.currentText(),
            "FedProx μ值": self.mu_combo.currentText(), "Batch Size": self.batch_input.text(),
        }

        self.append_log(f"启动命令: {' '.join(args)}")
        self.training_thread = TrainingThread(args, script_dir)
        self.training_thread.log_signal.connect(self.append_log)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()

    def stop_training(self):
        if self.training_thread: self.training_thread.stop(); self.training_thread = None
        self.status_label.setText("状态: 手动终止")
        self.status_label.setStyleSheet("color: #8B0000; font-weight:bold; font-size: 15px;")
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.append_log(">>> 已接收中断指令，停止训练过程。")

    def training_finished(self, exit_code):
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        if self.training_failed or exit_code != 0:
            self.status_label.setText("状态: 训练异常")
            self.status_label.setStyleSheet("color: #8B0000; font-weight:bold; font-size: 15px;")
            self.append_log("\n训练异常终止，请排查报错信息！")
        else:
            self.status_label.setText("状态: 训练完成")
            self.status_label.setStyleSheet(f"color: {NWAFU_GREEN}; font-weight:bold; font-size: 15px;")
            self.append_log("\n训练正常结束，即将展示评估报告...")
            self.progress_bar.setValue(100)
            QTimer.singleShot(800, self.open_result_window)

    def open_result_window(self):
        if not self.result_win or not self.result_win.isVisible():
            self.result_win = ResultWindow(self.collected_rounds[:], self.collected_accs[:], self.collected_losses[:], self.total_rounds, dict(self.config_info), auc_info=dict(self.auc_info), parent=self)
            self.result_win.show()
        else: self.result_win.activateWindow()

    def open_classify_window(self):
        if not self.classify_win or not self.classify_win.isVisible():
            self.classify_win = ClassifyWindow(parent=self)
            self.classify_win.show()
        else: self.classify_win.activateWindow()

    def clear_log(self): self.log_text.clear()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = FedProxApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
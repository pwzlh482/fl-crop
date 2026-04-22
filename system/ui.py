#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习客户端协同训练平台 (终极稳定版)
- 修复核心Bug：彻底解决频繁点击开始/停止引发的 QThread 闪退问题（采用安全挂起与信号回调策略）
- 优化：删除分类结果、模型加载及帮助文本中的所有小图标 (Emoji)
- 优化：分类界面主标题修改为“农作物识别分类系统”
- 优化：规范按钮状态流转，避免重复点击引发的并发冲突
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
# 全局样式常量
# ============================================================
FONT_SIZE_LABEL = 13       
FONT_SIZE_INPUT = 13       
FONT_SIZE_TITLE = 34       
FONT_SIZE_GROUP = 14       
FONT_SIZE_BTN = 15         
FONT_SIZE_LOG = 13         

NWAFU_GREEN = "#2E8B57"
NWAFU_YELLOW = "#F39C12"
TEXT_MAIN = "#2C3E50"

BG_GRADIENT = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #F0F9F0, stop:1 #FFFDF2)"

GLOBAL_STYLE = f"""
QMainWindow {{ background-color: #F0F9F0; }}

QWidget#central_widget {{ background: {BG_GRADIENT}; }}

QGroupBox {{
    font-weight: bold; font-size: {FONT_SIZE_GROUP}px;
    border: 1px solid rgba(46, 139, 87, 0.3);
    border-radius: 8px;
    margin-top: 16px; padding-top: 15px; 
    background-color: rgba(255, 255, 255, 0.75); 
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 15px;
    padding: 0 8px; color: {NWAFU_GREEN};
}}

QPushButton {{
    background-color: {NWAFU_GREEN}; color: white;
    border: none; padding: 10px 24px;
    border-radius: 6px; font-weight: bold;
    font-size: {FONT_SIZE_BTN}px; min-height: 25px;
}}
QPushButton:hover {{ background-color: #247146; }}
QPushButton:disabled {{ background-color: #A0B2A6; color: #F0F0F0; }}

QPushButton#stop_btn {{ background-color: #C0392B; }}
QPushButton#stop_btn:hover {{ background-color: #A93226; }}

QPushButton#clear_btn {{ background-color: #7F8C8D; }}
QPushButton#clear_btn:hover {{ background-color: #616A6B; }}

QPushButton#help_btn {{
    background-color: #3498DB; padding: 5px 12px; 
    font-size: 12px; border-radius: 4px; min-height: 15px;
    margin-right: 15px;
}}
QPushButton#help_btn:hover {{ background-color: #2980B9; }}

QPushButton#result_btn {{
    background-color: {NWAFU_YELLOW}; color: white;
    padding: 12px 30px; border-radius: 6px;
    font-size: {FONT_SIZE_BTN}px; font-weight: 900;
}}
QPushButton#result_btn:hover {{ background-color: #D68910; }}

QComboBox, QLineEdit {{
    padding: 5px 10px; border: 1px solid #BDC3C7;
    border-radius: 4px; background-color: white;
    font-size: {FONT_SIZE_INPUT}px; min-width: 90px;
    color: {TEXT_MAIN};
}}
QComboBox:hover, QLineEdit:focus {{ border: 1px solid {NWAFU_GREEN}; }}
QComboBox:disabled {{ background-color: #EAEDED; color: #95A5A6; }}
QComboBox QAbstractItemView {{
    background-color: white; selection-background-color: #E8F5E9;
    selection-color: {NWAFU_GREEN};
}}

QLabel {{ color: {TEXT_MAIN}; font-size: {FONT_SIZE_LABEL}px; background: transparent; }}

QProgressBar {{
    border: 1px solid rgba(46, 139, 87, 0.4);
    border-radius: 6px; text-align: center;
    background-color: rgba(255,255,255,0.9); color: {TEXT_MAIN}; font-weight: bold;
    height: 22px;
}}
QProgressBar::chunk {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2E8B57, stop:1 #3CB371);
    border-radius: 4px;
}}

QHeaderView::section {{
    background-color: {NWAFU_GREEN}; color: white;
    font-weight: bold; padding: 8px; border: 1px solid #247146;
}}
QTableWidget {{ gridline-color: #D0D0D0; background: white; }}
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
            
            # 等待进程完全结束
            self.process.wait()
            self.exit_code = self.process.returncode
        except Exception as e:
            self.log_signal.emit(f"[系统错误] {str(e)}")
            self.exit_code = 1
        finally:
            # 必须等到进程死透了再发射完成信号，这样外面覆盖线程时才安全
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
        main_layout = QVBoxLayout(central)
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
        self.ax.set_ylabel('Accuracy (%)', fontsize=11, color='#2E8B57')
        self.ax.grid(True, alpha=0.3)
        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel('Loss', fontsize=11, color='#C0392B')

        if self.rounds_data:
            self.ax.plot(self.rounds_data, self.acc_data, 'o-', color='#2E8B57', label='Test Acc', linewidth=2, markersize=5)
            self.ax2.plot(self.rounds_data, self.loss_data, 's-', color='#C0392B', label='Train Loss', linewidth=2, markersize=5)
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

        config_group = QGroupBox("训练配置存档")
        cfg_layout = QVBoxLayout()
        config_text = QTextEdit()
        config_text.setReadOnly(True)
        config_text.setMaximumHeight(100)
        lines = [f"• {k}: {v}" for k, v in self.config_info.items()]
        config_text.setText('\n'.join(lines))
        config_text.setStyleSheet(f"background-color: white; color: {TEXT_MAIN}; font-size: {FONT_SIZE_LABEL}px; border-radius: 4px; padding: 10px;")
        cfg_layout.addWidget(config_text)
        config_group.setLayout(cfg_layout)
        main_layout.addWidget(config_group, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("导出数据与图表")
        save_btn.setObjectName("result_btn")
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
        path, _ = QFileDialog.getSaveFileName(self, "保存图表", default_path, "PNG (*.png)")
        if path:
            self.fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
            QMessageBox.information(self, "成功", f"图表已保存至:\n{path}")


class ClassifyWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.loaded_model, self.loaded_dataset, self.class_names, self.image_path = None, None, None, None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("农作物识别分类系统")
        self.setGeometry(180, 100, 950, 680)
        self.setStyleSheet(GLOBAL_STYLE)

        central = QWidget()
        central.setObjectName("central_widget")
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(25, 25, 25, 25)

        # 标题修改
        title = QLabel("农作物识别分类系统")
        title.setStyleSheet(f"color: {NWAFU_GREEN}; font-size: 28px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        top_bar = QGroupBox("1. 模型配置与图像上传")
        top_bar_layout = QHBoxLayout()
        load_model_btn = QPushButton("加载训练模型 (.pt)")
        load_model_btn.clicked.connect(self.on_load_model)
        top_bar_layout.addWidget(load_model_btn)
        
        self.model_status = QLabel("当前模型: 未加载")
        self.model_status.setStyleSheet("color: #C0392B; font-weight: bold;")
        top_bar_layout.addWidget(self.model_status)
        top_bar_layout.addSpacing(30)
        
        upload_btn = QPushButton("选择待测图片")
        upload_btn.clicked.connect(self.on_select_image)
        top_bar_layout.addWidget(upload_btn)
        
        self.image_status = QLabel("图片状态: 未选择")
        self.image_status.setStyleSheet("color: #C0392B; font-weight: bold;")
        top_bar_layout.addWidget(self.image_status)
        
        top_bar_layout.addStretch()
        top_bar.setLayout(top_bar_layout)
        main_layout.addWidget(top_bar)

        content_layout = QHBoxLayout()
        img_group = QGroupBox("2. 图像预览区")
        img_layout = QVBoxLayout()
        
        # 去掉占位图的小图标
        self.result_img_label = QLabel("暂无图片\n请点击上方按钮上传")
        self.result_img_label.setAlignment(Qt.AlignCenter)
        self.result_img_label.setMinimumSize(420, 420)
        self.result_img_label.setStyleSheet(f"background-color: white; border: 2px dashed {NWAFU_GREEN}; border-radius: 8px; color: #7F8C8D;")
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
            color: {TEXT_MAIN}; padding: 12px;
        """)
        self.result_text.setPlaceholderText("推理结果、置信度以及耗时将在此处展示...")
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
                if not model_str: model_str, _ = QInputDialog.getText(self, "输入", "请输入模型 (默认 MobileNet):", text="MobileNet")
                
                num_classes = 10 if dataset in ['Cifar10', 'MNIST', 'Digit5'] else 6
                self.loaded_model = load_pytorch_model(path, model_str, dataset, num_classes, self.device)
                self.loaded_model_str, self.loaded_dataset = model_str, dataset
                self.class_names = ['airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] if dataset == 'Cifar10' else [str(i) for i in range(10)]
                
                self.model_status.setText(f"已加载: {model_str} ({dataset})")
                self.model_status.setStyleSheet(f"color: {NWAFU_GREEN}; font-weight: bold;")
                # 删除加载成功文本中的 Emoji
                self.result_text.setPlainText(f"模型加载成功！\n\n网络: {model_str}\n数据集: {dataset}\n类型数: {num_classes}\n\n请点击上方开始预测。")
            except Exception as e:
                QMessageBox.critical(self, "加载失败", f"模型解析失败，请检查文件: {str(e)}")

    def on_classify(self):
        if not self.loaded_model or not self.image_path:
            return QMessageBox.warning(self, "警告", "请确保模型和待测图片均已加载！")
        try:
            image = Image.open(self.image_path).convert('L' if self.loaded_dataset == 'MNIST' else 'RGB')
            size, mean, std = ((28,28), (0.5,), (0.5,)) if self.loaded_dataset == 'MNIST' else ((32,32), (0.5,0.5,0.5), (0.5,0.5,0.5))
            transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean, std)])
            
            with torch.no_grad():
                out = self.loaded_model(transform(image).unsqueeze(0).to(self.device))
                prob = torch.nn.functional.softmax(out, dim=1)[0]
                
                k = min(3, len(prob)) 
                topk_prob, topk_indices = torch.topk(prob, k)
            
            # 删除推理结果展示中的 Emoji
            res_str = "预测完成！\n\n【Top-3 预测结果】\n"
            for i in range(k):
                idx = topk_indices[i].item()
                p = topk_prob[i].item()
                cname = self.class_names[idx] if self.class_names else str(idx)
                res_str += f"TOP-{i+1}: {cname} (ID: {idx})\n     置信度: {p:.2%}\n"
                
            res_str += f"\n-----------------------------\n"
            res_str += f"文件: {os.path.basename(self.image_path)}\n"
            res_str += f"模型: {self.loaded_model_str}\n"
            res_str += f"数据集: {self.loaded_dataset}"
            
            self.result_text.setPlainText(res_str)
        except Exception as e:
            QMessageBox.critical(self, "分类推理失败", f"处理图像时发生错误: {str(e)}")


class FedProxApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.training_thread, self.result_win, self.classify_win = None, None, None
        self.current_round, self.total_rounds, self.current_acc, self.current_loss = 0, 100, 0.0, 0.0
        self.training_failed = False
        self.is_stopped_manually = False 
        self.collected_rounds, self.collected_accs, self.collected_losses = [], [], []
        self.auc_info = {}

        self.init_ui()
        self._detect_device()

    def init_ui(self):
        self.setWindowTitle("联邦学习客户端协同训练平台")
        self.setGeometry(100, 80, 1080, 760)
        self.setStyleSheet(GLOBAL_STYLE)

        central = QWidget()
        central.setObjectName("central_widget")
        self.setCentralWidget(central)
        
        main = QVBoxLayout(central)
        main.setSpacing(12)
        main.setContentsMargins(25, 20, 25, 10) 

        # ===== 标题、校徽与帮助区 =====
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        left_lyt = QHBoxLayout()
        self.logo_label = QLabel()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logo_paths = [os.path.join(script_dir, "nwafu_logo.png"), os.path.join(script_dir, "image_e9656f.png")]
        
        logo_loaded = False
        for path in logo_paths:
            if os.path.exists(path):
                pixmap = QPixmap(path).scaled(75, 75, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.logo_label.setPixmap(pixmap)
                logo_loaded = True
                break
        if not logo_loaded:
            self.logo_label.setText("") 
            
        left_lyt.addWidget(self.logo_label)
        left_lyt.addStretch()
        
        title = QLabel("联邦学习客户端协同训练平台")
        title.setStyleSheet(f"color: {NWAFU_GREEN}; font-size: {FONT_SIZE_TITLE}px; font-weight: 900; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        
        right_lyt = QHBoxLayout()
        right_lyt.addStretch()
        self.help_btn = QPushButton("使用帮助")
        self.help_btn.setObjectName("help_btn")
        self.help_btn.clicked.connect(self.show_help)
        self.help_btn.setCursor(Qt.PointingHandCursor)
        right_lyt.addWidget(self.help_btn)

        header_layout.addLayout(left_lyt, 1)
        header_layout.addWidget(title, 2)
        header_layout.addLayout(right_lyt, 1)

        main.addWidget(header_widget)

        # ===== 配置区 =====
        cfg_grp = QGroupBox("训练参数配置")
        cfg_lyt = QVBoxLayout()
        row1 = QHBoxLayout()
        
        left_form = QFormLayout()
        left_form.setSpacing(12)
        left_form.setLabelAlignment(Qt.AlignRight)

        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["FedProxV2", "FedProx", "FedAvg"])
        self.algo_combo.currentTextChanged.connect(self.on_algo_changed) 
        left_form.addRow("聚合算法:", self.algo_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["MobileNet", "ResNet18", "CNN", "DNN"]) 
        left_form.addRow("网络模型:", self.model_combo)

        self.dataset_combo = QComboBox()
        self.dataset_combo.setEditable(True)
        self.dataset_combo.addItems(["Cifar10", "MNIST"])
        left_form.addRow("数据集:", self.dataset_combo)

        row1.addLayout(left_form, 1)
        row1.addSpacing(40)

        right_form = QFormLayout()
        right_form.setSpacing(12)
        right_form.setLabelAlignment(Qt.AlignRight)

        self.rounds_combo = QComboBox()
        self.rounds_combo.setEditable(True)
        self.rounds_combo.addItems(["50", "100", "200", "300"])
        self.rounds_combo.setCurrentText("100")
        right_form.addRow("全局轮次:", self.rounds_combo)

        self.clients_combo = QComboBox()
        self.clients_combo.setEditable(True)
        self.clients_combo.addItems(["1", "2", "5", "10"])
        self.clients_combo.setCurrentText("2")
        right_form.addRow("客户端数:", self.clients_combo)

        self.join_ratio_combo = QComboBox()
        self.join_ratio_combo.setEditable(True)
        self.join_ratio_combo.addItems(["0.1", "0.5", "1.0"])
        self.join_ratio_combo.setCurrentText("1")
        right_form.addRow("参与率:", self.join_ratio_combo)

        row1.addLayout(right_form, 1)
        cfg_lyt.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(10)
        
        def add_param_widget(layout, label_str, widget):
            wrapper = QWidget()
            l = QHBoxLayout(wrapper)
            l.setContentsMargins(0, 0, 0, 0)
            l.addWidget(QLabel(label_str))
            l.addWidget(widget)
            layout.addWidget(wrapper, stretch=1)

        self.lr_combo = QComboBox()
        self.lr_combo.setEditable(True)
        self.lr_combo.addItems(["0.01", "0.05", "0.1", "0.001"])
        self.lr_combo.setCurrentText("0.05")
        add_param_widget(row2, "学习率:", self.lr_combo)

        self.batch_combo = QComboBox()
        self.batch_combo.setEditable(True)
        self.batch_combo.addItems(["16", "32", "64", "128"])
        self.batch_combo.setCurrentText("64")
        add_param_widget(row2, "Batch Size:", self.batch_combo)

        self.local_epochs_combo = QComboBox()
        self.local_epochs_combo.setEditable(True)
        self.local_epochs_combo.addItems(["1", "3", "5", "10"])
        self.local_epochs_combo.setCurrentText("3")
        add_param_widget(row2, "本地轮次:", self.local_epochs_combo)

        self.device_combo = QComboBox()
        add_param_widget(row2, "运行设备:", self.device_combo)

        self.mu_combo = QComboBox()
        self.mu_combo.setEditable(True)
        self.mu_combo.addItems(["0", "0.01", "0.1", "1.0"])
        self.mu_combo.setCurrentText("0.1")
        add_param_widget(row2, "FedProx μ值:", self.mu_combo)

        cfg_lyt.addLayout(row2)
        cfg_grp.setLayout(cfg_lyt)
        main.addWidget(cfg_grp)

        # ===== 控制按钮 =====
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

        btn_row.addStretch()

        self.classify_btn = QPushButton("👉 进入农作物分类系统")
        self.classify_btn.setObjectName("result_btn")
        self.classify_btn.clicked.connect(self.open_classify_window)
        btn_row.addWidget(self.classify_btn)
        
        main.addLayout(btn_row)

        # ===== 状态栏可视化反馈 =====
        stat_grp = QGroupBox("训练状态监控")
        stat_lyt = QHBoxLayout()
        
        self.status_label = QLabel("状态: 已就绪")
        self.status_label.setStyleSheet(f"color: {TEXT_MAIN}; font-weight:bold; font-size: 15px;")
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
            font-size: {FONT_SIZE_LOG}px; border: 1px solid rgba(243, 156, 18, 0.5);
        """)
        log_lyt.addWidget(self.log_text)
        log_grp.setLayout(log_lyt)
        main.addWidget(log_grp, stretch=1)

        self.statusBar().showMessage(" 西北农林科技大学 | 联邦学习协同平台 v2.0 | 仅供学术研究使用")
        self.statusBar().setStyleSheet(f"color: #7F8C8D; font-size: 12px; background: transparent;")

        self.on_algo_changed(self.algo_combo.currentText())

    def on_algo_changed(self, text):
        if text == "FedAvg":
            self.mu_combo.setEnabled(False)
            self.mu_combo.setToolTip("FedAvg算法无需近端项 μ 值")
        else:
            self.mu_combo.setEnabled(True)
            self.mu_combo.setToolTip("非独立同分布(Non-IID)下的近端约束系数")

    def show_help(self):
        # 移除帮助文本中的所有图标和特殊符号格式
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
            # 移除了诊断建议中的灯泡图标
            if "CUDA out of memory" in msg:
                self.training_failed = True
                self.log_text.append(f"\n<span style='color:#E74C3C; font-weight:bold;'>诊断建议：GPU显存不足！请尝试减小 Batch Size。</span>\n")
            elif "No such file or directory" in msg and ("data" in msg.lower() or "dataset" in msg.lower()):
                self.training_failed = True
                self.log_text.append(f"\n<span style='color:#E74C3C; font-weight:bold;'>诊断建议：找不到数据集，请检查数据集是否已下载或路径。</span>\n")
            elif re.search(r'(?:Traceback|Error|Exception|failed)', msg, re.IGNORECASE): 
                self.training_failed = True

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
        self.status_label.setText("状态: 训练中...")
        self.status_label.setStyleSheet("color: #D68910; font-weight:bold; font-size: 15px;")
        
        self.training_failed, self.current_round, self.current_acc, self.current_loss = False, 0, 0.0, 0.0
        self.is_stopped_manually = False 
        
        self.total_rounds = int(self.rounds_combo.currentText())
        self.progress_bar.setValue(0)
        self.collected_rounds.clear(); self.collected_accs.clear(); self.collected_losses.clear()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        args = [
            sys.executable, os.path.join(script_dir, "main_v2.py"),
            "-algo", self.algo_combo.currentText(), "-m", self.model_combo.currentText(),
            "-data", self.dataset_combo.currentText(), "-gr", self.rounds_combo.currentText(),
            "-nc", self.clients_combo.currentText(), "-lr", self.lr_combo.currentText(),
            "-lbs", self.batch_combo.currentText(), "-dev", self.device_combo.currentText(),
            "-ls", self.local_epochs_combo.currentText(), "-jr", self.join_ratio_combo.currentText(),
            "-eg", "1", "-mu", self.mu_combo.currentText(), "-fs", "0",
        ]

        self.config_info = {
            "聚合算法": self.algo_combo.currentText(), "网络模型": self.model_combo.currentText(),
            "数据集": self.dataset_combo.currentText(), "全局通信轮次": self.rounds_combo.currentText(),
            "参与客户端": self.clients_combo.currentText(), "本地学习率": self.lr_combo.currentText(),
            "FedProx μ值": self.mu_combo.currentText(), "Batch Size": self.batch_combo.currentText(),
        }

        self.append_log(f"启动命令: {' '.join(args)}")
        
        # 安全处理旧线程（防止连续点击重叠）
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.training_thread.wait()
            
        self.training_thread = TrainingThread(args, script_dir)
        self.training_thread.log_signal.connect(self.append_log)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()

    def stop_training(self):
        # 激活手动中断锁，并禁用开始按钮直到彻底结束
        self.is_stopped_manually = True
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(False) # 防止在杀进程期间被再次启动
        
        self.status_label.setText("状态: 正在中断...")
        self.status_label.setStyleSheet("color: #C0392B; font-weight:bold; font-size: 15px;")
        self.append_log(">>> 正在发送中断指令，请稍候...")
        
        if self.training_thread: 
            self.training_thread.stop()
            # 核心修复点：切勿执行 self.training_thread = None !
            # 必须等待底层 C++ 线程发出 finished_signal 信号安全回收

    def training_finished(self, exit_code):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # 拦截手动终止的事件，只更新UI，不触发报错弹窗
        if self.is_stopped_manually:
            self.status_label.setText("状态: ⏸ 训练中断")
            self.status_label.setStyleSheet("color: #C0392B; font-weight:bold; font-size: 15px;")
            self.append_log(">>> ⏸ 训练已安全中断。")
            return

        if self.training_failed or exit_code != 0:
            self.status_label.setText("状态: ⚠️ 训练异常")
            self.status_label.setStyleSheet("color: #C0392B; font-weight:bold; font-size: 15px;")
            self.append_log("\n⚠️ 训练异常终止，请往上查看日志或修改配置重试。")
        else:
            self.status_label.setText("状态: 训练完成")
            self.status_label.setStyleSheet(f"color: {NWAFU_GREEN}; font-weight:bold; font-size: 15px;")
            self.append_log("\n训练正常结束，即将展示评估报告图表...")
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
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = FedProxApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
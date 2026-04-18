#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习客户端协同训练平台 (V3)
- 主窗口：配置 + 进度 + 日志（紧凑）
- 结果窗口：训练结束后自动弹出，图表 + 性能分析
- 分类窗口：农作物分类（预留）
"""

import sys
import os
import subprocess
import time
import re
import platform
import signal
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QProgressBar, QGroupBox,
    QComboBox, QLineEdit, QFormLayout, QTabWidget, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QMessageBox, QFileDialog, QFrame, QSizePolicy,
    QAbstractItemView, QInputDialog
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
# 全局样式常量 — 统一字体大小
# ============================================================
FONT_SIZE_LABEL = 12       # 标签字体
FONT_SIZE_INPUT = 12       # 输入框字体
FONT_SIZE_TITLE = 28       # 标题字体
FONT_SIZE_GROUP = 13       # 分组标题字体
FONT_SIZE_BTN = 14         # 按钮字体
FONT_SIZE_LOG = 11         # 日志字体

GLOBAL_STYLE = f"""
QMainWindow {{ background-color: #f5f6fa; }}
QGroupBox {{
    font-weight: bold; font-size: {FONT_SIZE_GROUP}px;
    border: 2px solid #3498db; border-radius: 8px;
    margin-top: 12px; padding-top: 12px; background-color: white;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px;
    padding: 0 8px; color: #3498db;
}}
QPushButton {{
    background-color: #3498db; color: white;
    border: none; padding: 14px 32px;
    border-radius: 6px; font-weight: bold;
    font-size: {FONT_SIZE_BTN}px; min-height: 20px;
}}
QPushButton:hover {{ background-color: #2980b9; }}
QPushButton:disabled {{ background-color: #bdc3c7; }}
QPushButton#stop_btn {{ background-color: #e74c3c; }}
QPushButton#stop_btn:hover {{ background-color: #c0392b; }}
QPushButton#clear_btn {{ background-color: #95a5a6; }}
QPushButton#clear_btn:hover {{ background-color: #7f8c8d; }}
QPushButton#result_btn {{
    background-color: #27ae60; padding: 10px 24px;
    font-size: {FONT_SIZE_BTN - 1}px;
}}
QPushButton#result_btn:hover {{ background-color: #219a52; }}
QComboBox {{
    padding: 8px 12px; border: 1px solid #bdc3c7;
    border-radius: 4px; background-color: white;
    font-size: {FONT_SIZE_INPUT}px; min-width: 130px;
}}
QLineEdit {{
    padding: 8px 12px; border: 1px solid #bdc3c7;
    border-radius: 4px; background-color: white;
    font-size: {FONT_SIZE_INPUT}px; min-width: 100px;
}}
QLabel {{ color: #2c3e50; font-size: {FONT_SIZE_LABEL}px; }}
QTextEdit {{ font-family: 'Consolas', monospace; font-size: {FONT_SIZE_LOG}px; }}
QHeaderView::section {{
    background-color: #3498db; color: white;
    font-weight: bold; padding: 6px; border: 1px solid #2980b9;
}}
QTableWidget {{ gridline-color: #d0d0d0; }}
"""


class TrainingThread(QThread):
    """训练线程"""
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
                'cwd': self.cwd,
                'stdout': subprocess.PIPE,
                'stderr': subprocess.STDOUT,
                'text': True, 'encoding': 'utf-8',
                'errors': 'replace', 'bufsize': 1, 'env': env
            }
            if platform.system() == 'Windows':
                kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs['preexec_fn'] = os.setsid

            self.process = subprocess.Popen(self.args, **kwargs)

            for line in iter(self.process.stdout.readline, ''):
                if not self.running:
                    break
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
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.process.pid)],
                                    creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except Exception:
                self.process.kill()


# ============================================================
# 结果窗口 — 训练结束时自动弹出
# ============================================================
class ResultWindow(QMainWindow):
    """训练结果窗口：完整图表 + 性能分析"""

    def __init__(self, rounds_data, acc_data, loss_data, total_rounds,
                 config_info, auc_info=None, parent=None):
        super().__init__(parent)
        self.rounds_data = rounds_data
        self.acc_data = acc_data
        self.loss_data = loss_data
        self.total_rounds = total_rounds
        self.config_info = config_info
        self.auc_info = auc_info or {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("训练结果分析")
        self.setGeometry(150, 80, 1100, 750)
        self.setStyleSheet(GLOBAL_STYLE)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 标题
        title = QLabel("📊 训练结果分析")
        title.setFont(QFont("Microsoft YaHei", FONT_SIZE_TITLE - 2, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50;")
        main_layout.addWidget(title)

        # 上半部分：双图并排
        chart_row = QHBoxLayout()

        # --- 左：准确率 + Loss 双轴图 ---
        fig_group = QGroupBox("训练曲线")
        fig_layout = QVBoxLayout()
        self.fig, self.ax = plt.subplots(figsize=(7, 4), dpi=120)
        self.canvas = FigureCanvas(self.fig)

        self.ax.set_xlabel('Global Rounds', fontsize=11, labelpad=8)
        self.ax.set_ylabel('Accuracy (%)', fontsize=11, color='#2ecc71')
        self.ax.grid(True, alpha=0.3)
        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel('Loss', fontsize=11, color='#e74c3c')

        if self.rounds_data:
            self.line1, = self.ax.plot(
                self.rounds_data, self.acc_data,
                'o-', color='#2ecc71', label='Test Acc',
                linewidth=2.2, markersize=5)
            self.line2, = self.ax2.plot(
                self.rounds_data, self.loss_data,
                's-', color='#e74c3c', label='Train Loss',
                linewidth=2.2, markersize=5)

            total = max(self.total_rounds, max(self.rounds_data) + 1)
            self.ax.set_xlim(0, total)

            # 智能刻度
            if total <= 30:
                self.ax.set_xticks(range(0, total + 1))
            else:
                step = max(1, total // 10)
                ticks = list(range(0, total + step, step))
                if ticks[-1] != total:
                    ticks.append(total)
                self.ax.set_xticks(ticks)

            self.ax.set_ylim(
                0, max(100, max(self.acc_data) + 5) if self.acc_data else 100)
            self.ax2.set_ylim(
                0, max(5, max(self.loss_data) + 0.5) if self.loss_data else 5)
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

        # --- 右：性能指标表 ---
        stats_group = QGroupBox("性能指标")
        stats_layout = QVBoxLayout()
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["指标", "值"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setColumnWidth(0, 140)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.setSelectionBehavior(
            QAbstractItemView.SelectRows)
        self._fill_stats_table()
        stats_layout.addWidget(self.stats_table)
        stats_group.setLayout(stats_layout)
        chart_row.addWidget(stats_group, 2)

        main_layout.addLayout(chart_row, stretch=3)

        # 下半部分：训练配置摘要
        config_group = QGroupBox("训练配置")
        cfg_layout = QVBoxLayout()
        config_text = QTextEdit()
        config_text.setReadOnly(True)
        config_text.setMaximumHeight(90)
        lines = [f"• {k}: {v}" for k, v in self.config_info.items()]
        config_text.setText('\n'.join(lines))
        config_text.setStyleSheet(
            "background-color: #f8f9fa; color: #2c3e50;"
            f"font-size: {FONT_SIZE_LABEL}px; border-radius: 4px;"
            "border: 1px solid #dee2e6; padding: 8px;"
        )
        cfg_layout.addWidget(config_text)
        config_group.setLayout(cfg_layout)
        main_layout.addWidget(config_group)

        # 底部操作按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("💾 导出图片")
        save_btn.setObjectName("result_btn")
        save_btn.clicked.connect(self.export_figure)
        btn_row.addWidget(save_btn)

        close_btn = QPushButton("关闭窗口")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        btn_row.addStretch()
        main_layout.addLayout(btn_row)

    def _fill_stats_table(self):
        """填充性能指标表格"""
        rows_data = []

        if self.acc_data and self.rounds_data:
            final_acc = self.acc_data[-1]
            rows_data.append(("最终准确率", f"{final_acc:.2f}%"))
            best_acc = max(self.acc_data)
            best_round = self.acc_data.index(best_acc) + 1
            rows_data.append(("最佳准确率", f"{best_acc:.2f}% (Round {best_round})"))

        # 从配置中获取 AUC / Std 信息
        auc_info = getattr(self, 'auc_info', {})
        if auc_info:
            if 'avg_auc' in auc_info:
                rows_data.append(("平均测试 AUC", f"{auc_info['avg_auc']:.4f}"))
            if 'std_acc' in auc_info:
                rows_data.append(("标准差 (Accuracy)", f"{auc_info['std_acc']:.4f}"))
            if 'std_auc' in auc_info:
                rows_data.append(("标准差 (AUC)", f"{auc_info['std_auc']:.4f}"))

        if self.loss_data and self.rounds_data:
            rows_data.append(("最终损失", f"{self.loss_data[-1]:.4f}"))

        rows_data.append(("总轮次", str(len(self.rounds_data))))
        rows_data.append(("配置轮次", str(self.total_rounds)))

        self.stats_table.setRowCount(len(rows_data))
        for i, (metric, value) in enumerate(rows_data):
            self.stats_table.setItem(i, 0, QTableWidgetItem(metric))
            item = QTableWidgetItem(value)
            # 最佳值高亮
            if "最佳" in metric or "平均测试 AUC" in metric:
                item.setForeground(Qt.darkGreen)
            self.stats_table.setItem(i, 1, item)

    def export_figure(self):
        """导出图片为PNG"""
        default_path = os.path.join(os.path.expanduser("~"),
                                    "Desktop",
                                    f"training_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", default_path, "PNG (*.png)")
        if path:
            self.fig.savefig(path, dpi=200, bbox_inches='tight',
                              facecolor='white')
            QMessageBox.information(self, "成功", f"已保存至:\n{path}")


# ============================================================
# 分类窗口 — 农作物分类（预留）
# ============================================================
class ClassifyWindow(QMainWindow):
    """农作物分类窗口（预留接口）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # 分类功能相关属性
        self.loaded_model = None
        self.loaded_dataset = None
        self.loaded_model_str = None
        self.class_names = None
        self.image_path = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("🌾 农作物分类")
        self.setGeometry(180, 100, 800, 600)
        self.setStyleSheet(GLOBAL_STYLE)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 标题
        title = QLabel("农作物智能分类系统")
        title.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 15px 0;")
        layout.addWidget(title)

        # 上传区域（可点击按钮）
        upload_btn = QPushButton("📷 点击或拖拽上传农作物图片")
        upload_btn.setObjectName("upload_btn")
        upload_btn.setStyleSheet("""
            QPushButton#upload_btn {
                background-color: #ecf0f1; border: 2px dashed #bdc3c7;
                border-radius: 12px; color: #7f8c8d;
                font-size: %dpx; font-weight: bold; padding: 60px;
                min-height: 100px;
            }
            QPushButton#upload_btn:hover {
                background-color: #d5dbdb; border-color: #95a5a6;
            }
            QPushButton#upload_btn:pressed {
                background-color: #bfc9ca;
            }
        """ % (FONT_SIZE_LABEL + 2))
        upload_btn.clicked.connect(self.on_select_image)
        layout.addWidget(upload_btn)
        
        # 状态行
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(5, 5, 5, 5)
        
        self.model_status = QLabel("模型: 未加载")
        self.model_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
        status_layout.addWidget(self.model_status)
        
        status_layout.addStretch()
        
        self.image_status = QLabel("图片: 未选择")
        self.image_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
        status_layout.addWidget(self.image_status)
        
        layout.addWidget(status_frame)

        # 结果区域
        result_group = QGroupBox("分类结果")
        result_layout = QHBoxLayout()
        self.result_img_label = QLabel("预览区")
        self.result_img_label.setAlignment(Qt.AlignCenter)
        self.result_img_label.setMinimumSize(250, 250)
        self.result_img_label.setStyleSheet(
            "background-color: #f0f0f0; border-radius: 8px;")
        result_layout.addWidget(self.result_img_label)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(200)
        self.result_text.setStyleSheet("""
            background-color: #fffde7;
            border: 1px solid #ffd54f; border-radius: 6px;
            font-size: {FON}px;""".format(FON=FONT_SIZE_LABEL))
        self.result_text.setPlaceholderText("上传图片后将在此显示分类结果...")
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        # 按钮
        btn_row = QHBoxLayout()
        classify_btn = QPushButton("🔍 开始分类")
        classify_btn.setObjectName("result_btn")
        classify_btn.clicked.connect(self.on_classify)
        btn_row.addWidget(classify_btn)

        load_model_btn = QPushButton("📂 加载模型")
        load_model_btn.setObjectName("result_btn")
        load_model_btn.clicked.connect(self.on_load_model)
        btn_row.addWidget(load_model_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def on_select_image(self):
        """选择图片文件"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "", "Image files (*.jpg *.jpeg *.png *.bmp);;All Files (*)")
        if path:
            self.image_path = path
            # 显示预览
            pixmap = QPixmap(path)
            if pixmap.isNull():
                QMessageBox.warning(self, "警告", "无法加载图片文件")
                return
            # 缩放以适应标签
            pixmap = pixmap.scaled(self.result_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.result_img_label.setPixmap(pixmap)
            # 更新状态标签
            if hasattr(self, 'image_status'):
                self.image_status.setText(f"图片: {os.path.basename(path)}")
                self.image_status.setStyleSheet("color: #27ae60; font-weight: bold;")
            # 更新结果文本
            self.result_text.setPlainText(f"已加载图片:\n{path}\n\n请加载模型后点击'开始分类'")

    def on_classify(self):
        """执行分类预测"""
        if self.loaded_model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        if self.image_path is None:
            QMessageBox.warning(self, "警告", "请先选择图片")
            return
        
        try:
            # 获取数据集类别数
            num_classes = self._get_num_classes(self.loaded_dataset)
            
            # 图片预处理（根据数据集调整）
            image = Image.open(self.image_path)
            
            # 确定预处理参数
            if self.loaded_dataset == 'MNIST':
                # MNIST是灰度图
                image = image.convert('L')
                target_size = (28, 28)
                normalize_mean = (0.5,)
                normalize_std = (0.5,)
                in_channels = 1
            else:
                # 其他数据集默认为RGB
                image = image.convert('RGB')
                target_size = (32, 32)  # Cifar10等
                normalize_mean = (0.5, 0.5, 0.5)
                normalize_std = (0.5, 0.5, 0.5)
                in_channels = 3
            
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
            ])
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.loaded_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # 显示结果
            class_name = self.class_names[predicted_class] if self.class_names else str(predicted_class)
            result = f"""分类结果:
图片: {os.path.basename(self.image_path)}
预测类别: {class_name} (索引: {predicted_class})
置信度: {confidence:.2%}
模型: {self.loaded_model_str}
数据集: {self.loaded_dataset}
"""
            self.result_text.setPlainText(result)
            
            # 可选：在状态栏显示分类结果
            if hasattr(self, 'model_status'):
                self.model_status.setText(f"模型: {self.loaded_model_str} ({self.loaded_dataset}) ✓")
            
        except Exception as e:
            QMessageBox.critical(self, "分类失败", f"错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def _get_num_classes(self, dataset):
        """根据数据集名称获取类别数"""
        if dataset == 'Cifar10':
            return 10
        elif dataset == 'MNIST':
            return 10
        elif dataset == 'Omniglot':
            return 1623
        elif dataset == 'Digit5':
            return 10
        elif dataset == 'HAR':
            return 6
        elif dataset == 'PAMAP2':
            return 12
        elif dataset == 'Shakespeare':
            return 80
        else:
            return 10  # 默认

    def on_load_model(self):
        """加载模型文件并解析信息"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch (*.pt *.pth);;All Files (*)")
        if path:
            try:
                # 导入model_predict中的函数
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from model_predict import parse_model_info_from_path, load_pytorch_model, create_model
                
                # 解析模型信息
                dataset, model_str, acc = parse_model_info_from_path(path)
                if dataset is None:
                    QMessageBox.warning(self, "警告", "无法从文件名推断数据集，请手动输入数据集名称。")
                    dataset, ok = QInputDialog.getText(self, "输入数据集", "请输入数据集名称 (如 Cifar10, MNIST):")
                    if not ok or not dataset:
                        return
                if model_str is None:
                    QMessageBox.warning(self, "警告", "无法从文件名推断模型类型，请手动输入模型类型。")
                    model_str, ok = QInputDialog.getText(self, "输入模型类型", "请输入模型类型 (如 MobileNet, ResNet18, CNN):")
                    if not ok or not model_str:
                        return
                
                # 获取类别数
                num_classes = self._get_num_classes(dataset)
                
                # 加载模型
                model = load_pytorch_model(path, model_str, dataset, num_classes, self.device)
                
                # 保存到属性
                self.loaded_model = model
                self.loaded_model_path = path
                self.loaded_model_str = model_str
                self.loaded_dataset = dataset
                
                # 设置类别名称（根据数据集）
                self.class_names = self._get_class_names(dataset)
                
                # 更新状态标签
                if hasattr(self, 'model_status'):
                    self.model_status.setText(f"模型: {model_str} ({dataset})")
                    self.model_status.setStyleSheet("color: #27ae60; font-weight: bold;")
                
                # 显示成功信息
                acc_text = f"，准确率: {acc:.4f}" if acc else ""
                self.result_text.setPlainText(f"""模型加载成功!
文件: {os.path.basename(path)}
数据集: {dataset}
模型类型: {model_str}{acc_text}
类别数: {num_classes}
设备: {self.device}

请选择图片后点击'开始分类'""")
                
            except Exception as e:
                QMessageBox.critical(self, "加载失败", f"错误: {str(e)}")
                import traceback
                traceback.print_exc()

    def _get_class_names(self, dataset):
        """根据数据集返回类别名称列表"""
        if dataset == 'Cifar10':
            return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif dataset == 'MNIST':
            return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        else:
            return None  # 未知数据集


# ============================================================
# 主窗口 — 紧凑版：配置 + 进度 + 日志
# ============================================================
class FedProxApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.current_round = 0
        self.total_rounds = 100
        self.training_failed = False
        self.current_acc = 0.0
        self.current_loss = 0.0
        # 收集数据供结果窗口使用
        self.collected_rounds = []
        self.collected_accs = []
        self.collected_losses = []
        # AUC / Std 统计信息（从日志解析）
        self.auc_info = {}
        # 窗口引用
        self.result_win = None
        self.classify_win = None

        self.init_ui()
        self._detect_device()

    def init_ui(self):
        self.setWindowTitle("联邦学习客户端协同训练平台")
        self.setGeometry(100, 100, 1150, 720)
        self.setStyleSheet(GLOBAL_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setSpacing(8)
        main.setContentsMargins(15, 12, 15, 12)

        # ===== 标题 =====
        title = QLabel("联邦学习客户端协同训练平台")
        title.setFont(QFont("", FONT_SIZE_TITLE, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1a1a2e; font-size: {}px; font-weight: bold; margin-bottom: 4px;".format(FONT_SIZE_TITLE))
        main.addWidget(title)

        # ===== 配置区 =====
        cfg_grp = QGroupBox("训练配置")
        cfg_lyt = QVBoxLayout()

        # 第一行：左右对称各3个参数（全部用 QComboBox）
        row1 = QHBoxLayout()

        left_form = QFormLayout()
        left_form.setSpacing(10)
        left_form.setLabelAlignment(Qt.AlignRight)

        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["FedProxV2", "FedProx", "FedAvg"])
        left_form.addRow("算法:", self.algo_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["ResNet18", "MobileNet", "CNN", "DNN"])
        left_form.addRow("模型:", self.model_combo)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Cifar10", "MNIST"])
        left_form.addRow("数据集:", self.dataset_combo)

        row1.addLayout(left_form, 1)
        row1.addSpacing(30)

        right_form = QFormLayout()
        right_form.setSpacing(10)
        right_form.setLabelAlignment(Qt.AlignRight)

        self.clients_combo = QComboBox()
        self.clients_combo.setEditable(True)
        self.clients_combo.addItems(["1", "2", "5", "10"])
        self.clients_combo.setCurrentIndex(1)
        right_form.addRow("客户端数:", self.clients_combo)

        self.lr_combo = QComboBox()
        self.lr_combo.setEditable(True)
        self.lr_combo.addItems(["0.01", "0.05", "0.1", "0.001"])
        self.lr_combo.setCurrentIndex(1)
        right_form.addRow("学习率:", self.lr_combo)

        self.mu_combo = QComboBox()
        self.mu_combo.setEditable(True)
        self.mu_combo.addItems(["0", "0.01", "0.1", "1.0"])
        self.mu_combo.setCurrentIndex(2)
        right_form.addRow("\u03bc (FedProx):", self.mu_combo)

        row1.addLayout(right_form, 1)
        cfg_lyt.addLayout(row1)

        # 第二行：辅助参数
        row2 = QHBoxLayout()
        row2.setSpacing(18)

        row2.addWidget(QLabel("全局轮次:"))
        self.rounds_input = QLineEdit("100")
        self.rounds_input.setMinimumWidth(85)
        row2.addWidget(self.rounds_input)

        row2.addWidget(QLabel("设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu"])  # 默认，后续检测覆盖
        self.device_combo.setMinimumWidth(75)
        row2.addWidget(self.device_combo)

        row2.addWidget(QLabel("批次大小:"))
        self.batch_input = QLineEdit("64")
        self.batch_input.setMinimumWidth(65)
        row2.addWidget(self.batch_input)

        row2.addWidget(QLabel("本地轮次:"))
        self.local_epochs_input = QLineEdit("3")
        self.local_epochs_input.setMinimumWidth(65)
        row2.addWidget(self.local_epochs_input)

        row2.addWidget(QLabel("参与率:"))
        self.join_ratio_input = QLineEdit("1")
        self.join_ratio_input.setMinimumWidth(65)
        row2.addWidget(self.join_ratio_input)

        row2.addStretch()
        cfg_lyt.addLayout(row2)
        cfg_grp.setLayout(cfg_lyt)
        main.addWidget(cfg_grp)

        # ===== 按钮行（独立，不在任何框内）=====
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.start_btn = QPushButton("开始训练")
        self.start_btn.clicked.connect(self.start_training)
        btn_row.addWidget(self.start_btn)

        btn_row.addSpacing(22)
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        btn_row.addWidget(self.stop_btn)

        btn_row.addSpacing(22)
        self.clear_btn = QPushButton("清空日志")
        self.clear_btn.setObjectName("clear_btn")
        self.clear_btn.clicked.connect(self.clear_log)
        btn_row.addWidget(self.clear_btn)

        btn_row.addSpacing(30)

        self.classify_btn = QPushButton("农作物分类")
        self.classify_btn.setObjectName("result_btn")
        self.classify_btn.clicked.connect(self.open_classify_window)
        btn_row.addWidget(self.classify_btn)

        btn_row.addStretch()
        main.addLayout(btn_row)

        # ===== 状态 + 进度条（紧凑）=====
        stat_grp = QGroupBox("训练状态")
        stat_lyt = QHBoxLayout()

        self.status_label = QLabel("状态: 就绪")
        self.status_label.setStyleSheet("color: #27ae60; font-weight:bold; font-size: 13px;")
        stat_lyt.addWidget(self.status_label)
        stat_lyt.addSpacing(25)

        self.round_label = QLabel("轮次: 0/0")
        self.round_label.setStyleSheet("font-size: 13px; font-weight:bold;")
        stat_lyt.addWidget(self.round_label)
        stat_lyt.addSpacing(15)

        self.acc_label = QLabel("准确率: ---%")
        self.acc_label.setStyleSheet("font-size: 13px; font-weight:bold;")
        stat_lyt.addWidget(self.acc_label)
        stat_lyt.addSpacing(15)

        self.loss_label = QLabel("损失: ---")
        self.loss_label.setStyleSheet("font-size: 13px; font-weight:bold;")
        stat_lyt.addWidget(self.loss_label)
        stat_lyt.addStretch()
        stat_grp.setLayout(stat_lyt)
        main.addWidget(stat_grp)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v%")
        main.addWidget(self.progress_bar)

        # ===== 日志区 =====
        log_grp = QGroupBox("运行日志")
        log_lyt = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(160)
        self.log_text.setMaximumHeight(220)
        self.log_text.setStyleSheet(
            "background-color: #1e1e1e; color: #00ff00; "
            "padding: 6px; border-radius: 4px;")
        log_lyt.addWidget(self.log_text)
        log_grp.setLayout(log_lyt)
        main.addWidget(log_grp)

    def _detect_device(self):
        """检测 CUDA 是否可用，动态设置设备选项"""
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        self.device_combo.clear()
        if has_cuda:
            self.device_combo.addItems(["cuda", "cpu"])
            self.device_combo.setCurrentText("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            self.append_log("[系统] CUDA 可用，默认设备: cuda | GPU: {}".format(gpu_name))
        else:
            self.device_combo.addItem("cpu")
            self.device_combo.setCurrentText("cpu")
            self.append_log("[系统] CUDA 不可用，仅 CPU 模式")

    # ---- 日志 & 解析 ----

    def append_log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
        self.parse_training_info(msg)

    def parse_training_info(self, msg):
        try:
            flag = False

            # 错误检测
            if re.search(r'(?:Traceback|Error|Exception|error|failed)',
                         msg, re.IGNORECASE):
                self.training_failed = True

            rm = re.search(r'Round\s*number:\s*(\d+)', msg, re.IGNORECASE)
            if rm:
                self.current_round = int(rm.group(1))
                self.round_label.setText(
                    f"轮次: {self.current_round}/{self.total_rounds}")
                pct = int((self.current_round / self.total_rounds) * 100)
                self.progress_bar.setValue(pct)

            # 优先匹配 "Best accuracy" (真实最终准确率)
            bm = re.search(r'(?:Best\s+accuracy)[^0-9]*(\d+\.\d+)',
                           msg, re.IGNORECASE)
            if bm:
                av = float(bm.group(1))
                self.current_acc = av * 100 if av <= 1.0 else av
                self.acc_label.setText(f"准确率: {self.current_acc:.2f}%")
                flag = True

            # 其次匹配普通 Accuracy
            if not bm:
                am = re.search(r'(?:Accuracy|acc|accuracy).*?(\d+\.\d+)',
                               msg, re.IGNORECASE)
                if am:
                    av = float(am.group(1))
                    # 过滤掉明显不对的值（如 loss 值被误匹配）
                    if av <= 100:
                        self.current_acc = av * 100 if av <= 1.0 else av
                        self.acc_label.setText(f"准确率: {self.current_acc:.2f}%")
                        flag = True

            lm = re.search(r'(?:loss|Loss).*?(\d+\.\d+)',
                           msg, re.IGNORECASE)
            if lm:
                self.current_loss = float(lm.group(1))
                self.loss_label.setText(f"损失: {self.current_loss:.3f}")
                flag = True

            # AUC / Std 统计信息
            auc_m = re.search(r'(?:Averaged\s+Test\s+AUC|Avg.*AUC)[^0-9]*(\d+\.\d+)',
                              msg, re.IGNORECASE)
            if auc_m:
                self.auc_info['avg_auc'] = float(auc_m.group(1))

            std_acc_m = re.search(r'(?:Std\s+Test\s+Accuracy|Std.*Acc)[^0-9]*(\d+\.\d+)',
                                 msg, re.IGNORECASE)
            if std_acc_m:
                self.auc_info['std_acc'] = float(std_acc_m.group(1))

            std_auc_m = re.search(r'(?:Std\s+Test\s+AUC|Std.*AUC)[^0-9]*(\d+\.\d+)',
                                 msg, re.IGNORECASE)
            if std_auc_m:
                self.auc_info['std_auc'] = float(std_auc_m.group(1))

            # 收集数据点（仅当有 Best accuracy 或正常 Accuracy 时）
            if flag and self.current_round > 0:
                self._collect_point()
        except Exception:
            pass

    def _collect_point(self):
        """收集数据点供结果窗口使用"""
        r = self.current_round
        if not self.collected_rounds or self.collected_rounds[-1] != r:
            self.collected_rounds.append(r)
            self.collected_accs.append(self.current_acc)
            self.collected_losses.append(self.current_loss)
        else:
            self.collected_accs[-1] = self.current_acc
            self.collected_losses[-1] = self.current_loss

    # ---- 训练控制 ----

    def start_training(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("状态: 训练中...")
        self.status_label.setStyleSheet("color: #e67e22; font-weight:bold;")
        self.training_failed = False

        self.total_rounds = int(self.rounds_input.text())
        self.current_round = 0
        self.current_acc = 0.0
        self.current_loss = 0.0
        self.progress_bar.setValue(0)

        # 清空收集的数据
        self.collected_rounds.clear()
        self.collected_accs.clear()
        self.collected_losses.clear()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_v2 = os.path.join(script_dir, "main_v2.py")

        args = [
            sys.executable, main_v2,
            "-algo", self.algo_combo.currentText(),
            "-m", self.model_combo.currentText(),
            "-data", self.dataset_combo.currentText(),
            "-gr", self.rounds_input.text(),
            "-nc", self.clients_combo.currentText(),
            "-lr", self.lr_combo.currentText(),
            "-lbs", self.batch_input.text(),
            "-dev", self.device_combo.currentText(),
            "-ls", self.local_epochs_input.text(),
            "-jr", self.join_ratio_input.text(),
            "-eg", "1",
            "-mu", self.mu_combo.currentText(),
            "-fs", "0",
        ]

        self.config_info = {
            "算法": self.algo_combo.currentText(),
            "模型": self.model_combo.currentText(),
            "数据集": self.dataset_combo.currentText(),
            "全局轮次": self.rounds_input.text(),
            "客户端数": self.clients_combo.currentText(),
            "学习率": self.lr_combo.currentText(),
            "\u03bc (FedProx)": self.mu_combo.currentText(),
            "设备": self.device_combo.currentText(),
            "批次大小": self.batch_input.text(),
            "本地轮次": self.local_epochs_input.text(),
        }

        self.append_log(f"执行命令: {' '.join(args)}")

        self.training_thread = TrainingThread(args, script_dir)
        self.training_thread.log_signal.connect(self.append_log)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()

    def stop_training(self):
        if self.training_thread:
            self.training_thread.stop()
            self.training_thread = None
        self.status_label.setText("状态: 已停止")
        self.status_label.setStyleSheet("color: #e74c3c; font-weight:bold;")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.append_log(">>> 已停止训练")

    def training_finished(self, exit_code):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if self.training_failed or exit_code != 0:
            self.status_label.setText("状态: 训练失败")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight:bold;")
            self.append_log("=" * 40)
            self.append_log("⚠ 训练异常终止！请查看上方日志排查错误。")
            self.append_log("=" * 40)
        else:
            self.status_label.setText("状态: 完成")
            self.status_label.setStyleSheet("color: #27ae60; font-weight:bold;")
            self.append_log("=" * 40)
            self.append_log("✅ 训练成功完成！正在打开结果分析窗口...")
            self.append_log("=" * 40)
            self.progress_bar.setValue(100)

            # 自动打开结果窗口
            QTimer.singleShot(500, self.open_result_window)

    # ---- 子窗口 ----

    def open_result_window(self):
        """打开结果分析窗口"""
        if self.result_win is None or not self.result_win.isVisible():
            self.result_win = ResultWindow(
                self.collected_rounds[:],
                self.collected_accs[:],
                self.collected_losses[:],
                self.total_rounds,
                dict(self.config_info),
                auc_info=dict(self.auc_info),
                parent=self
            )
            self.result_win.show()
        else:
            self.result_win.activateWindow()
            self.result_win.raise_()

    def open_classify_window(self):
        """打开分类窗口"""
        if self.classify_win is None or not self.classify_win.isVisible():
            self.classify_win = ClassifyWindow(parent=self)
            self.classify_win.show()
        else:
            self.classify_win.activateWindow()
            self.classify_win.raise_()

    def clear_log(self):
        self.log_text.clear()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FedProxApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

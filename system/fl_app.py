#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习农作物识别可视化系统 (FL-Crop Vis)
基于 PyQt5，集成训练监控 + 图片识别 + 参数配置
"""

import sys
import os

# 确保能导入同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import cv2
from collections import OrderedDict

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QFileDialog, QTextEdit, QStackedWidget, QListWidget, QListWidgetItem,
    QGroupBox, QFormLayout, QSplitter, QProgressBar, QMessageBox,
    QScrollArea, QFrame, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QIcon, QPainter, QPen

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')

# ============================================================
# 全局共享训练指标（训练线程写入，UI线程读取）
# ============================================================
train_metrics = {
    'rounds': [],
    'test_acc': [],
    'train_loss': [],
    'test_auc': [],
    'current_round': 0,
    'total_rounds': 0,
    'best_acc': 0.0,
    'is_training': False,
    'log': '',
}

CLASSES = ["wheat", "corn", "rice"]
CLASS_NAMES_CN = {"wheat": "小麦", "corn": "玉米", "rice": "水稻"}
CLASS_COLORS = {
    "wheat": "#4CAF50",  # 绿色
    "corn": "#FF9800",   # 橙色
    "rice": "#2196F3",   # 蓝色
}

# 配色方案
COLORS = {
    'bg_dark': '#1e1e2e',
    'bg_medium': '#2a2a3d',
    'bg_light': '#363650',
    'bg_card': '#404060',
    'accent': '#4CAF50',
    'accent_light': '#66BB6A',
    'text_primary': '#E0E0E0',
    'text_secondary': '#9E9E9E',
    'text_highlight': '#FFFFFF',
    'danger': '#EF5350',
    'warning': '#FFC107',
    'info': '#42A5F5',
    'border': '#4a4a6a',
}


# ============================================================
# 样式表
# ============================================================
STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_primary']};
}}
QWidget {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_primary']};
    font-family: "Microsoft YaHei", "SimHei", sans-serif;
}}
QLabel {{
    color: {COLORS['text_primary']};
    background: transparent;
}}
QPushButton {{
    background-color: {COLORS['accent']};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-size: 13px;
    font-weight: bold;
    min-height: 32px;
}}
QPushButton:hover {{
    background-color: {COLORS['accent_light']};
}}
QPushButton:pressed {{
    background-color: #388E3C;
}}
QPushButton:disabled {{
    background-color: {COLORS['bg_light']};
    color: {COLORS['text_secondary']};
}}
QPushButton#btn_danger {{
    background-color: {COLORS['danger']};
}}
QPushButton#btn_danger:hover {{
    background-color: #E53935;
}}
QPushButton#btn_outline {{
    background-color: transparent;
    border: 1px solid {COLORS['accent']};
    color: {COLORS['accent']};
}}
QPushButton#btn_outline:hover {{
    background-color: rgba(76, 175, 80, 0.15);
}}
QGroupBox {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px;
    padding-top: 24px;
    font-weight: bold;
    font-size: 13px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 8px;
    color: {COLORS['accent']};
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 6px 10px;
    color: {COLORS['text_primary']};
    min-height: 28px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {COLORS['accent']};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_primary']};
    selection-background-color: {COLORS['accent']};
}}
QTextEdit {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 8px;
    color: {COLORS['text_secondary']};
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
}}
QListWidget {{
    background-color: {COLORS['bg_medium']};
    border: none;
    border-radius: 8px;
    padding: 8px;
    outline: none;
}}
QListWidget::item {{
    padding: 12px 16px;
    border-radius: 6px;
    margin: 2px 4px;
    color: {COLORS['text_secondary']};
    font-size: 13px;
}}
QListWidget::item:selected {{
    background-color: rgba(76, 175, 80, 0.2);
    color: {COLORS['accent']};
    font-weight: bold;
}}
QListWidget::item:hover {{
    background-color: rgba(76, 175, 80, 0.1);
    color: {COLORS['text_primary']};
}}
QProgressBar {{
    background-color: {COLORS['bg_light']};
    border-radius: 4px;
    text-align: center;
    color: white;
    min-height: 20px;
}}
QProgressBar::chunk {{
    background-color: {COLORS['accent']};
    border-radius: 4px;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QSplitter::handle {{
    background-color: {COLORS['border']};
    width: 1px;
}}
"""


# ============================================================
# 自定义组件
# ============================================================

class StatCard(QFrame):
    """数据统计卡片"""
    def __init__(self, title, value="0", color=COLORS['accent'], parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            StatCard {{
                background-color: {COLORS['bg_card']};
                border-radius: 10px;
                border: 1px solid {COLORS['border']};
                padding: 12px;
            }}
        """)
        self.setMinimumSize(140, 80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(self.title_label)

        self.value_label = QLabel(str(value))
        self.value_label.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold; background: transparent; border: none;")
        layout.addWidget(self.value_label)

    def set_value(self, value):
        self.value_label.setText(str(value))


class MplCanvas(FigureCanvas):
    """Matplotlib 嵌入 PyQt5 的画布"""
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor(COLORS['bg_medium'])
        self.axes = self.fig.add_subplot(111)
        self.setup_axes()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setup_axes(self):
        self.axes.set_facecolor(COLORS['bg_dark'])
        self.axes.tick_params(colors=COLORS['text_secondary'], labelsize=9)
        self.axes.spines['bottom'].set_color(COLORS['border'])
        self.axes.spines['left'].set_color(COLORS['border'])
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.xaxis.label.set_color(COLORS['text_secondary'])
        self.axes.yaxis.label.set_color(COLORS['text_secondary'])
        self.axes.title.set_color(COLORS['text_primary'])
        self.axes.grid(True, alpha=0.15, color=COLORS['text_secondary'])


class ImageDropLabel(QLabel):
    """支持拖拽上传图片的标签"""
    image_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 250)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_light']};
                border: 2px dashed {COLORS['border']};
                border-radius: 12px;
                color: {COLORS['text_secondary']};
                font-size: 14px;
            }}
        """)
        self.setText("拖拽图片到此处\n或点击下方按钮选择")
        self.image_path = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.image_path = path
                self.display_image(path)
                self.image_dropped.emit(path)

    def display_image(self, path):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)
            self.setStyleSheet(f"""
                QLabel {{
                    background-color: {COLORS['bg_light']};
                    border: 2px solid {COLORS['accent']};
                    border-radius: 12px;
                }}
            """)

    def resizeEvent(self, event):
        if self.image_path:
            self.display_image(self.image_path)
        super().resizeEvent(event)


class ProbabilityBar(QWidget):
    """概率分布条形图"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.bars = {}
        for cls_name, cls_cn in CLASS_NAMES_CN.items():
            row = QHBoxLayout()
            name_label = QLabel(f"{cls_cn} ({cls_name})")
            name_label.setFixedWidth(120)
            name_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 12px; background: transparent;")

            bar_bg = QFrame()
            bar_bg.setFixedHeight(24)
            bar_bg.setStyleSheet(f"background-color: {COLORS['bg_light']}; border-radius: 4px;")

            bar_fill = QFrame(bar_bg)
            bar_fill.setFixedHeight(24)
            bar_fill.setGeometry(0, 0, 0, 24)
            bar_fill.setStyleSheet(f"background-color: {CLASS_COLORS[cls_name]}; border-radius: 4px;")

            pct_label = QLabel("0.0%")
            pct_label.setFixedWidth(60)
            pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pct_label.setStyleSheet(f"color: {CLASS_COLORS[cls_name]}; font-size: 13px; font-weight: bold; background: transparent;")

            row.addWidget(name_label)
            row.addWidget(bar_bg, 1)
            row.addWidget(pct_label)
            layout.addLayout(row)
            self.bars[cls_name] = (bar_bg, bar_fill, pct_label)

    def update_probs(self, probs_dict):
        max_w = 0
        for cls_name, (bar_bg, bar_fill, pct_label) in self.bars.items():
            if bar_bg.width() > max_w:
                max_w = bar_bg.width()
        for cls_name, (bar_bg, bar_fill, pct_label) in self.bars.items():
            prob = probs_dict.get(cls_name, 0.0)
            bar_fill.setGeometry(0, 0, int(max_w * prob), 24)
            pct_label.setText(f"{prob*100:.1f}%")


# ============================================================
# 训练线程
# ============================================================

class TrainThread(QThread):
    """在子线程中运行联邦学习训练"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, args_dict):
        super().__init__()
        self.args_dict = args_dict

    def run(self):
        train_metrics['is_training'] = True
        train_metrics['rounds'] = []
        train_metrics['test_acc'] = []
        train_metrics['train_loss'] = []
        train_metrics['test_auc'] = []

        try:
            import argparse
            from main import run

            # 构建 args
            args = argparse.Namespace(**self.args_dict)

            # Hook evaluate 函数来捕获指标
            from flcore.servers.serverbase import Server
            original_evaluate = Server.evaluate

            def hooked_evaluate(self_server, acc=None, loss=None):
                original_evaluate(self_server, acc, loss)
                if self_server.rs_test_acc:
                    current_acc = self_server.rs_test_acc[-1]
                    current_loss = self_server.rs_train_loss[-1]
                    current_auc = self_server.rs_test_auc[-1]
                    round_num = len(self_server.rs_test_acc) - 1

                    train_metrics['rounds'].append(round_num)
                    train_metrics['test_acc'].append(current_acc)
                    train_metrics['train_loss'].append(current_loss)
                    train_metrics['test_auc'].append(current_auc)
                    train_metrics['current_round'] = round_num
                    train_metrics['best_acc'] = max(train_metrics['best_acc'], current_acc)

                    self.log_signal.emit(
                        f"Round {round_num}: Acc={current_acc:.4f}, Loss={current_loss:.4f}, AUC={current_auc:.4f}"
                    )

            Server.evaluate = hooked_evaluate
            run(args)
            Server.evaluate = original_evaluate

            self.finished_signal.emit(True, "训练完成！")
        except Exception as e:
            self.finished_signal.emit(False, f"训练出错: {str(e)}")
        finally:
            train_metrics['is_training'] = False


# ============================================================
# 预测线程
# ============================================================

class PredictThread(QThread):
    """在子线程中运行图片预测"""
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, image_path, model_path, model_name, device):
        super().__init__()
        self.image_path = image_path
        self.model_path = model_path
        self.model_name = model_name
        self.device = device

    def run(self):
        try:
            import torchvision
            import torch.nn as nn

            # 构建模型
            if self.model_name == "ResNet18":
                model = torchvision.models.resnet18(pretrained=False, num_classes=len(CLASSES))
                # BN -> GN
                def replace_bn(module):
                    for name, child in module.named_children():
                        if isinstance(child, nn.BatchNorm2d):
                            setattr(module, name, nn.GroupNorm(2, child.num_features))
                        else: replace_bn(child)
                replace_bn(model)
            elif self.model_name == "MobileNet":
                from flcore.trainmodel.mobilenet_v2 import mobilenet_v2
                model = mobilenet_v2(pretrained=False, num_classes=len(CLASSES))
            else:
                model = torchvision.models.resnet18(pretrained=False, num_classes=len(CLASSES))

            # 加载权重
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()

            # 预处理
            img = cv2.imread(self.image_path)
            if img is None:
                self.error_signal.emit(f"无法读取图片: {self.image_path}")
                return

            img_ori = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64, 64))
            img = img / 255.0
            img_tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # 推理
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                pred_cls = CLASSES[pred_idx]
                confidence = probs[0][pred_idx].item()

                probs_dict = {}
                for i, cls in enumerate(CLASSES):
                    probs_dict[cls] = probs[0][i].item()

            # 绘制结果
            label_text = f"{CLASS_NAMES_CN[pred_cls]}({pred_cls}): {confidence*100:.1f}%"
            color_map = {"wheat": (0,255,0), "corn": (255,165,0), "rice": (0,100,255)}
            cv2.putText(img_ori, label_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_map[pred_cls], 2)

            # 转QPixmap
            img_rgb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            result_pixmap = QPixmap.fromImage(q_img)

            self.result_signal.emit({
                'pred_cls': pred_cls,
                'pred_cls_cn': CLASS_NAMES_CN[pred_cls],
                'confidence': confidence,
                'probs': probs_dict,
                'result_pixmap': result_pixmap,
            })
        except Exception as e:
            self.error_signal.emit(f"预测出错: {str(e)}")


# ============================================================
# 页面：训练监控
# ============================================================

class TrainMonitorPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.train_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)

        # 顶部控制栏
        ctrl_layout = QHBoxLayout()

        self.btn_start = QPushButton("开始训练")
        self.btn_start.clicked.connect(self.start_training)

        self.btn_stop = QPushButton("停止训练")
        self.btn_stop.setObjectName("btn_danger")
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)

        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet(f"color: {COLORS['text_secondary']}; background: transparent;")

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(200)

        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.lbl_status)
        ctrl_layout.addWidget(self.progress)
        layout.addLayout(ctrl_layout)

        # 统计卡片
        cards_layout = QHBoxLayout()
        self.card_round = StatCard("当前轮次", "0", COLORS['info'])
        self.card_best = StatCard("最佳精度", "0.0000", COLORS['accent'])
        self.card_loss = StatCard("训练损失", "-", COLORS['warning'])
        self.card_auc = StatCard("测试AUC", "-", COLORS['info'])
        cards_layout.addWidget(self.card_round)
        cards_layout.addWidget(self.card_best)
        cards_layout.addWidget(self.card_loss)
        cards_layout.addWidget(self.card_auc)
        layout.addLayout(cards_layout)

        # 曲线区域
        charts_layout = QHBoxLayout()

        # 精度曲线
        acc_group = QGroupBox("测试精度 (Accuracy)")
        acc_layout = QVBoxLayout(acc_group)
        self.acc_canvas = MplCanvas(self, width=5, height=3)
        self.acc_canvas.axes.set_xlabel("Round")
        self.acc_canvas.axes.set_ylabel("Accuracy")
        acc_layout.addWidget(self.acc_canvas)
        charts_layout.addWidget(acc_group)

        # 损失曲线
        loss_group = QGroupBox("训练损失 (Loss)")
        loss_layout = QVBoxLayout(loss_group)
        self.loss_canvas = MplCanvas(self, width=5, height=3)
        self.loss_canvas.axes.set_xlabel("Round")
        self.loss_canvas.axes.set_ylabel("Loss")
        loss_layout.addWidget(self.loss_canvas)
        charts_layout.addWidget(loss_group)

        layout.addLayout(charts_layout, 1)

        # 训练日志
        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(140)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        # 定时刷新
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_charts)
        self.refresh_timer.start(2000)

    def start_training(self):
        if train_metrics['is_training']:
            return
        # 从 SettingsPage 读取参数（通过主窗口）
        main_window = self.window()
        args_dict = main_window.get_train_args()
        if not args_dict:
            QMessageBox.warning(self, "参数错误", "请先在参数设置页配置训练参数")
            return

        train_metrics['best_acc'] = 0.0
        train_metrics['current_round'] = 0
        train_metrics['total_rounds'] = args_dict.get('global_rounds', 100)

        self.train_thread = TrainThread(args_dict)
        self.train_thread.log_signal.connect(self.append_log)
        self.train_thread.finished_signal.connect(self.on_train_finished)
        self.train_thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("训练中...")
        self.lbl_status.setStyleSheet(f"color: {COLORS['accent']}; background: transparent;")

    def stop_training(self):
        if self.train_thread and self.train_thread.isRunning():
            self.train_thread.terminate()
            self.train_thread.wait()
            train_metrics['is_training'] = False
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.lbl_status.setText("已停止")
            self.lbl_status.setStyleSheet(f"color: {COLORS['danger']}; background: transparent;")
            self.append_log("训练已被手动停止")

    def on_train_finished(self, success, msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if success:
            self.lbl_status.setText("训练完成")
            self.lbl_status.setStyleSheet(f"color: {COLORS['accent']}; background: transparent;")
        else:
            self.lbl_status.setText("训练异常")
            self.lbl_status.setStyleSheet(f"color: {COLORS['danger']}; background: transparent;")
        self.append_log(msg)
        self.refresh_charts()

    def append_log(self, text):
        self.log_text.append(text)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def refresh_charts(self):
        if not train_metrics['rounds']:
            return

        rounds = train_metrics['rounds']
        acc = train_metrics['test_acc']
        loss = train_metrics['train_loss']
        auc = train_metrics['test_auc']

        # 更新卡片
        self.card_round.set_value(str(train_metrics['current_round']))
        self.card_best.set_value(f"{train_metrics['best_acc']:.4f}")
        if loss:
            self.card_loss.set_value(f"{loss[-1]:.4f}")
        if auc:
            self.card_auc.set_value(f"{auc[-1]:.4f}")

        # 更新进度条
        total = train_metrics['total_rounds']
        if total > 0:
            self.progress.setValue(int(train_metrics['current_round'] / total * 100))

        # 精度曲线
        self.acc_canvas.axes.clear()
        self.acc_canvas.setup_axes()
        self.acc_canvas.axes.plot(rounds, acc, color=COLORS['accent'], linewidth=2, marker='o', markersize=3)
        self.acc_canvas.axes.set_xlabel("Round")
        self.acc_canvas.axes.set_ylabel("Accuracy")
        self.acc_canvas.axes.set_title("Test Accuracy")
        self.acc_canvas.fig.tight_layout()
        self.acc_canvas.draw()

        # 损失曲线
        self.loss_canvas.axes.clear()
        self.loss_canvas.setup_axes()
        self.loss_canvas.axes.plot(rounds, loss, color=COLORS['warning'], linewidth=2, marker='o', markersize=3)
        self.loss_canvas.axes.set_xlabel("Round")
        self.loss_canvas.axes.set_ylabel("Loss")
        self.loss_canvas.axes.set_title("Train Loss")
        self.loss_canvas.fig.tight_layout()
        self.loss_canvas.draw()


# ============================================================
# 页面：图片识别
# ============================================================

class PredictPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.predict_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)

        # 标题
        title = QLabel("农作物图片识别")
        title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {COLORS['text_highlight']}; background: transparent;")
        layout.addWidget(title)

        # 模型选择栏
        model_layout = QHBoxLayout()
        model_label = QLabel("选择模型:")
        model_label.setStyleSheet("background: transparent;")

        self.model_combo = QComboBox()
        self.model_combo.addItems(["ResNet18", "MobileNet"])
        self.model_combo.setFixedWidth(160)

        self.btn_model_path = QPushButton("选择模型文件")
        self.btn_model_path.setObjectName("btn_outline")
        self.btn_model_path.clicked.connect(self.select_model)

        self.lbl_model_path = QLabel("未选择")
        self.lbl_model_path.setStyleSheet(f"color: {COLORS['text_secondary']}; background: transparent;")

        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addSpacing(20)
        model_layout.addWidget(self.btn_model_path)
        model_layout.addWidget(self.lbl_model_path, 1)
        layout.addLayout(model_layout)

        # 主内容区
        content_layout = QHBoxLayout()

        # 左侧：图片上传
        img_group = QGroupBox("上传图片")
        img_layout = QVBoxLayout(img_group)
        self.image_label = ImageDropLabel()
        img_layout.addWidget(self.image_label)

        self.btn_select_img = QPushButton("选择图片")
        self.btn_select_img.setObjectName("btn_outline")
        self.btn_select_img.clicked.connect(self.select_image)

        self.btn_predict = QPushButton("开始识别")
        self.btn_predict.clicked.connect(self.predict)

        img_btn_layout = QHBoxLayout()
        img_btn_layout.addWidget(self.btn_select_img)
        img_btn_layout.addWidget(self.btn_predict)
        img_layout.addLayout(img_btn_layout)
        content_layout.addWidget(img_group, 1)

        # 右侧：识别结果
        result_group = QGroupBox("识别结果")
        result_layout = QVBoxLayout(result_group)

        self.lbl_result_img = QLabel("等待识别...")
        self.lbl_result_img.setAlignment(Qt.AlignCenter)
        self.lbl_result_img.setMinimumSize(300, 250)
        self.lbl_result_img.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_light']};
                border: 2px dashed {COLORS['border']};
                border-radius: 12px;
                color: {COLORS['text_secondary']};
                font-size: 13px;
            }}
        """)
        result_layout.addWidget(self.lbl_result_img, 1)

        # 预测类别
        self.lbl_pred_class = QLabel("")
        self.lbl_pred_class.setAlignment(Qt.AlignCenter)
        self.lbl_pred_class.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {COLORS['accent']}; background: transparent;")
        result_layout.addWidget(self.lbl_pred_class)

        # 概率分布
        self.prob_bar = ProbabilityBar()
        result_layout.addWidget(self.prob_bar)
        content_layout.addWidget(result_group, 1)

        layout.addLayout(content_layout, 1)

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Model Files (*.pt *.pth);;All Files (*)")
        if path:
            self.lbl_model_path.setText(path)
            self.lbl_model_path.setStyleSheet(f"color: {COLORS['accent']}; background: transparent;")

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)")
        if path:
            self.image_label.image_path = path
            self.image_label.display_image(path)

    def predict(self):
        if not self.image_label.image_path:
            QMessageBox.warning(self, "提示", "请先上传图片")
            return

        model_path = self.lbl_model_path.text()
        if model_path == "未选择":
            # 尝试自动查找
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
            if os.path.exists(save_dir):
                models = [f for f in os.listdir(save_dir) if f.endswith(('.pt', '.pth'))]
                if models:
                    models.sort(key=lambda x: os.path.getmtime(os.path.join(save_dir, x)), reverse=True)
                    model_path = os.path.join(save_dir, models[0])
                    self.lbl_model_path.setText(model_path)
                else:
                    QMessageBox.warning(self, "提示", "未找到模型文件，请手动选择")
                    return
            else:
                QMessageBox.warning(self, "提示", "请先选择模型文件")
                return

        model_name = self.model_combo.currentText()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.btn_predict.setEnabled(False)
        self.lbl_pred_class.setText("识别中...")
        self.lbl_pred_class.setStyleSheet(f"font-size: 18px; color: {COLORS['warning']}; background: transparent;")

        self.predict_thread = PredictThread(self.image_label.image_path, model_path, model_name, device)
        self.predict_thread.result_signal.connect(self.on_predict_result)
        self.predict_thread.error_signal.connect(self.on_predict_error)
        self.predict_thread.start()

    def on_predict_result(self, result):
        self.btn_predict.setEnabled(True)

        pred_cls = result['pred_cls']
        pred_cn = result['pred_cls_cn']
        confidence = result['confidence']

        self.lbl_pred_class.setText(f"{pred_cn} ({pred_cls})")
        self.lbl_pred_class.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {CLASS_COLORS[pred_cls]}; background: transparent;")

        # 显示结果图
        pixmap = result['result_pixmap']
        scaled = pixmap.scaled(self.lbl_result_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_result_img.setPixmap(scaled)
        self.lbl_result_img.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_light']};
                border: 2px solid {CLASS_COLORS[pred_cls]};
                border-radius: 12px;
            }}
        """)

        # 更新概率条
        self.prob_bar.update_probs(result['probs'])

    def on_predict_error(self, msg):
        self.btn_predict.setEnabled(True)
        self.lbl_pred_class.setText("识别失败")
        self.lbl_pred_class.setStyleSheet(f"font-size: 18px; color: {COLORS['danger']}; background: transparent;")
        QMessageBox.critical(self, "预测错误", msg)


# ============================================================
# 页面：参数设置
# ============================================================

class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(0, 0, 16, 0)
        main_layout.setSpacing(16)

        # 算法选择
        algo_group = QGroupBox("算法与模型")
        algo_layout = QFormLayout(algo_group)

        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["FedProx", "FedProxV2", "FedAvg", "SCAFFOLD", "FedDyn", "MOON"])
        algo_layout.addRow("联邦算法:", self.combo_algo)

        self.combo_model = QComboBox()
        self.combo_model.addItems(["ResNet18", "MobileNet"])
        algo_layout.addRow("模型:", self.combo_model)

        self.combo_dataset = QComboBox()
        self.combo_dataset.addItems(["Cifar10", "MNIST", "crop"])
        algo_layout.addRow("数据集:", self.combo_dataset)

        self.spin_num_classes = QSpinBox()
        self.spin_num_classes.setRange(2, 1000)
        self.spin_num_classes.setValue(10)
        algo_layout.addRow("类别数:", self.spin_num_classes)

        main_layout.addWidget(algo_group)

        # 训练参数
        train_group = QGroupBox("训练参数")
        train_layout = QFormLayout(train_group)

        self.spin_rounds = QSpinBox()
        self.spin_rounds.setRange(1, 10000)
        self.spin_rounds.setValue(100)
        train_layout.addRow("全局轮数:", self.spin_rounds)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 100)
        self.spin_epochs.setValue(5)
        train_layout.addRow("本地轮数:", self.spin_epochs)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 1.0)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setValue(0.01)
        train_layout.addRow("学习率:", self.spin_lr)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 512)
        self.spin_batch.setValue(64)
        train_layout.addRow("批大小:", self.spin_batch)

        self.spin_eval_gap = QSpinBox()
        self.spin_eval_gap.setRange(1, 100)
        self.spin_eval_gap.setValue(5)
        train_layout.addRow("评估间隔:", self.spin_eval_gap)

        main_layout.addWidget(train_group)

        # 联邦学习参数
        fl_group = QGroupBox("联邦学习参数")
        fl_layout = QFormLayout(fl_group)

        self.spin_clients = QSpinBox()
        self.spin_clients.setRange(1, 1000)
        self.spin_clients.setValue(20)
        fl_layout.addRow("客户端数:", self.spin_clients)

        self.spin_mu = QDoubleSpinBox()
        self.spin_mu.setRange(0.0, 10.0)
        self.spin_mu.setDecimals(3)
        self.spin_mu.setSingleStep(0.01)
        self.spin_mu.setValue(0.05)
        fl_layout.addRow("FedProx mu:", self.spin_mu)

        self.spin_join = QDoubleSpinBox()
        self.spin_join.setRange(0.1, 1.0)
        self.spin_join.setSingleStep(0.1)
        self.spin_join.setValue(1.0)
        fl_layout.addRow("参与比例:", self.spin_join)

        main_layout.addWidget(fl_group)

        # 设备配置
        device_group = QGroupBox("设备配置")
        device_layout = QFormLayout(device_group)

        self.combo_device = QComboBox()
        self.combo_device.addItems(["cuda", "cpu"])
        device_layout.addRow("设备:", self.combo_device)

        self.spin_device_id = QSpinBox()
        self.spin_device_id.setRange(0, 7)
        device_layout.addRow("GPU编号:", self.spin_device_id)

        self.check_lr_decay = QComboBox()
        self.check_lr_decay.addItems(["True", "False"])
        self.check_lr_decay.setCurrentText("True")
        device_layout.addRow("学习率衰减:", self.check_lr_decay)

        self.spin_lr_gamma = QDoubleSpinBox()
        self.spin_lr_gamma.setRange(0.8, 1.0)
        self.spin_lr_gamma.setDecimals(3)
        self.spin_lr_gamma.setSingleStep(0.01)
        self.spin_lr_gamma.setValue(0.98)
        device_layout.addRow("衰减系数:", self.spin_lr_gamma)

        main_layout.addWidget(device_group)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_reset = QPushButton("恢复默认")
        btn_reset.setObjectName("btn_outline")
        btn_reset.clicked.connect(self.reset_defaults)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_reset)
        main_layout.addLayout(btn_layout)

        main_layout.addStretch()
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 16, 24, 16)
        outer.addWidget(scroll)

    def reset_defaults(self):
        self.combo_algo.setCurrentText("FedProxV2")
        self.combo_model.setCurrentText("ResNet18")
        self.combo_dataset.setCurrentText("Cifar10")
        self.spin_num_classes.setValue(10)
        self.spin_rounds.setValue(100)
        self.spin_epochs.setValue(5)
        self.spin_lr.setValue(0.01)
        self.spin_batch.setValue(64)
        self.spin_eval_gap.setValue(5)
        self.spin_clients.setValue(20)
        self.spin_mu.setValue(0.05)
        self.spin_join.setValue(1.0)
        self.combo_device.setCurrentText("cuda")
        self.spin_device_id.setValue(0)
        self.check_lr_decay.setCurrentText("True")
        self.spin_lr_gamma.setValue(0.98)

    def get_args(self):
        algo = self.combo_algo.currentText()
        model = self.combo_model.currentText()
        dataset = self.combo_dataset.currentText()

        model_map = {"ResNet18": "ResNet18", "MobileNet": "MobileNet"}

        if dataset == "crop":
            num_classes = 3
            self.spin_num_classes.setValue(3)

        args = {
            'algorithm': algo,
            'model': model_map.get(model, model),
            'dataset': dataset,
            'num_classes': self.spin_num_classes.value(),
            'global_rounds': self.spin_rounds.value(),
            'local_epochs': self.spin_epochs.value(),
            'local_learning_rate': self.spin_lr.value(),
            'batch_size': self.spin_batch.value(),
            'eval_gap': self.spin_eval_gap.value(),
            'num_clients': self.spin_clients.value(),
            'mu': self.spin_mu.value(),
            'join_ratio': self.spin_join.value(),
            'random_join_ratio': False,
            'device': torch.device(self.combo_device.currentText) if torch.cuda.is_available() else torch.device('cpu'),
            'device_id': str(self.spin_device_id.value()),
            'learning_rate_decay': self.check_lr_decay.currentText() == "True",
            'learning_rate_decay_gamma': self.spin_lr_gamma.value(),
            'goal': 'test',
            'times': 1,
            'prev': 0,
            'top_cnt': 100,
            'auto_break': False,
            'dlg_eval': False,
            'dlg_gap': 100,
            'batch_num_per_client': 2,
            'num_new_clients': 0,
            'fine_tuning_epoch_new': 0,
            'few_shot': 0,
            'client_drop_rate': 0.0,
            'train_slow_rate': 0.0,
            'send_slow_rate': 0.0,
            'time_select': False,
            'time_threthold': 10000,
            'beta': 0.0,
            'lamda': 1.0,
            'K': 5,
            'p_learning_rate': 0.01,
            'M': 5,
            'itk': 4000,
            'alphaK': 1.0,
            'sigma': 1.0,
            'alpha': 1.0,
            'plocal_epochs': 1,
            'tau': 1.0,
            'fine_tuning_epochs': 10,
            'dr_learning_rate': 0.0,
            'L': 1.0,
            'noise_dim': 512,
            'generator_learning_rate': 0.005,
            'hidden_dim': 512,
            'server_epochs': 1000,
            'localize_feature_extractor': False,
            'server_learning_rate': 1.0,
            'eta': 1.0,
            'rand_percent': 80,
            'layer_idx': 2,
            'mentee_learning_rate': 0.005,
            'T_start': 0.95,
            'T_end': 0.98,
            'momentum': 0.1,
            'kl_weight': 0.0,
            'first_stage_bound': 0,
            'fedcross_alpha': 0.99,
            'collaberative_model_select_strategy': 1,
            'feature_dim': 512,
            'vocab_size': 80,
            'max_len': 200,
            'save_folder_name': 'items',
            'save_dir': 'saved_models',
            'multi_gpu': False,
        }
        return args


# ============================================================
# 主窗口
# ============================================================

class FLApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FL-Crop Vis - 联邦学习农作物识别系统")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 左侧导航
        nav_widget = QWidget()
        nav_widget.setFixedWidth(180)
        nav_widget.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-right: 1px solid {COLORS['border']};")
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(8, 16, 8, 16)
        nav_layout.setSpacing(8)

        # Logo
        logo_label = QLabel("FL-Crop")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {COLORS['accent']}; padding: 16px; background: transparent;")
        nav_layout.addWidget(logo_label)

        subtitle = QLabel("联邦学习作物识别")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"font-size: 11px; color: {COLORS['text_secondary']}; background: transparent; margin-bottom: 16px;")
        nav_layout.addWidget(subtitle)

        # 导航列表
        self.nav_list = QListWidget()
        self.nav_list.setStyleSheet(f"QListWidget {{ background: transparent; }}")

        items = [
            ("训练监控", "training"),
            ("图片识别", "predict"),
            ("参数设置", "settings"),
        ]
        for text, key in items:
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, key)
            self.nav_list.addItem(item)

        self.nav_list.currentRowChanged.connect(self.switch_page)
        nav_layout.addWidget(self.nav_list)

        # 底部设备信息
        nav_layout.addStretch()
        device_text = "GPU" if torch.cuda.is_available() else "CPU"
        if torch.cuda.is_available():
            device_text = f"GPU: {torch.cuda.get_device_name(0)[:20]}"
        device_label = QLabel(device_text)
        device_label.setAlignment(Qt.AlignCenter)
        device_label.setStyleSheet(f"font-size: 10px; color: {COLORS['text_secondary']}; background: transparent; padding: 8px;")
        nav_layout.addWidget(device_label)

        main_layout.addWidget(nav_widget)

        # 右侧内容
        self.stack = QStackedWidget()
        self.train_page = TrainMonitorPage()
        self.predict_page = PredictPage()
        self.settings_page = SettingsPage()

        self.stack.addWidget(self.train_page)
        self.stack.addWidget(self.predict_page)
        self.stack.addWidget(self.settings_page)

        main_layout.addWidget(self.stack, 1)

        # 默认选中第一个
        self.nav_list.setCurrentRow(0)

        # 状态栏
        self.statusBar().setStyleSheet(f"background-color: {COLORS['bg_medium']}; color: {COLORS['text_secondary']}; border-top: 1px solid {COLORS['border']}; font-size: 11px;")
        self.statusBar().showMessage("就绪 | 系统: FL-Crop Vis v1.0")

    def switch_page(self, index):
        self.stack.setCurrentIndex(index)

    def get_train_args(self):
        return self.settings_page.get_args()


# ============================================================
# 入口
# ============================================================

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    window = FLApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

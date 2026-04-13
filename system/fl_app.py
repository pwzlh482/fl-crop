"""
fl_app.py - 联邦学习可视化监控软件 V2
基于实际服务器代码架构重写

功能：
1. 训练监控：实时显示 loss/accuracy 曲线、训练日志
2. 图片预测：加载模型 + 预测单张图片
3. 参数配置：修改训练参数并启动训练

运行方式：python fl_app.py
"""

import sys
import os
import json
import time
import threading
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox, QTabWidget,
    QFileDialog, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QProgressBar, QSplitter, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QImage

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 深色主题样式
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
}
QTabWidget::pane {
    border: 1px solid #45475a;
    background-color: #1e1e2e;
}
QTabBar::tab {
    background-color: #313244;
    color: #cdd6f4;
    padding: 10px 25px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-size: 13px;
}
QTabBar::tab:selected {
    background-color: #45475a;
    color: #89b4fa;
    font-weight: bold;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 6px 10px;
    min-height: 28px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #89b4fa;
}
QPushButton {
    background-color: #45475a;
    color: #cdd6f4;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #585b70;
}
QPushButton:pressed {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QPushButton:disabled {
    background-color: #313244;
    color: #6c7086;
}
QPushButton#startBtn {
    background-color: #a6e3a1;
    color: #1e1e2e;
    font-weight: bold;
}
QPushButton#startBtn:hover {
    background-color: #94e2d5;
}
QPushButton#stopBtn {
    background-color: #f38ba8;
    color: #1e1e2e;
    font-weight: bold;
}
QPushButton#stopBtn:hover {
    background-color: #eba0ac;
}
QPushButton#predictBtn {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-weight: bold;
}
QTextEdit {
    background-color: #11111b;
    color: #a6e3a1;
    border: 1px solid #45475a;
    border-radius: 4px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    padding: 6px;
}
QProgressBar {
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    color: #cdd6f4;
    background-color: #313244;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 3px;
}
QLabel#statValue {
    color: #89b4fa;
    font-size: 22px;
    font-weight: bold;
}
QLabel#statLabel {
    color: #6c7086;
    font-size: 11px;
}
QCheckBox {
    color: #cdd6f4;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
}
"""


class MplCanvas(FigureCanvas):
    """matplotlib画布"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e2e')
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self._setup_axis(self.ax)

    def _setup_axis(self, ax):
        ax.set_facecolor('#11111b')
        ax.tick_params(colors='#6c7086', labelsize=9)
        ax.spines['bottom'].set_color('#45475a')
        ax.spines['left'].set_color('#45475a')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.label.set_color('#cdd6f4')
        ax.yaxis.label.set_color('#cdd6f4')
        ax.title.set_color('#89b4fa')


class StatCard(QWidget):
    """统计卡片"""
    def __init__(self, label_text, value_text="--", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        self.value_label = QLabel(value_text)
        self.value_label.setObjectName("statValue")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.label_label = QLabel(label_text)
        self.label_label.setObjectName("statLabel")
        self.label_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)
        layout.addWidget(self.label_label)

    def set_value(self, text):
        self.value_label.setText(text)


class TrainMonitorPage(QWidget):
    """训练监控页面"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 统计卡片行
        cards_layout = QHBoxLayout()
        self.card_round = StatCard("当前轮次", "0")
        self.card_loss = StatCard("训练Loss", "--")
        self.card_acc = StatCard("测试Accuracy", "--")
        self.card_lr = StatCard("学习率", "--")
        self.card_best = StatCard("最佳Accuracy", "--")
        self.card_time = StatCard("轮次耗时", "--")
        for card in [self.card_round, self.card_loss, self.card_acc, self.card_lr, self.card_best, self.card_time]:
            cards_layout.addWidget(card)
        layout.addLayout(cards_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("等待启动...")
        layout.addWidget(self.progress_bar)

        # 图表区域
        charts_layout = QHBoxLayout()
        self.loss_canvas = MplCanvas(self, width=6, height=3.5)
        self.acc_canvas = MplCanvas(self, width=6, height=3.5)
        charts_layout.addWidget(self.loss_canvas)
        charts_layout.addWidget(self.acc_canvas)
        layout.addLayout(charts_layout)

        # 日志区域
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(160)
        layout.addWidget(self.log_text)

        # 初始化数据
        self.rounds = []
        self.losses = []
        self.accs = []
        self.best_acc = 0.0

    def update_data(self, round_num, loss, acc, lr, time_cost):
        self.rounds.append(round_num)
        self.losses.append(loss)
        self.accs.append(acc)

        self.card_round.set_value(str(round_num))
        self.card_loss.set_value(f"{loss:.4f}")
        self.card_acc.set_value(f"{acc:.4f}")
        self.card_lr.set_value(f"{lr:.6f}")
        self.card_time.set_value(f"{time_cost:.1f}s")

        if acc > self.best_acc:
            self.best_acc = acc
        self.card_best.set_value(f"{self.best_acc:.4f}")

        self._update_charts()

    def _update_charts(self):
        # Loss曲线
        ax = self.loss_canvas.ax
        ax.clear()
        self.loss_canvas._setup_axis(ax)
        if len(self.rounds) > 0:
            ax.plot(self.rounds, self.losses, color='#f38ba8', linewidth=1.5, label='Train Loss')
            ax.legend(facecolor='#313244', edgecolor='#45475a', labelcolor='#cdd6f4', fontsize=9)
        ax.set_xlabel('Round')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        self.loss_canvas.draw()

        # Accuracy曲线
        ax2 = self.acc_canvas.ax
        ax2.clear()
        self.acc_canvas._setup_axis(ax2)
        if len(self.rounds) > 0:
            ax2.plot(self.rounds, self.accs, color='#a6e3a1', linewidth=1.5, label='Test Acc')
            ax2.legend(facecolor='#313244', edgecolor='#45475a', labelcolor='#cdd6f4', fontsize=9)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Test Accuracy')
        self.acc_canvas.draw()

    def append_log(self, text):
        self.log_text.append(text)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def set_progress(self, current, total, text=""):
        pct = int(current / max(total, 1) * 100)
        self.progress_bar.setValue(pct)
        if text:
            self.progress_bar.setFormat(text)
        else:
            self.progress_bar.setFormat(f"Round {current}/{total} ({pct}%)")

    def reset(self):
        self.rounds = []
        self.losses = []
        self.accs = []
        self.best_acc = 0.0
        self.card_round.set_value("0")
        self.card_loss.set_value("--")
        self.card_acc.set_value("--")
        self.card_lr.set_value("--")
        self.card_best.set_value("--")
        self.card_time.set_value("--")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("等待启动...")
        self.log_text.clear()


class ImageDropLabel(QLabel):
    """图片拖放/点击标签"""
    image_loaded = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("点击选择图片\n或拖放图片到此处")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #45475a;
                border-radius: 8px;
                color: #6c7086;
                font-size: 14px;
                background-color: #313244;
                min-height: 250px;
            }
            QLabel:hover {
                border-color: #89b4fa;
                color: #89b4fa;
            }
        """)
        self.setAcceptDrops(True)
        self._image_path = None

    def mousePressEvent(self, event):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self._load_image(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self._load_image(path)
                break

    def _load_image(self, path):
        self._image_path = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)
        self.image_loaded.emit(path)

    def get_image_path(self):
        return self._image_path


class PredictPage(QWidget):
    """图片预测页面"""
    prediction_done = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 顶部：模型选择
        model_group = QGroupBox("模型设置")
        model_layout = QHBoxLayout(model_group)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("选择模型文件 (.pt)")
        self.model_browse_btn = QPushButton("浏览")
        self.model_browse_btn.clicked.connect(self._browse_model)
        model_layout.addWidget(QLabel("模型路径:"))
        model_layout.addWidget(self.model_path_edit, 1)
        model_layout.addWidget(self.model_browse_btn)
        layout.addWidget(model_group)

        # 中间：图片 + 结果
        mid_layout = QHBoxLayout()

        # 左侧图片区
        self.image_label = ImageDropLabel()
        mid_layout.addWidget(self.image_label, 1)

        # 右侧结果区
        result_group = QGroupBox("预测结果")
        result_layout = QVBoxLayout(result_group)
        self.result_canvas = MplCanvas(self, width=5, height=3.5)
        result_layout.addWidget(self.result_canvas)
        self.result_label = QLabel("等待预测...")
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        self.result_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.result_label)
        mid_layout.addWidget(result_group, 1)
        layout.addLayout(mid_layout)

        # 底部：预测按钮
        btn_layout = QHBoxLayout()
        self.predict_btn = QPushButton("开始预测")
        self.predict_btn.setObjectName("predictBtn")
        self.predict_btn.setMinimumHeight(40)
        self.predict_btn.clicked.connect(self._do_predict)
        btn_layout.addStretch()
        btn_layout.addWidget(self.predict_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self._model = None

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Model (*.pt *.pth)")
        if path:
            self.model_path_edit.setText(path)

    def _do_predict(self):
        image_path = self.image_label.get_image_path()
        model_path = self.model_path_edit.text().strip()

        if not image_path:
            QMessageBox.warning(self, "提示", "请先选择要预测的图片")
            return
        if not model_path:
            QMessageBox.warning(self, "提示", "请先选择模型文件")
            return

        self.predict_btn.setEnabled(False)
        self.result_label.setText("预测中...")

        try:
            import torch
            import torchvision.transforms as transforms
            from PIL import Image

            # 加载模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loaded = torch.load(model_path, map_location=device)

            if isinstance(loaded, torch.nn.Module):
                model = loaded
            else:
                from torchvision.models import mobilenet_v2
                model = mobilenet_v2(num_classes=10)
                model.load_state_dict(loaded)

            model = model.to(device)
            model.eval()

            # 预处理图片
            transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])

            img = Image.open(image_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

            # CIFAR-10 类别名
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']

            top_idx = np.argmax(probs)
            self.result_label.setText(f"预测结果: {class_names[top_idx]} ({probs[top_idx]:.2%})")

            # 绘制概率条
            self._plot_probs(probs, class_names)

        except Exception as e:
            self.result_label.setText(f"预测失败: {str(e)}")
        finally:
            self.predict_btn.setEnabled(True)

    def _plot_probs(self, probs, class_names):
        ax = self.result_canvas.ax
        ax.clear()
        self.result_canvas._setup_axis(ax)

        colors = ['#89b4fa' if i != np.argmax(probs) else '#a6e3a1' for i in range(len(probs))]
        y_pos = np.arange(len(class_names))
        ax.barh(y_pos, probs, color=colors, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel('Probability')
        ax.set_title('Class Probabilities')
        ax.set_xlim(0, 1)
        self.result_canvas.draw()


class SettingsPage(QWidget):
    """参数配置页面"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # 基础参数
        basic_group = QGroupBox("基础参数")
        basic_form = QFormLayout(basic_group)

        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["FedProxV2", "FedProx"])
        basic_form.addRow("算法:", self.algo_combo)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Cifar10", "MNIST"])
        basic_form.addRow("数据集:", self.dataset_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["MobileNet", "ResNet18", "CNN", "DNN"])
        basic_form.addRow("模型:", self.model_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        basic_form.addRow("设备:", self.device_combo)

        self.device_id_edit = QLineEdit("0")
        self.device_id_edit.setPlaceholderText("如 0 或 0,1,2,3")
        basic_form.addRow("GPU ID:", self.device_id_edit)

        layout.addWidget(basic_group)

        # 训练参数
        train_group = QGroupBox("训练参数")
        train_form = QFormLayout(train_group)

        self.rounds_spin = QSpinBox()
        self.rounds_spin.setRange(1, 10000)
        self.rounds_spin.setValue(70)
        train_form.addRow("全局轮次:", self.rounds_spin)

        self.clients_spin = QSpinBox()
        self.clients_spin.setRange(2, 1000)
        self.clients_spin.setValue(20)
        train_form.addRow("客户端数量:", self.clients_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(1)
        train_form.addRow("本地轮次:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, 1024)
        self.batch_spin.setValue(64)
        train_form.addRow("Batch Size:", self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.001)
        train_form.addRow("学习率:", self.lr_spin)

        self.mu_spin = QDoubleSpinBox()
        self.mu_spin.setRange(0.0, 1.0)
        self.mu_spin.setValue(0.01)
        self.mu_spin.setDecimals(4)
        self.mu_spin.setSingleStep(0.001)
        train_form.addRow("Mu (FedProx):", self.mu_spin)

        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 0.999)
        self.momentum_spin.setValue(0.9)
        self.momentum_spin.setDecimals(2)
        train_form.addRow("Momentum:", self.momentum_spin)

        self.wd_spin = QDoubleSpinBox()
        self.wd_spin.setRange(0.0, 0.1)
        self.wd_spin.setValue(1e-4)
        self.wd_spin.setDecimals(5)
        self.wd_spin.setSingleStep(0.0001)
        train_form.addRow("Weight Decay:", self.wd_spin)

        layout.addWidget(train_group)

        # V2 优化参数
        v2_group = QGroupBox("V2 优化参数")
        v2_form = QFormLayout(v2_group)

        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 50)
        self.warmup_spin.setValue(5)
        v2_form.addRow("Warmup轮次:", self.warmup_spin)

        self.milestones_edit = QLineEdit("20,40")
        self.milestones_edit.setPlaceholderText("如 20,40")
        v2_form.addRow("LR衰减轮次:", self.milestones_edit)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.01, 1.0)
        self.gamma_spin.setValue(0.2)
        self.gamma_spin.setDecimals(2)
        v2_form.addRow("LR衰减系数:", self.gamma_spin)

        self.pretrained_edit = QLineEdit()
        self.pretrained_edit.setPlaceholderText("如 resnet18_imagenet.pt（空则不加载）")
        v2_form.addRow("预训练权重:", self.pretrained_edit)

        layout.addWidget(v2_group)

        layout.addStretch()

    def get_args_dict(self):
        milestones_str = self.milestones_edit.text().strip()
        milestones = [int(x.strip()) for x in milestones_str.split(',') if x.strip()]

        return {
            'algorithm': self.algo_combo.currentText(),
            'dataset': self.dataset_combo.currentText(),
            'model': self.model_combo.currentText(),
            'device': self.device_combo.currentText(),
            'device_id': self.device_id_edit.text().strip(),
            'global_rounds': self.rounds_spin.value(),
            'num_clients': self.clients_spin.value(),
            'local_epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'local_learning_rate': self.lr_spin.value(),
            'mu': self.mu_spin.value(),
            'momentum': self.momentum_spin.value(),
            'weight_decay': self.wd_spin.value(),
            'warmup_rounds': self.warmup_spin.value(),
            'lr_decay_milestones': milestones,
            'learning_rate_decay_gamma': self.gamma_spin.value(),
            'pretrained_path': self.pretrained_edit.text().strip(),
        }


class TrainThread(QThread):
    """训练线程"""
    log_signal = pyqtSignal(str)
    data_signal = pyqtSignal(int, float, float, float, float)  # round, loss, acc, lr, time
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(str)

    def __init__(self, args_dict):
        super().__init__()
        self.args_dict = args_dict
        self._stop_flag = False

    def run(self):
        import subprocess
        import re

        # 构建命令
        cmd = [sys.executable, "main_v2.py"]
        args_map = {
            'algorithm': '-algo',
            'dataset': '-data',
            'model': '-m',
            'device': '-dev',
            'device_id': '-did',
            'global_rounds': '-gr',
            'num_clients': '-nc',
            'local_epochs': '-ls',
            'batch_size': '-lbs',
            'local_learning_rate': '-lr',
            'mu': '-mu',
            'momentum': '-mo',
            'weight_decay': '--weight-decay',
            'warmup_rounds': '-wr',
            'learning_rate_decay_gamma': '-ldg',
            'pretrained_path': '-pp',
        }

        for key, flag in args_map.items():
            val = self.args_dict.get(key, '')
            if val != '' and val is not None:
                cmd.extend([flag, str(val)])

        # milestones 单独处理
        milestones = self.args_dict.get('lr_decay_milestones', [])
        if milestones:
            cmd.extend(['-ldm'] + [str(m) for m in milestones])

        self.log_signal.emit(f"启动命令: {' '.join(cmd)}\n")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            current_round = 0
            total_rounds = self.args_dict.get('global_rounds', 70)

            for line in process.stdout:
                if self._stop_flag:
                    process.terminate()
                    break

                self.log_signal.emit(line.rstrip())

                # 解析轮次
                round_match = re.search(r'Round number:\s*(\d+)', line)
                if round_match:
                    current_round = int(round_match.group(1))
                    self.progress_signal.emit(current_round, total_rounds, f"Round {current_round}/{total_rounds}")

                # 解析指标
                loss_match = re.search(r'Averaged Train Loss:\s*([\d.]+)', line)
                acc_match = re.search(r'Averaged Test Accuracy:\s*([\d.]+)', line)
                lr_match = re.search(r'Current Learning Rate:\s*([\d.eE+-]+)', line)
                time_match = re.search(r'time cost\s*[-]+\s*([\d.]+)', line)

                if loss_match and acc_match:
                    loss_val = float(loss_match.group(1))
                    acc_val = float(acc_match.group(1))
                    lr_val = float(lr_match.group(1)) if lr_match else 0.0
                    time_val = float(time_match.group(1)) if time_match else 0.0
                    self.data_signal.emit(current_round, loss_val, acc_val, lr_val, time_val)

            process.wait()
            self.finished_signal.emit("训练完成" if not self._stop_flag else "训练已停止")

        except Exception as e:
            self.finished_signal.emit(f"训练出错: {str(e)}")

    def stop(self):
        self._stop_flag = True


class FLApp(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("联邦学习训练监控 - FL-Crop V2")
        self.setMinimumSize(1100, 750)
        self.setStyleSheet(DARK_STYLE)

        # 中心控件
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Tab页
        self.tabs = QTabWidget()
        self.monitor_page = TrainMonitorPage()
        self.predict_page = PredictPage()
        self.settings_page = SettingsPage()
        self.tabs.addTab(self.monitor_page, "训练监控")
        self.tabs.addTab(self.predict_page, "图片预测")
        self.tabs.addTab(self.settings_page, "参数配置")
        main_layout.addWidget(self.tabs)

        # 底部控制栏
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setMinimumHeight(38)
        self.start_btn.clicked.connect(self._start_training)

        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setMinimumHeight(38)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)

        control_layout.addStretch()
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        # 状态栏
        self.statusBar().setStyleSheet("color: #6c7086; font-size: 11px;")
        self.statusBar().showMessage("就绪")

        self.train_thread = None

    def _start_training(self):
        args = self.settings_page.get_args_dict()
        self.monitor_page.reset()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage(f"训练中 - {args['algorithm']} / {args['dataset']} / {args['model']}")

        self.train_thread = TrainThread(args)
        self.train_thread.log_signal.connect(self.monitor_page.append_log)
        self.train_thread.data_signal.connect(self.monitor_page.update_data)
        self.train_thread.progress_signal.connect(self.monitor_page.set_progress)
        self.train_thread.finished_signal.connect(self._on_training_finished)
        self.train_thread.start()

    def _stop_training(self):
        if self.train_thread:
            self.train_thread.stop()
        self.statusBar().showMessage("正在停止训练...")

    def _on_training_finished(self, msg):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage(msg)
        QMessageBox.information(self, "提示", msg)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FLApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

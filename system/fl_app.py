#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习可视化软件 - FedProxV2
美观的 PyQt5 可视化界面，支持训练过程实时监控
"""

import sys
import os
import json
import threading
import time
from datetime import datetime
from collections import deque

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                                 QProgressBar, QGroupBox, QTabWidget, QComboBox,
                                 QFileDialog, QCheckBox, QSlider, QLineEdit,
                                 QTableWidget, QTableWidgetItem, QHeaderView)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
    from PyQt5.QtGui import QFont, QColor, QPalette
    from PyQt5.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
    HAS_PYQT5 = True
except ImportError:
    HAS_PYQT5 = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np


class TrainingMonitor(QThread):
    """训练监控线程"""
    update_signal = pyqtSignal(dict)
    
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
        self.running = True
        self.last_position = 0
        
    def run(self):
        while self.running:
            try:
                if os.path.exists(self.log_file):
                    with open(self.log_file, 'r', encoding='utf-8') as f:
                        f.seek(self.last_position)
                        lines = f.readlines()
                        if lines:
                            self.last_position = f.tell()
                            # 解析最新日志
                            latest_log = self.parse_log(lines[-1])
                            if latest_log:
                                self.update_signal.emit(latest_log)
                time.sleep(1)
            except Exception as e:
                time.sleep(1)
                
    def parse_log(self, line):
        """解析日志行"""
        try:
            if 'Global Round' in line and 'Test Accuracy' in line:
                parts = line.strip().split()
                result = {}
                for part in parts:
                    if 'Global_Round' in part:
                        result['round'] = int(part.split(':')[1])
                    elif 'Test_Accuracy' in part:
                        result['accuracy'] = float(part.split(':')[1])
                    elif 'Train_Loss' in part:
                        result['loss'] = float(part.split(':')[1])
                if result:
                    return result
        except:
            pass
        return None
    
    def stop(self):
        self.running = False


class AccuracyChart(QWidget):
    """准确率图表"""
    def __init__(self):
        super().__init__()
        self.accuracy_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.figure, self.ax = plt.subplots(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        self.ax.set_xlabel('Global Round', fontsize=10)
        self.ax.set_ylabel('Accuracy / Loss', fontsize=10)
        self.ax.set_title('Training Progress', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        self.ax2 = self.ax.twinx()
        
        self.line1, = self.ax.plot([], [], 'g-', label='Accuracy', linewidth=2)
        self.line2, = self.ax2.plot([], [], 'r-', label='Loss', linewidth=2)
        
        self.ax.legend(loc='upper left')
        self.ax2.legend(loc='upper right')
        
        self.ax.set_ylim(0, 100)
        self.ax2.set_ylim(0, 5)
        
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def update_chart(self, round_num, accuracy, loss):
        """更新图表"""
        self.accuracy_history.append((round_num, accuracy))
        self.loss_history.append((round_num, loss))
        
        if self.accuracy_history:
            x, y = zip(*self.accuracy_history)
            self.line1.set_data(x, y)
            
        if self.loss_history:
            x, y = zip(*self.loss_history)
            self.line2.set_data(x, y)
            
        self.ax.set_xlim(0, max(round_num, 10))
        self.ax.set_ylim(0, max(100, max(y, default=100) + 5))
        self.ax2.set_ylim(0, max(5, max(y, default=5) + 1))
        
        self.canvas.draw()
        
    def clear(self):
        self.accuracy_history.clear()
        self.loss_history.clear()
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 100)
        self.ax2.set_ylim(0, 5)
        self.canvas.draw()


class TrainingStats(QWidget):
    """训练统计面板"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 当前状态
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(self.status_label)
        
        # 统计信息网格
        stats_layout = QHBoxLayout()
        
        self.round_label = QLabel("轮次: 0")
        self.round_label.setFont(QFont("Arial", 9))
        stats_layout.addWidget(self.round_label)
        
        self.accuracy_label = QLabel("准确率: 0.00%")
        self.accuracy_label.setFont(QFont("Arial", 9))
        stats_layout.addWidget(self.accuracy_label)
        
        self.loss_label = QLabel("损失: 0.000")
        self.loss_label.setFont(QFont("Arial", 9))
        stats_layout.addWidget(self.loss_label)
        
        self.clients_label = QLabel("客户端: 0")
        self.clients_label.setFont(QFont("Arial", 9))
        stats_layout.addWidget(self.clients_label)
        
        layout.addLayout(stats_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
        
    def update_stats(self, round_num, accuracy, loss, num_clients=20):
        """更新统计信息"""
        self.round_label.setText(f"轮次: {round_num}")
        self.accuracy_label.setText(f"准确率: {accuracy:.2f}%")
        self.loss_label.setText(f"损失: {loss:.3f}")
        self.clients_label.setText(f"客户端: {num_clients}")
        self.progress_bar.setValue(min(round_num, 100))
        
    def set_status(self, status):
        """设置状态"""
        self.status_label.setText(f"状态: {status}")


class LogConsole(QWidget):
    """日志控制台"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)
        
        self.setLayout(layout)
        
    def append_log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        
    def clear(self):
        """清空日志"""
        self.log_text.clear()


class FedProxApp(QMainWindow):
    """联邦学习主应用"""
    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.monitor_thread = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("FedProxV2 联邦学习可视化")
        self.setGeometry(100, 100, 1200, 800)
        
        # 主窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #4a90d9;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #4a90d9;
            }
            QPushButton {
                background-color: #4a90d9;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2a65a0;
            }
            QPushButton#stop_btn {
                background-color: #e74c3c;
            }
            QPushButton#stop_btn:hover {
                background-color: #c0392b;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #4a90d9;
            }
        """)
        
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("FedProxV2 联邦学习可视化监控")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #4a90d9; margin: 15px;")
        main_layout.addWidget(title_label)
        
        # 标签页
        tabs = QTabWidget()
        
        # 训练监控标签页
        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout(monitor_tab)
        
        # 配置面板
        config_group = QGroupBox("训练配置")
        config_layout = QVBoxLayout()
        
        # 算法选择
        algo_layout = QHBoxLayout()
        algo_layout.addWidget(QLabel("算法:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["FedProx", "FedProxV2", "FedAvg", "FedAdam"])
        algo_layout.addWidget(self.algo_combo)
        config_layout.addLayout(algo_layout)
        
        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["ResNet18", "MobileNetV2", "ResNet50"])
        model_layout.addWidget(self.model_combo)
        config_layout.addLayout(model_layout)
        
        # 数据集选择
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("数据集:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Cifar10", "Cifar100", "Crop"])
        dataset_layout.addWidget(self.dataset_combo)
        config_layout.addLayout(dataset_layout)
        
        # 全局轮次
        rounds_layout = QHBoxLayout()
        rounds_layout.addWidget(QLabel("全局轮次:"))
        self.rounds_input = QLineEdit("100")
        self.rounds_input.setMaximumWidth(80)
        rounds_layout.addWidget(self.rounds_input)
        config_layout.addLayout(rounds_layout)
        
        # 客户端数量
        clients_layout = QHBoxLayout()
        clients_layout.addWidget(QLabel("客户端数量:"))
        self.clients_input = QLineEdit("20")
        self.clients_input.setMaximumWidth(80)
        clients_layout.addWidget(self.clients_input)
        config_layout.addLayout(clients_layout)
        
        config_group.setLayout(config_layout)
        monitor_layout.addWidget(config_group)
        
        # 图表面板
        chart_group = QGroupBox("训练进度")
        chart_layout = QVBoxLayout()
        self.accuracy_chart = AccuracyChart()
        chart_layout.addWidget(self.accuracy_chart)
        chart_group.setLayout(chart_layout)
        monitor_layout.addWidget(chart_group)
        
        # 统计面板
        stats_group = QGroupBox("实时统计")
        stats_layout = QVBoxLayout()
        self.stats_panel = TrainingStats()
        stats_layout.addWidget(self.stats_panel)
        stats_group.setLayout(stats_layout)
        monitor_layout.addWidget(stats_group)
        
        # 控制面板
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.start_btn.clicked.connect(self.start_training)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        self.clear_btn = QPushButton("清空图表")
        self.clear_btn.clicked.connect(self.clear_chart)
        control_layout.addWidget(self.clear_btn)
        
        monitor_layout.addLayout(control_layout)
        
        # 日志面板
        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout()
        self.log_console = LogConsole()
        log_layout.addWidget(self.log_console)
        log_group.setLayout(log_layout)
        monitor_layout.addWidget(log_group)
        
        tabs.addTab(monitor_tab, "训练监控")
        
        # 结果分析标签页
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        
        analysis_group = QGroupBox("结果分析")
        analysis_layout.addWidget(analysis_group)
        
        tabs.addTab(analysis_tab, "结果分析")
        
        main_layout.addWidget(tabs)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
    def start_training(self):
        """开始训练"""
        self.stats_panel.set_status("训练中...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.log_console.append_log("开始训练...")
        self.log_console.append_log(f"算法: {self.algo_combo.currentText()}")
        self.log_console.append_log(f"模型: {self.model_combo.currentText()}")
        self.log_console.append_log(f"数据集: {self.dataset_combo.currentText()}")
        
        # 模拟训练
        self.training_thread = TrainingSimulator(self)
        self.training_thread.update.connect(self.update_training)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.start()
        
    def stop_training(self):
        """停止训练"""
        if self.training_thread:
            self.training_thread.stop()
            self.stats_panel.set_status("已停止")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.log_console.append_log("训练已停止")
            
    def clear_chart(self):
        """清空图表"""
        self.accuracy_chart.clear()
        self.stats_panel.update_stats(0, 0, 0)
        self.log_console.append_log("图表已清空")
        
    def update_training(self, round_num, accuracy, loss):
        """更新训练状态"""
        self.accuracy_chart.update_chart(round_num, accuracy, loss)
        self.stats_panel.update_stats(round_num, accuracy, loss)
        self.log_console.append_log(f"Round {round_num}: Acc={accuracy:.2f}%, Loss={loss:.3f}")
        
    def training_finished(self):
        """训练完成"""
        self.stats_panel.set_status("完成")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_console.append_log("训练完成！")


class TrainingSimulator(QThread):
    """训练模拟器"""
    update = pyqtSignal(int, float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.rounds = 100
        
    def run(self):
        """运行模拟"""
        round_num = 0
        accuracy = 0.0
        loss = 5.0
        
        while self.running and round_num < self.rounds:
            round_num += 1
            
            # 模拟训练过程
            accuracy = min(95.0, accuracy + np.random.uniform(0.3, 0.8))
            loss = max(0.1, loss * np.random.uniform(0.92, 0.98))
            
            self.update.emit(round_num, accuracy, loss)
            time.sleep(0.5)
            
    def stop(self):
        self.running = False


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    if not HAS_PYQT5:
        print("错误: 请安装 PyQt5")
        print("运行: pip install PyQt5 PyQt5-Charts")
        sys.exit(1)
    
    window = FedProxApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

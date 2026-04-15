#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习客户端协同训练平台
"""

import sys
import os
import subprocess
import threading
import time
from datetime import datetime
from collections import deque

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                             QProgressBar, QGroupBox, QTabWidget, QComboBox,
                             QFileDialog, QLineEdit, QFormLayout, QMessageBox,
                             QSplitter, QFrame)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class TrainingThread(QThread):
    """训练线程 - 直接运行 main_v2.py"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def __init__(self, args, cwd):
        super().__init__()
        self.args = args
        self.cwd = cwd
        self.process = None
        self.running = True
        
    def run(self):
        try:
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            self.process = subprocess.Popen(
                self.args,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                env=env
            )
            
            for line in iter(self.process.stdout.readline, ''):
                if not self.running:
                    break
                self.log_signal.emit(line.rstrip())
            
            self.process.wait()
        except Exception as e:
            self.log_signal.emit(f"[错误] {str(e)}")
        finally:
            self.finished_signal.emit()
    
    def stop(self):
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except:
                self.process.kill()


class AccuracyChart(QWidget):
    """准确率图表"""
    def __init__(self):
        super().__init__()
        self.rounds = []
        self.accuracies = []
        self.losses = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.figure, self.ax = plt.subplots(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        self.ax.set_xlabel('Round', fontsize=11)
        self.ax.set_ylabel('Accuracy (%)', fontsize=11, color='#2ecc71')
        self.ax.set_title('Training Progress', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel('Loss', fontsize=11, color='#e74c3c')
        
        self.line1, = self.ax.plot([], [], 'o-', color='#2ecc71', label='Accuracy', linewidth=2, markersize=3)
        self.line2, = self.ax2.plot([], [], 's-', color='#e74c3c', label='Loss', linewidth=2, markersize=3)
        
        self.ax.legend(loc='upper left', fontsize=9)
        self.ax2.legend(loc='upper right', fontsize=9)
        
        self.figure.tight_layout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def update_chart(self, round_num, accuracy, loss):
        self.rounds.append(round_num)
        self.accuracies.append(accuracy)
        self.losses.append(loss)
        
        self.line1.set_data(self.rounds, self.accuracies)
        self.line2.set_data(self.rounds, self.losses)
        
        self.ax.set_xlim(0, max(round_num + 5, 10))
        self.ax.set_ylim(0, max(100, max(self.accuracies) + 5) if self.accuracies else 100)
        self.ax2.set_ylim(0, max(5, max(self.losses) + 1) if self.losses else 5)
        
        self.canvas.draw()
        
    def clear(self):
        self.rounds.clear()
        self.accuracies.clear()
        self.losses.clear()
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 100)
        self.ax2.set_ylim(0, 5)
        self.canvas.draw()


class FedProxApp(QMainWindow):
    """联邦学习主应用"""
    
    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.current_round = 0
        self.total_rounds = 100
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("联邦学习客户端协同训练平台")
        self.setGeometry(100, 100, 1200, 800)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #3498db;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #3498db;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QPushButton#stop_btn {
                background-color: #e74c3c;
            }
            QPushButton#stop_btn:hover {
                background-color: #c0392b;
            }
            QPushButton#clear_btn {
                background-color: #95a5a6;
            }
            QPushButton#clear_btn:hover {
                background-color: #7f8c8d;
            }
            QComboBox {
                padding: 8px 12px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                min-width: 140px;
                font-size: 11px;
            }
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                font-size: 11px;
            }
            QLabel {
                color: #2c3e50;
                font-size: 11px;
            }
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        title_label = QLabel("联邦学习客户端协同训练平台")
        title_label.setFont(QFont("Microsoft YaHei", 64, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin: 10px; padding: 10px;")
        main_layout.addWidget(title_label)
        
        subtitle_label = QLabel("Federated Learning Collaborative Training Platform")
        subtitle_label.setFont(QFont("Arial", 12))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #7f8c8d; margin-bottom: 10px;")
        main_layout.addWidget(subtitle_label)
        
        config_group = QGroupBox("训练配置")
        config_layout = QVBoxLayout()
        config_layout.setSpacing(15)
        
        row1 = QHBoxLayout()
        row1.setSpacing(20)
        
        left_form = QFormLayout()
        left_form.setSpacing(10)
        
        algo_label = QLabel("算法:")
        algo_label.setFont(QFont("Microsoft YaHei", 11))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["FedProxV2", "FedProx", "FedAvg"])
        left_form.addRow(algo_label, self.algo_combo)
        
        model_label = QLabel("模型:")
        model_label.setFont(QFont("Microsoft YaHei", 11))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["MobileNet", "ResNet18", "CNN", "DNN"])
        left_form.addRow(model_label, self.model_combo)
        
        data_label = QLabel("数据集:")
        data_label.setFont(QFont("Microsoft YaHei", 11))
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Cifar10", "MNIST"])
        left_form.addRow(data_label, self.dataset_combo)
        
        row1.addLayout(left_form)
        
        right_form = QFormLayout()
        right_form.setSpacing(10)
        
        rounds_label = QLabel("全局轮次:")
        rounds_label.setFont(QFont("Microsoft YaHei", 11))
        self.rounds_input = QLineEdit("10")
        right_form.addRow(rounds_label, self.rounds_input)
        
        clients_label = QLabel("客户端数量:")
        clients_label.setFont(QFont("Microsoft YaHei", 11))
        self.clients_input = QLineEdit("2")
        right_form.addRow(clients_label, self.clients_input)
        
        lr_label = QLabel("学习率:")
        lr_label.setFont(QFont("Microsoft YaHei", 11))
        self.lr_input = QLineEdit("0.05")
        right_form.addRow(lr_label, self.lr_input)
        
        batch_label = QLabel("批次大小:")
        batch_label.setFont(QFont("Microsoft YaHei", 11))
        self.batch_input = QLineEdit("64")
        right_form.addRow(batch_label, self.batch_input)
        
        row1.addLayout(right_form)
        config_layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        row2.setSpacing(15)
        
        dev_label = QLabel("设备:")
        dev_label.setFont(QFont("Microsoft YaHei", 11))
        row2.addWidget(dev_label)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        self.device_combo.setMaximumWidth(80)
        row2.addWidget(self.device_combo)
        
        ls_label = QLabel("本地轮次:")
        ls_label.setFont(QFont("Microsoft YaHei", 11))
        row2.addWidget(ls_label)
        self.local_epochs_input = QLineEdit("1")
        self.local_epochs_input.setMaximumWidth(50)
        row2.addWidget(self.local_epochs_input)
        
        jr_label = QLabel("参与率:")
        jr_label.setFont(QFont("Microsoft YaHei", 11))
        row2.addWidget(jr_label)
        self.join_ratio_input = QLineEdit("1")
        self.join_ratio_input.setMaximumWidth(50)
        row2.addWidget(self.join_ratio_input)
        
        eg_label = QLabel("评估间隔:")
        eg_label.setFont(QFont("Microsoft YaHei", 11))
        row2.addWidget(eg_label)
        self.eval_gap_input = QLineEdit("1")
        self.eval_gap_input.setMaximumWidth(50)
        row2.addWidget(self.eval_gap_input)
        
        mu_label = QLabel("μ:")
        mu_label.setFont(QFont("Microsoft YaHei", 11))
        row2.addWidget(mu_label)
        self.mu_input = QLineEdit("0.1")
        self.mu_input.setMaximumWidth(50)
        row2.addWidget(self.mu_input)
        
        row2.addStretch()
        config_layout.addLayout(row2)
        
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
        stats_group = QGroupBox("训练状态")
        stats_layout = QVBoxLayout()
        
        status_row = QHBoxLayout()
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.status_label.setStyleSheet("color: #27ae60;")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        
        self.round_label = QLabel("轮次: 0/0")
        self.round_label.setFont(QFont("Microsoft YaHei", 11))
        status_row.addWidget(self.round_label)
        
        self.acc_label = QLabel("准确率: 0.00%")
        self.acc_label.setFont(QFont("Microsoft YaHei", 11))
        status_row.addWidget(self.acc_label)
        
        self.loss_label = QLabel("损失: 0.000")
        self.loss_label.setFont(QFont("Microsoft YaHei", 11))
        status_row.addWidget(self.loss_label)
        
        stats_layout.addLayout(status_row)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                background-color: #ecf0f1;
                height: 25px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 5px;
            }
        """)
        stats_layout.addWidget(self.progress_bar)
        
        stats_group.setLayout(stats_layout)
        main_layout.addWidget(stats_group)
        
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(250)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                border: 2px solid #34495e;
                border-radius: 5px;
                padding: 8px;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        
        self.start_btn = QPushButton("开始训练")
        self.start_btn.setMinimumHeight(45)
        self.start_btn.clicked.connect(self.start_training)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setMinimumHeight(45)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        btn_layout.addWidget(self.stop_btn)
        
        self.clear_btn = QPushButton("清空日志")
        self.clear_btn.setObjectName("clear_btn")
        self.clear_btn.setMinimumHeight(45)
        self.clear_btn.clicked.connect(self.clear_log)
        btn_layout.addWidget(self.clear_btn)
        
        main_layout.addLayout(btn_layout)
        
        self.statusBar().showMessage("就绪")
        self.statusBar().setStyleSheet("QStatusBar { background-color: #ecf0f1; color: #2c3e50; font-size: 11px; }")
        
    def append_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        self.parse_training_info(message)
        
    def parse_training_info(self, message):
        try:
            if 'Test Accuracy' in message or 'test_acc' in message.lower():
                import re
                acc_match = re.search(r'(\d+\.\d+)%', message)
                if acc_match:
                    acc = float(acc_match.group(1))
                    self.acc_label.setText(f"准确率: {acc:.2f}%")
                    
            if 'Round' in message:
                import re
                round_match = re.search(r'Round[:\s]+(\d+)', message, re.IGNORECASE)
                if round_match:
                    self.current_round = int(round_match.group(1))
                    self.round_label.setText(f"轮次: {self.current_round}/{self.total_rounds}")
                    progress = int((self.current_round / self.total_rounds) * 100)
                    self.progress_bar.setValue(progress)
                    
            if 'Loss' in message:
                import re
                loss_match = re.search(r'Loss[:\s]+(\d+\.\d+)', message, re.IGNORECASE)
                if loss_match:
                    loss = float(loss_match.group(1))
                    self.loss_label.setText(f"损失: {loss:.3f}")
        except:
            pass
        
    def start_training(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("状态: 训练中...")
        self.status_label.setStyleSheet("color: #e67e22;")
        
        self.total_rounds = int(self.rounds_input.text())
        self.current_round = 0
        self.progress_bar.setValue(0)
        
        self.append_log("=" * 50)
        self.append_log("开始训练...")
        self.append_log(f"算法: {self.algo_combo.currentText()}")
        self.append_log(f"模型: {self.model_combo.currentText()}")
        self.append_log(f"数据集: {self.dataset_combo.currentText()}")
        self.append_log(f"轮次: {self.total_rounds}")
        self.append_log(f"客户端: {self.clients_input.text()}")
        self.append_log("=" * 50)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_v2_path = os.path.join(script_dir, "main_v2.py")
        
        args = [
            sys.executable,
            main_v2_path,
            "-algo", self.algo_combo.currentText(),
            "-m", self.model_combo.currentText(),
            "-data", self.dataset_combo.currentText(),
            "-gr", self.rounds_input.text(),
            "-nc", self.clients_input.text(),
            "-lr", self.lr_input.text(),
            "-lbs", self.batch_input.text(),
            "-dev", self.device_combo.currentText(),
            "-ls", self.local_epochs_input.text(),
            "-jr", self.join_ratio_input.text(),
            "-eg", self.eval_gap_input.text(),
            "-mu", self.mu_input.text(),
            "-fs", "0",
        ]
        
        self.append_log(f"命令: {' '.join(args)}")
        
        self.training_thread = TrainingThread(args, script_dir)
        self.training_thread.log_signal.connect(self.append_log)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()
        
    def stop_training(self):
        if self.training_thread:
            self.training_thread.stop()
            self.training_thread = None
            
        self.status_label.setText("状态: 已停止")
        self.status_label.setStyleSheet("color: #e74c3c;")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.append_log("训练已停止")
        
    def training_finished(self):
        self.status_label.setText("状态: 完成")
        self.status_label.setStyleSheet("color: #27ae60;")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.append_log("=" * 50)
        self.append_log("训练完成!")
        self.append_log("=" * 50)
        self.progress_bar.setValue(100)
        
    def clear_log(self):
        self.log_text.clear()
        self.append_log("日志已清空")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = FedProxApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

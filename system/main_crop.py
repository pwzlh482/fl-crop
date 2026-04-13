#!/usr/bin/env python
"""
main_crop.py - Crop 联邦学习快捷入口
直接调用 main_v2.py，默认参数为 FedProxV2 + MobileNet + Cifar10

用法:
    python main_crop.py                    # 默认参数直接跑
    python main_crop.py -gr 100 -mu 0.01   # 传参覆盖
"""

import os
import sys

# 确保在 system 目录下运行
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from main_v2 import main

if __name__ == "__main__":
    main()

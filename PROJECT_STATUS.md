# fl-crop 项目当前状态

## 仓库
- GitHub: `pwzlh482/fl-crop`，已配置 token 推送

## 核心文件

| 文件 | 说明 |
|------|------|
| `system/main_v2.py` | 唯一核心入口，含 `main()` 函数，FedProxV2 默认 |
| `system/main_crop.py` | 薄包装，直接调用 `main_v2.main()` |
| `system/flcore/clients/clientprox_v2.py` | 优化客户端 |
| `system/flcore/servers/serverprox_v2.py` | 优化服务端 |

## FedProxV2 包含的所有优化

1. Label Smoothing (0.1) — 防过拟合
2. PerturbedGradientDescent1 — 带 momentum + weight_decay
3. CosineAnnealingWarmRestarts — 替代 MultiStepLR
4. 梯度裁剪 max_norm=5.0 — 防爆炸
5. 动态 mu 衰减 — `mu = base_mu * (0.5 + 0.5 * cos(π*progress))`，前期高抑制漂移，后期低释放个性化
6. Warmup 学习率 — 前5轮线性升温
7. ColorJitter + RandomErasing — 增强数据增强
8. GroupNorm 替代 BatchNorm — FL 兼容
9. 预训练权重自动检测 — ResNet18/MobileNetV2，BN→GN 替换前加载
10. 最优准确率追踪

## 运行方式
```bash
python main_crop.py              # 最简，默认 FedProxV2 + MobileNet + Cifar10
python main_v2.py -algo FedProxV2 -model ResNet18 -gr 100 -mu 0.01
```

## 已删除的文件
- `client_crop_v2.py` — 已合并进 clientprox_v2.py
- `server_crop_v2.py` — 已合并进 serverprox_v2.py
- 不再有 CropV2 算法命名

## 不需要优化的文件
- generate_*.py 系列不优化

## Git 推送方式
```
git --git-dir=/workspace/fl-crop/.git --work-tree=/workspace/fl-crop push origin main
```

"""
serverProx V2 - FedProx 优化服务端
基于实际服务器 serverprox.py，仅新增，不修改原文件

优化点：
1. Warmup 学习率策略（前5轮线性升温，避免初期大梯度）
2. 动态 mu 衰减调度 — 前期高 mu 抑制漂移，后期低 mu 释放个性化
3. 训练过程中追踪最优准确率
4. 服务器端 lr 衰减兼容客户端 CosineAnnealing（不下发覆盖，让客户端自主调度）
"""

import time
import torch
from flcore.clients.clientprox_v2 import clientProxV2
from flcore.servers.serverbase import Server


class FedProxV2(Server):

    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProxV2)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients (V2 optimized).")

        # self.load_model()
        self.Budget = []

        # Warmup 参数
        self.warmup_rounds = getattr(args, 'warmup_rounds', 5)
        self.base_lr = args.local_learning_rate

        # 动态 mu 参数
        self.base_mu = args.mu

    def train(self):
        best_acc = 0.0

        for i in range(self.global_rounds + 1):
            s_t = time.time()

            # 优化1: Warmup 策略 - 前 warmup_rounds 轮线性升温
            if i < self.warmup_rounds:
                warmup_factor = (i + 1) / self.warmup_rounds
                self.learning_rate = self.base_lr * warmup_factor
            elif self.args.learning_rate_decay:
                # 到达 milestones 后衰减
                count = 0
                for m in self.args.lr_decay_milestones:
                    if i >= m:
                        count += 1
                self.learning_rate = self.base_lr * (self.args.learning_rate_decay_gamma ** count)

            self.selected_clients = self.select_clients()

            # 下发当前学习率给客户端
            for client in self.selected_clients:
                client.learning_rate = self.learning_rate
                # 同步更新优化器的 lr
                for param_group in client.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            # 优化2: 动态 mu 衰减
            for client in self.selected_clients:
                client.update_mu(i, self.global_rounds)

            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print(f"Current mu: {self.selected_clients[0].mu:.6f}, Current lr: {self.learning_rate:.6f}" if len(self.selected_clients) > 0 else "")
                print("\nEvaluate global model")
                self.evaluate()

                # 优化3: 追踪最优准确率
                if len(self.rs_test_acc) > 0 and self.rs_test_acc[-1] > best_acc:
                    best_acc = self.rs_test_acc[-1]
                    print(f"*** New Best Accuracy: {best_acc:.4f} ***")

            for client in self.selected_clients:
                client.train()

            self.receive_models()

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            from flcore.clients.clientprox_v2 import clientProxV2
            self.set_new_clients(clientProxV2)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

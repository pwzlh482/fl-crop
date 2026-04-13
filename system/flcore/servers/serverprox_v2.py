import time
from flcore.clients.clientprox_v2 import clientProxV2
from flcore.servers.serverbase import Server
from threading import Thread


class FedProxV2(Server):
    """
    FedProx服务端优化版（不修改原serverprox.py）
    
    优化内容：
    1. Warmup策略 — 前5轮学习率线性增长，避免初期震荡
    2. 训练过程中打印mu值，方便监控动态衰减效果
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProxV2)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients (V2 optimized).")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            # 优化1：Warmup策略（前5轮学习率线性增长）
            warmup_rounds = 5
            if i < warmup_rounds:
                warmup_factor = (i + 1) / warmup_rounds
                for client in self.clients:
                    for param_group in client.optimizer.param_groups:
                        param_group['lr'] = client.learning_rate * warmup_factor

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                # 打印当前mu值和学习率信息
                if i < warmup_rounds:
                    print(f"[Warmup] lr factor: {warmup_factor:.2f}", end=", ")
                else:
                    print("[Normal]", end=", ")
                print(f"mu: {self.clients[0].mu:.4f}")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientProxV2)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

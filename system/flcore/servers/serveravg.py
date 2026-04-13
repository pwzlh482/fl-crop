import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread

class FedAvg(Server):
    def __init__(self, args, times):

        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            # 打印轮次和选中的客户端数量（保留train函数的核心打印逻辑）
            print(f"\n-------------Round number: {i}-------------")
            print(f"Selected clients: {len(self.selected_clients)} / {self.num_clients}")

            # 按eval_gap频率评估全局模型（保留train1的评估逻辑）
            if i % self.eval_gap == 0:
                print("\nEvaluate global model")
                self.evaluate()

            # 客户端本地训练（强制执行，保留train的核心逻辑）
            for client in self.selected_clients:
                client.train()

            # 接收客户端模型（原train1的核心逻辑，处理慢客户端/掉线客户端）
            self.receive_models()

            # DLG评估（原train1逻辑）
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            # 聚合前检查：确保有模型可聚合，避免断言错误（修复核心报错）
            if len(self.uploaded_models) == 0:
                print("Warning: No client models uploaded, skip aggregation")
                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])
                continue

            # 聚合客户端参数
            self.aggregate_parameters()

            # 统计单轮耗时（原train1逻辑）
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            # 自动中断（原train1逻辑）
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # 训练结束后打印结果（替换不存在的print_final_results）
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        # 保存结果和模型（修复save_model不存在的报错，替换为原代码的save_global_model）
        self.save_results()
        self.save_global_model()

        # 新客户端微调与评估（原train1逻辑）
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
            

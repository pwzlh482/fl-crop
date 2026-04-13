from flcore.servers.serverprox import FedProx
from flcore.clients.clientprox_yolo import clientProxYolo

class FedProxYolo(FedProx):
    def __init__(self, args, times):
        super().__init__(args, times)
        # 强制指定使用检测客户端
        self.set_clients(clientProxYolo) 

    def evaluate(self):
        # 检测任务评估较慢，训练时建议打印Loss，最后再统一测mAP
        print("Evaluating Global YOLO Model...")
        # 这里可以调用 client.test_metrics()

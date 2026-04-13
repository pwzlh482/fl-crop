# flcore/servers/server_crop.py
# 仅适配PFLlib框架，复用原生Server逻辑
from flcore.servers.serveravg import FedAvg  # 导入PFLlib原生的FedAvg服务端

# 自定义ServerCrop类，继承ServerAvg
class ServerCrop(FedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)  # 直接复用原生初始化逻辑
        
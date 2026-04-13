import torch
import torch.optim as optim
from flcore.clients.clientbase import Client

class clientCrop(Client):
    def __init__(self, args, id, train_samples, test_samples, val_data, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.device = args.device
        # Initialize models
        from flcore.trainmodel.models import get_model
        self.resnet18 = get_model("ResNet18", num_classes=3).to(self.device)
        self.yolov8n = get_model("YOLOv8n", num_classes=3).to(self.device)
        # Optimizers
        self.opt_resnet = optim.Adam(self.resnet18.parameters(), lr=args.local_learning_rate)
        self.opt_yolo = optim.Adam(self.yolov8n.parameters(), lr=args.local_learning_rate / 10)
        # Validation data
        self.val_data = val_data
        self.model_params = [self.resnet18.state_dict(), self.yolov8n.state_dict()]

    def train(self):
        """Local training for dual models"""
        self.resnet18.train()
        self.yolov8n.train()
        for epoch in range(self.local_epochs):
            for (imgs, (cls_labels, det_annots)) in self.train_data:
                imgs = imgs.to(self.device, dtype=torch.float32)
                cls_labels = cls_labels.to(self.device, dtype=torch.long)
                # Train ResNet18
                self.opt_resnet.zero_grad()
                loss_cls, _ = self.resnet18.get_loss(imgs, cls_labels)
                loss_cls.backward()
                self.opt_resnet.step()
                # Train YOLOv8n
                self.opt_yolo.zero_grad()
                loss_det, _ = self.yolov8n.get_loss(imgs, det_annots)
                loss_det.backward()
                self.opt_yolo.step()
            # Training log
            print(f"Client {self.id} | Epoch {epoch+1}/{self.local_epochs} | ClsLoss: {loss_cls.item():.4f} | DetLoss: {loss_det.item():.4f}")
        # Update model parameters
        self.model_params = [self.resnet18.state_dict(), self.yolov8n.state_dict()]
import sys
import time 
import torch
from datetime import timedelta
import sklearn.metrics as metrics
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None, device=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device is not None:
            self.device = device

        self.model = model
        self.model = self.model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.cache = {
            "train_loss": [],
            "valid_loss": [],
            "train_acc": [],
            "valid_acc": [],
            "lr": []
        }
    #

    def load_checkpoint(self, path, only_load_backbone=False, load_optimizer=False, load_scheduler=False):
        params = torch.load(path)
        if only_load_backbone:
            params = params["model"]
            self.model.backbone.load_state_dict(params)
        else:
            self.model.load_state_dict(params["model"])
            if load_optimizer:
                self.optimizer.load_state_dict(params["optimizer"])
            if self.scheduler is not None and load_scheduler:
                self.scheduler.load_state_dict(params["scheduler"])
            self.cache = params["cache"]
        print("[+] Model load successful")
    #

    def save_checkpoint(self, path):
        params = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "cache": self.cache
        }
        torch.save(params, path)
    #

    def compute_metrics(self, preds, labels):
        accuracy = metrics.accuracy_score(labels, preds)
        precision = metrics.precision_score(labels, preds, average='micro')
        recall = metrics.recall_score(labels, preds, average='micro')
        f1 = metrics.f1_score(labels, preds, average='micro')
        return {
            'accuracy': round(accuracy, 3),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
        }
    #

    def forward(self, dataloader, fw_mode="train"):
        if fw_mode == "train":
            self.model.train()
        else:
            self.model.eval()

        cache = {"loss": [], "predicts": [], "labels": []}            
        N = len(dataloader)
        for idx, (images, labels) in enumerate(dataloader, 1):
            if fw_mode == "train":
                self.optimizer.zero_grad()

            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.set_grad_enabled(fw_mode=="train"):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                if fw_mode == "train":
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

            cache["loss"].append(loss.item())
            cache["predicts"] += logits.softmax(dim=-1).argmax(dim=-1).tolist()
            cache["labels"] += labels.tolist()

            print("\r", end="")
            print(f"{fw_mode.capitalize()} step: {idx} / {N}", end="" if idx != N else "\n")

        loss = sum(cache["loss"]) / len(cache["loss"])
        acc = self.compute_metrics(cache["predicts"], cache["labels"])
        self.cache[f"{fw_mode}_loss"].append(loss)
        self.cache[f"{fw_mode}_acc"].append(acc)
    #

    def fit(self, train_loader, valid_loader=None, epochs=10, checkpoint="./checkpoint.pt"):
        print(f"Running on: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Total update step: {len(train_loader) * epochs}")

        for epoch in range(1, epochs+1):
            start_time = time.time()
            print(f"Epoch: {epoch}")
            logs = []
            current_lr = f": {self.optimizer.param_groups[0]['lr']:.5}"
            try:
                self.forward(train_loader, "train")
                train_loss = round(self.cache["train_loss"][-1], 5)
                train_acc = [str(k) + ": " + str(v) for k, v in self.cache["train_acc"][-1].items()]
                train_acc = " - ".join(train_acc)
                logs.append(f"\t=> Train epoch: loss: {train_loss} - {train_acc}")
            except KeyboardInterrupt:
                sys.exit()
            if valid_loader is not None:
                try:
                    self.forward(valid_loader, "valid")
                    valid_loss = round(self.cache["valid_loss"][-1], 5)
                    valid_acc = [str(k) + ": " + str(v) for k, v in self.cache["valid_acc"][-1].items()]
                    valid_acc = " - ".join(valid_acc)
                    logs.append(f"\t=> Valid epoch: loss: {valid_loss} - {valid_acc}")
                except KeyboardInterrupt:
                    sys.exit()
            total_time = round(time.time() - start_time, 1)
            logs.append(f"\t=> Learning Rate: {current_lr} - Time: {timedelta(seconds=int(total_time))}/step\n")
            print("\n".join(logs))
            self.cache["lr"].append(current_lr)
            self.save_checkpoint(checkpoint)
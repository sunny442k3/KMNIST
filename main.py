import sys
import torch
import utils
import argparse
from model import CNNModelOptimal, Backbone, CNN
from resnet import ResNet
from trainer import Trainer 
from dataset import JPDDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Dataset use for training. Example: k10 or k49 for 10 or 49 classes corresponding', default='k49')
    parser.add_argument('--epoch', default=40, type=int)
    parser.add_argument('--img_size', default=28, type=int)
    parser.add_argument("--input_dim", default=1, type=int)
    parser.add_argument("--num_classes", default=49, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--backbone', default="cnn", help="Name of pretrained model. Example: cnn or rescnn")
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--checkpoint_path', default='./checkpoint.pt')
    parser.add_argument('--query_weight', default="./dataset/query_embedding_k49.pt")
    parser.add_argument('--freeze_backbone', action="store_true")
    parser.add_argument('--load_pretrained_weight', default="", type=str)
    parser.add_argument('--load_backbone_weight', default="", type=str)
    args = parser.parse_args()
    return args


def create_loader(cf):
    train_images, train_labels, test_images, test_labels = utils.load_dataset(cf)
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((cf.img_size, cf.img_size)),
        T.RandomAffine(degrees = 15),
        T.ToTensor(),
    ])
    test_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((cf.img_size, cf.img_size)),
        T.ToTensor(),
    ])
    train_dataset = JPDDataset(train_images, train_labels.tolist(), train_transform)
    test_dataset = JPDDataset(test_images, test_labels.tolist(), test_transform)
    print(f"[+] Train sample: {len(train_dataset)} - Test sample: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cf.batch_size*10, shuffle=True)
    print(f"[+] Train batch: {len(train_loader)} - Test batch: {len(test_loader)}")
    return train_loader, test_loader
#
    
def main():
    cf = config()
    train_loader, test_loader = create_loader(cf)
    if cf.train_backbone:
        assert len(cf.backbone.strip()), "Please give name of backbone when active train_backbone mode"
        model = CNN(cf.num_classes)
    else:
        model = CNNModelOptimal(cf)
    
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cf.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=cf.learning_rate, 
        steps_per_epoch=len(train_loader), 
        epochs=cf.epoch,
        pct_start=0.2
    )
    trainer = Trainer(model, optimizer, criterion, scheduler)
    if len(cf.load_pretrained_weight.strip()):
        trainer.load_checkpoint(cf.load_pretrained_weight.strip())
    try:
        trainer.fit(train_loader, test_loader, cf.epoch, cf.checkpoint_path)
    except KeyboardInterrupt:
        sys.exit()

if __name__ == "__main__":
    main()
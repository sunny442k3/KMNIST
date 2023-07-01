import utils
import argparse
from trainer import Trainer 
from dataset import JPDDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Dataset use for training. Example: k10 or k49 for 10 or 49 classes corresponding', default='k49')
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--learning_rate', default=0.005)
    parser.add_argument('--checkpoint', default='./checkpoint.pt')
    args = parser.parse_args()
    return args


def create_loader(cf):
    train_images, train_labels, test_images, test_labels = utils.load_dataset(cf)
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    transform = T.Compose([
        T.ToPILImage(),
        T.RandomAffine(degrees = 30),
        T.RandomPerspective(),
        T.ToTensor(),
        T.Normalize(train_images.mean(), train_images.std())
    ])
    train_dataset = JPDDataset(train_images, train_labels, transform)
    test_dataset = JPDDataset(test_images, test_labels)
    print(f"[+] Train sample: {len(train_dataset)} - Test sample: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cf.batch_size*2, shuffle=True, pin_memory=True)
    print(f"[+] Train batch: {len(train_loader)} - Test batch: {len(test_loader)}")
    return train_loader, test_loader
    
def main():
    cf = config()
    train_loader, test_loader = create_loader(cf)
    


if __name__ == "__main__":
    main()
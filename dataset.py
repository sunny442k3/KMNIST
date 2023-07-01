import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class JPDDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        super(JPDDataset, self).__init__()
        self.transform = transform
        if transform is None:
            self.transform = transform = T.Compose([
                T.ToPILImage(),
                T.ToTensor()
            ])
        self.images = torch.tensor(images)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        image = self.images[idx].clone().detach()
        image = self.transform(image.float())
        label = self.labels[idx]
        return image, label
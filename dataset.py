import torch
from torch.utils.data import Dataset

class JPDDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        super(JPDDataset, self).__init__()
        self.transform = transform
      
        self.images = torch.from_numpy(images)
        self.labels = labels.tolist()

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform is not None:
            image = self.transform(image.float())
        label = torch.tensor(self.labels[idx])
        return image, label
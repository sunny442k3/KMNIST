import torch
import utils
import numpy as np
from model import CNNModelOptimal
from dataset import JPDDataset
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam import GradCAM

def gradcam(model, gradcam_obj, layers, num_classes, dataset, N=5, use_cuda=False, show_labels=False, idx_to_label=None, **gradcam_params):
    targets = [ClassifierOutputSoftmaxTarget(i) for i in range(num_classes)]
    random_indices = np.random.randint(0, len(dataset), N)
    random_indices = [idx for idx,i in enumerate(dataset) if i[1].item() == 1][:N]
    samples = [dataset[idx][0].unsqueeze(0) for idx in random_indices]
    input_tensor = torch.cat(samples, dim=0)
    
    if show_labels:
        labels = [dataset[idx][1].item() for idx in random_indices]
        if idx_to_label:
            labels = [idx_to_label[str(label)] for label in labels]
    
    for idx, layer in enumerate(layers):
        target_layers = [layer]
        cam = gradcam_obj(model=model, target_layers=target_layers, use_cuda=use_cuda)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, **gradcam_params)
        images = [input_tensor[idx].permute(1,2,0).numpy() for idx in range(N)]
        grayscaled_cam = [grayscale_cam[idx,:] for idx in range(N)]
        heatmaps_on_inputs = [show_cam_on_image(img, cam) for img,cam in zip(images, grayscaled_cam)]
        viz_img_list = [images, grayscaled_cam, heatmaps_on_inputs]
        subfig_titles = ["Input Images", "Grayscaled Heatmap", "Heatmaps on the Inputs"]
        fig = plt.figure(figsize=(14, 6))
        subfigs = fig.subfigures(nrows=3, ncols=1)
        fig.suptitle(f'GradCAM for layer: {idx+1}', fontsize=18, y=1.05)
        for subfig_idx, subfig in enumerate(subfigs):
            subfig.suptitle(subfig_titles[subfig_idx], y=1)
            viz_list = viz_img_list[subfig_idx]
            axs = subfig.subplots(nrows=1, ncols=N)
            for idx in range(N):
                axs[idx].imshow(viz_list[idx], cmap='gray')
                if show_labels:
                    axs[idx].set_title(labels[idx])
                axs[idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.savefig(f"./results/gradcam_layer_{idx}.png")
        plt.show()
#

class cf:
    dataset = "k10"
    input_dim = 1
    num_classes = 10
    img_size = 28
    use_cuda = True
    backbone_weight = "./checkpoint/cnn/k10_mldecoder_full_v2.pt"
    load_backbone_weight = ""
    freeze_backbone = False
    query_weight = "./dataset/query_embedding_k10.pt"

def main():
    model = CNNModelOptimal(cf)
    model.load_state_dict(torch.load(cf.backbone_weight, map_location='cpu')["model"])
    train_images, train_labels, test_images, test_labels = utils.load_dataset(cf)
    test_images = test_images / 255.0
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cf.img_size, cf.img_size)),
        transforms.ToTensor(),
    ])
    test_dataset = JPDDataset(test_images.tolist(), test_labels.tolist(), transform)
    idx_to_label = {"0": "o", "1": "ki", "2": "su", "3": "tsu", "4": "na", "5": "ha", "6": "ma", "7": "ya", "8": "re", "9": "wo"}
    cnn_layers = [model.backbone.cnn_block1, model.backbone.cnn_block2, model.backbone.cnn_block3]
    gradcam(model, GradCAM, cnn_layers, cf.num_classes, test_dataset, N=10, use_cuda=True,show_labels=True, idx_to_label=idx_to_label, aug_smooth=True, eigen_smooth=True)

if __name__ == "__main__":
    main()
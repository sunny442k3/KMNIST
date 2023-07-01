import os
import numpy as np
import requests
import gdown


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, total, unit): return x
    print('**** Could not import tqdm. Please install tqdm for download progressbars! (pip install tqdm) ****')


def drive_download(idx, output):
    url = 'https://drive.google.com/uc?id=' + idx
    gdown.download(url, output, quiet=False)
#

def download_dataset(num_classes, extention, root_path="./dataset"):
    dataset_download_dict = {
        '10_classes': {
            'gz':
            ['http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
             'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
             'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
             'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz'],
            'npz':
            ['http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz',
             'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz',
             'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz',
             'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz'],
        },
        '49_classes': {
            'npz':
            ['http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
             'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
             'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
             'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz'],
        },
        '3832_classes': {
            'tar':
            ['http://codh.rois.ac.jp/kmnist/dataset/kkanji/kkanji.tar'],
        }

    }
    if f"{num_classes}_classes" not in list(dataset_download_dict.keys()):
        print(f"Dataset with '{num_classes}' not available")
        return
    if extention not in list(dataset_download_dict[f"{num_classes}_classes"].keys()):
        print(f"Dataset with '.{extention}' format is not available ")
        return

    url_list = dataset_download_dict[f"{num_classes}_classes"][extention]
    for url in url_list:
        path = url.split('/')[-1]
        r = requests.get(url, stream=True)
        path = path.replace("kmnist", "k10")
        with open(f"{root_path}/{path}", 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))

            for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
                if chunk:
                    f.write(chunk)
    print('All dataset files downloaded!')
#


def load_dataset(cf):
    # Create dataset folder when not exist 
    if not os.path.exists("./dataset"): 
        os.makedirs("./dataset")
    
    # Download dataset if do not download previous
    dataset = str(cf.dataset).strip()
    assert dataset in ["k10", "k49"], "Argument 'dataset' must be in ['k10', 'k49']"
    if not os.path.exists(f"./dataset/{dataset}"):
        os.makedirs(f"./dataset/{dataset}")
        download_dataset(dataset[1:], "npz", f"./dataset/{dataset}")
    
    root_path = f"./dataset/{dataset}/{dataset}-"
    all_path = [
        root_path + "train-imgs.npz", # train images path
        root_path + "test-imgs.npz", # test images path
        root_path + "train-labels.npz", # train labels path
        root_path + "test-labels.npz" # test labels path
    ]
    for path in all_path:
        if not os.path.exists(path):
            download_dataset(dataset[1:], "npz", f"./dataset/{dataset}")
            break 

    # Load train and test dataset from folder
    train_images = np.load(f"./dataset/{dataset}/{dataset}-train-imgs.npz")['arr_0']
    train_labels = np.load(f"./dataset/{dataset}/{dataset}-train-labels.npz")['arr_0']
    test_images = np.load(f"./dataset/{dataset}/{dataset}-test-imgs.npz")['arr_0']
    test_labels = np.load(f"./dataset/{dataset}/{dataset}-test-labels.npz")['arr_0']
    return train_images, train_labels, test_images, test_labels
#
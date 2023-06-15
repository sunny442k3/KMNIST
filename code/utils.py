import requests
import gdown


try:
    from tqdm import tqdm
except ImportError:
    # If tqdm doesn't exist, replace it with a function that does nothing
    def tqdm(x, total, unit): return x
    print('**** Could not import tqdm. Please install tqdm for download progressbars! (pip install tqdm) ****')

def drive_download(idx, output):
    url = 'https://drive.google.com/uc?id=' + idx
    gdown.download(url, output, quiet=False)

def download_list(num_classes, extention):
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
        with open(path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))

            for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
                if chunk:
                    f.write(chunk)
    print('All dataset files downloaded!')

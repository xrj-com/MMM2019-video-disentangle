import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image

trans = transforms.Compose(
    [   
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    )
dataset = torchvision.datasets.ImageFolder("./dataset/LASIESTA/", transform=trans)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, extensions=IMG_EXTENSIONS):
    images = {}
    for value in class_to_idx.values():
        images[value]=[]        
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    images[class_to_idx[target]].append(path)
    return images

class scenePairs_dataset(Dataset):
    def __init__(self, root_dir, epoch_size, max_step, delta=1,random_classes=False, loader=default_loader, transform=trans):
        self.root_dir = root_dir
        self.classes, self.class_to_idx = find_classes(root_dir)
        self.imagesDict = make_dataset(root_dir, self.class_to_idx)
        self.frameNumDict = {}
        for c in self.classes:
            self.frameNumDict[self.class_to_idx[c]] = len(self.imagesDict[self.class_to_idx[c]])

        self.transform = transform
        self.max_step = max_step
        self.epoch_size = epoch_size
        self.delta = delta
        self.random_classes = random_classes
        self.loader = loader
    
    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):

        video_num = len(self.classes)

        vid = np.random.randint(video_num)
        max_trial = 10
        while self.frameNumDict[vid] <= self.max_step:
            vid = np.random.randint(video_num)
            max_trial -= 1
            assert max_trial>0, "max_step maybe too large"

        fn = self.frameNumDict[vid]

        start = np.random.randint(fn-self.max_step)
        k1 = np.random.randint(self.max_step)+1
        k2 = np.random.randint(self.max_step)+1
        end1 = start + k1
        end2 = start + k2
        
        imgPath_start = self.imagesDict[vid][start]
        imgPath_end1 = self.imagesDict[vid][end1]
        imgPath_end2 = self.imagesDict[vid][end2]

        img_start = self.loader(imgPath_start)
        img_end1 = self.loader(imgPath_end1)
        img_end2 = self.loader(imgPath_end2)
        if self.transform is not None:
            img_start = self.transform(img_start)
            img_end1 = self.transform(img_end1)
            img_end2 = self.transform(img_end2)

        return img_start, img_end1, img_end2, 1
    
class plot_dataset(Dataset):
    def __init__(self, root_dir, sample_num, max_step, delta=1, random_classes=False, loader=default_loader, transform=trans):
        self.root_dir = root_dir
        self.classes, self.class_to_idx = find_classes(root_dir)
        self.imagesDict = make_dataset(root_dir, self.class_to_idx)
        self.frameNumDict = {}
        for c in self.classes:
            self.frameNumDict[self.class_to_idx[c]] = len(self.imagesDict[self.class_to_idx[c]])

        self.transform = transform
        self.max_step = max_step
        self.batch_size = sample_num
        self.delta = delta
        self.random_classes = random_classes
        self.loader = loader

        self.video_list = []
        self.image_list = []
        for i in range(self.batch_size):
            video_num = len(self.classes)
            vid = np.random.randint(video_num)
            max_trial = 10
            while self.frameNumDict[vid] <= self.max_step:
                vid = np.random.randint(video_num)
                max_trial -= 1
                assert max_trial>0, "max_step maybe too large"

            self.video_list.append(vid)
            fn = self.frameNumDict[vid]

            start = np.random.randint(fn - self.max_step)
            # for j in range(self.max_step):
            self.image_list += self.imagesDict[vid][start : self.max_step + start]


    
    def __len__(self):
        return self.batch_size * self.max_step

    def __getitem__(self, index):
        img = self.loader(self.image_list[index])
        if self.transform is not None:
            img = self.transform(img)
        return img


if __name__ == "__main__":
    dir = "./dataset/LASIESTA/"
    # sg = scenePairs_dataset(dir, 500, 10, 10)

    # test = DataLoader(sg, batch_size=5)
    # for img1, img2, target in test:
    #     t[2][3:]=0
    #     print(t[2])
    #     break
    plot_d = plot_dataset(dir, 5, 20)

    print(plot_d[1])
    print(plot_d[99].size())
    print(len(plot_d))
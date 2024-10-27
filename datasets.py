import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
from natsort import natsorted
#————————————————————CHANGE————————————————————
#from fast_autoaug_torch.augment import Augmentation
#from fast_autoaug_torch.search_policy import train_network
import json
import argparse
#————————————————————CHANGE————————————————————

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


#————————————————————CHANGE————————————————————
def load_or_search_fastaa_policy():
    parser = argparse.ArgumentParser(description='FastAA Policy')
    parser.add_argument('--reduced-size', type=int, default=60000)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--nfolds', type=int, default=8)
    parser.add_argument('--num-trials', type=int, default=200)
    args, _ = parser.parse_known_args()

    try:
        with open('fastaa_policy.json', 'r') as f:
            fastaa_policy = json.load(f)
    except FileNotFoundError:
        search_args = {
            'reduced_size': args.reduced_size,
            'epochs': args.epochs,
            'nfolds': args.nfolds,
            'num_trials': args.num_trials,
        }
        fastaa_policy = train_network(search_args)
        with open('fastaa_policy.json', 'w') as f:
            json.dump(fastaa_policy, f)
    return fastaa_policy

class FastAAWrapper:
    def __init__(self, policy):
        self.augment = Augmentation(policy)

    def __call__(self, img):
        return self.augment(img)
#————————————————————CHANGE————————————————————

class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            # train
            self.files = natsorted(sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_train)))
        else:
            # test
            self.files = sorted(glob.glob(c.VAL_PATH + "/*." + c.format_val))



    def __getitem__(self, index):
        try:
            img = Image.open(self.files[index])
            img = to_rgb(img)
            item = self.transform(img)
            return item

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        if self.mode == 'shuffle':
            return max(len(self.files_cover), len(self.files_secret))

        else:
            return len(self.files)
        
#————————————————————CHANGE————————————————————
# fastaa_policy = load_or_search_fastaa_policy()
# fastaa_wrapper = FastAAWrapper(fastaa_policy)
#————————————————————CHANGE———————————————————— 

#-----------------CHANGE-----------------
# 导入必要的库
from torchvision.transforms import autoaugment, transforms

# 定义AutoAugment策略   中等数据集规模 不需要过强的数据增强
#auto_augment = autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET)

# 定义RandAugment
rand_augment = autoaugment.RandAugment(num_ops=2, magnitude=5)

# 更新transform
transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomCrop(c.cropsize),
    #auto_augment,
    rand_augment,
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])
#__________________________________________________




#-----------------CHANGE-----------------
#transform = T.Compose([
#      T.RandomHorizontalFlip(),
#    T.RandomVerticalFlip(),
#    T.RandomCrop(c.cropsize),
#    T.ToTensor()
    #————————————————————CHANGE———————————————————— 
  #  T.RandomHorizontalFlip(),
  #  fastaa_wrapper,
  #  T.ToTensor(),
  #  T.Normalize((0.5,), (0.5,))
    #————————————————————CHANGE————————————————————
#])

transform_val = T.Compose([
    T.CenterCrop(c.cropsize_val),
    T.ToTensor(),
])


# Training data loader
# trainloader = DataLoader(
#     Hinet_Dataset(transforms_=transform, mode="train"),
#     batch_size=c.batch_size,
#     shuffle=True,
#     pin_memory=True,
# #    num_workers=8,
#     drop_last=True
# )
trainloader = DataLoader(
    Hinet_Dataset(transforms_=transform, mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)
# Test data loader
testloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)
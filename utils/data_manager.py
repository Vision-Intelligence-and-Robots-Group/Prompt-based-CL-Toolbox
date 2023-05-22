import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from utils.data import iCIFAR100_vit,i5Datasets_vit, iImageNetR, iCore50, CDDB, iDomainNet
from torch.utils.data import DataLoader

def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    # pinna
    # cifar100_vit
    if name == "cifar100_vit":
        return iCIFAR100_vit(args)
    elif name == "5datasets_vit":
        return i5Datasets_vit(args)
    elif name == "core50":
        return iCore50(args)
    elif name == "imagenetr":
        return iImageNetR(args)
    elif name == "cddb":
        return CDDB(args)
    elif name == "domainnet":
        return iDomainNet(args)
    else:
        raise NotImplementedError('Unknown dataset {}.'.format(dataset_name))

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args=None):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), 'No enough classes.'
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

        self.continual_data = list()

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]
    
    def build_continual_dataloader(self):
        for task in range(self.nb_tasks):
            dataset_train = self.get_dataset(np.arange(sum(self._increments[:task]), sum(self._increments[:task+1])), source='train',mode='train')
            dataset_test = self.get_dataset(np.arange(0, sum(self._increments[:task+1])), source='test', mode='test')
            if self.args.distributed and self.args.world_size > 1:
                sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True) 
                sampler_val = torch.utils.data.SequentialSampler(dataset_test)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
                sampler_val = torch.utils.data.SequentialSampler(dataset_test)

            train_loader = DataLoader(dataset_train, sampler=sampler_train,
                                      batch_size=self.args.batch_size, 
                                      num_workers=self.args.num_workers, 
                                      pin_memory=self.args.pin_mem)
            
            test_loader = DataLoader(dataset_test, sampler=sampler_val,
                            batch_size=self.args.batch_size, 
                            num_workers=self.args.num_workers, 
                            pin_memory=self.args.pin_mem)
            self.continual_data.append({'train': train_loader, 'test': test_loader})

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = self._build_transform(True, self.args)
        elif mode == 'test':
            trsf = self._build_transform(False, self.args)
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)


    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]
    
    def _build_transform(self, is_train, args):
        # resize_im = args.input_size > 32
        if is_train:
            scale = (0.05, 1.0)
            ratio = (3. / 4., 4. / 3.)
            t = []
            t.append(transforms.Resize(256))
            if self.args.color_jitter:
                t.append(transforms.ColorJitter(brightness=63/255))
            t.append(transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio))
            t.append(transforms.RandomHorizontalFlip(p=0.5))
            t.append(transforms.ToTensor())
            if self.args.normalize:
                t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

            return transforms.Compose(t)
            
            # return transform

        t = []
        t.append(transforms.Resize(256))
        t.append(transforms.CenterCrop(224))
        t.append(transforms.ToTensor())
        if self.args.normalize:
                t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        return transforms.Compose(t)

class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label

def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # if img.format is 'PNG' and img.mode is not 'RGBA':
        #     img = img.convert('RGBA')
        #     background = Image.new('RGBA', img.size, (255, 255, 255))
        #     img = Image.alpha_composite(background, img)
        return img.convert('RGB')


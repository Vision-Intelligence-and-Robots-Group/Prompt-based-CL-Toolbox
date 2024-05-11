import os
import numpy as np
from torchvision import datasets, transforms
from utils.datautils.core50.core50data import CORE50
from utils.toolkit import split_images_labels
from PIL import Image

class iData(object):
    class_order = None

class iCore50(iData):
    use_path = False
    def __init__(self, args):
        self.args = args
        class_order = np.arange(8 * 50).tolist()
        self.class_order = class_order

    def download_data(self):
        datagen = CORE50(root=self.args.data_path, scenario="ni")
        dataset_list = []
        for i, train_batch in enumerate(datagen):
            imglist, labellist = train_batch
            labellist += i*50
            imglist = imglist.astype(np.uint8)
            dataset_list.append([imglist, labellist])
        train_x = np.concatenate(np.array(dataset_list, dtype=object)[:, 0])
        train_y = np.concatenate(np.array(dataset_list, dtype=object)[:, 1])
        self.train_data = train_x
        self.train_targets = train_y

        test_x, test_y = datagen.get_test_set()
        test_x = test_x.astype(np.uint8)
        self.test_data = test_x
        self.test_targets = test_y

class iImageNetR(iData):
    use_path = True
    def __init__(self, args):
        self.args = args
        class_order = np.arange(200).tolist()
        self.class_order = class_order

    def download_data(self):
        rootdir = self.args.data_path

        train_txt = './utils/datautils/imagenet-r/train.txt'
        test_txt = './utils/datautils/imagenet-r/test.txt'

        train_images = []
        train_labels = []
        with open(train_txt, 'r') as dict_file:
            for line in dict_file:
                (key, value) = line.strip().split('\t')
                train_images.append(os.path.join(rootdir, value))
                train_labels.append(int(key))
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)

        test_images = []
        test_labels = []
        with open(test_txt, 'r') as dict_file:
            for line in dict_file:
                (key, value) = line.strip().split('\t')
                test_images.append(os.path.join(rootdir, value))
                test_labels.append(int(key))
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

        self.train_data = train_images
        self.train_targets = train_labels
        self.test_data = test_images
        self.test_targets = test_labels

class iCIFAR100_vit(iData):
    use_path = False
    def __init__(self, args):
        self.args = args
        class_order = np.arange(100).tolist()
        self.class_order = class_order

    def download_data(self):
        train_dataset = datasets.CIFAR100(self.args.data_path, train=True, download=True)
        test_dataset = datasets.CIFAR100(self.args.data_path, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class CDDB(object):
    use_path = True
    def __init__(self, args):
        self.args = args
        class_order = args.class_order
        self.class_order = class_order

    def download_data(self):
        train_dataset = []
        test_dataset = []
        for id, name in enumerate(self.args.task_name):
            root_ = os.path.join(self.args.data_path, name, 'train')
            sub_classes = os.listdir(root_) if self.args.multiclass[id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    train_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    train_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        for id, name in enumerate(self.args.task_name):
            root_ = os.path.join(self.args.data_path, name, 'val')
            sub_classes = os.listdir(root_) if self.args.multiclass[id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    test_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    test_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)

class iDomainNet(iData):
    use_path = True
    def __init__(self, args):
        self.args = args
        class_order = np.arange(6 * 345).tolist()
        self.class_order = class_order
        self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]

    def download_data(self):
        self.image_list_root = self.args.data_path

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.test_data = np.array(train_x)
        self.test_targets = np.array(train_y)

class iCore50(iData):
    use_path = False
    def __init__(self, args):
        self.args = args
        class_order = np.arange(8 * 50).tolist()
        self.class_order = class_order

    def download_data(self):
        datagen = CORE50(root=self.args.data_path, scenario="ni")
        dataset_list = []
        for i, train_batch in enumerate(datagen):
            imglist, labellist = train_batch
            labellist += i*50
            imglist = imglist.astype(np.uint8)
            dataset_list.append([imglist, labellist])
        train_x = np.concatenate(np.array(dataset_list, dtype=object)[:, 0])
        train_y = np.concatenate(np.array(dataset_list, dtype=object)[:, 1])
        self.train_data = train_x
        self.train_targets = train_y

        test_x, test_y = datagen.get_test_set()
        test_x = test_x.astype(np.uint8)
        self.test_data = test_x
        self.test_targets = test_y
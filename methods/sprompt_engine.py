import copy
from distutils.log import info
from importlib.abc import MetaPathFinder
from locale import normalize
from pickle import FALSE
from random import random, shuffle
import torch
import os

from torch.nn import functional as F
from torch import optim

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from torch.nn.parallel import DistributedDataParallel as DDP
from models.sinet import SiNet
from models.slinet import SliNet

batch = [128,64,32,32,32,16,16,16]

class sprompt_engine(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        if args.net_type == "slip":
            self._network = SliNet(args)
        elif args.net_type == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args.net_type))
        
        self.args = args
        self.EPSILON = args.EPSILON
        
        self.num_workers = args.num_workers
        self.init_epoch = args.init_epoch
        self.init_weight_decay = args.init_weight_decay
        self.epochs = args.epochs
        self.weight_decay = args.weight_decay

        self.batch_size = batch[self.world_size-1]
        self.batch_size_ori = args.batch_size
        self.init_lr = args.init_lr
        self.lrate = args.lrate

        self.all_keys = []
        
    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def incremental_train(self, train_loader, test_loader, data_manager, task_id):
        self._cur_task = task_id
        self._total_classes = self._known_classes + data_manager.get_task_size(task_id)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if "classifier_pool" + "." + str(self._network.numtask - 1) in name:
                param.requires_grad_(True)
            if "prompt_pool" + "." + str(self._network.numtask - 1) in name:
                param.requires_grad_(True)

        # settings for DDP
        self._network.to(self._device)
        if self.distributed:
            self._network = DDP(self._network, device_ids=[self.local_rank])
        
        # training
        self._train(train_loader, test_loader)

        if self.distributed:
            self._network = self._network.module

        self.clustering(train_loader)

    def _train(self, train_loader, test_loader):
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self.distributed:
            named_parameters = self._network.module.named_parameters()
            network_params = self._network.module.parameters()
        else:
            named_parameters = self._network.named_parameters()
            network_params = self._network.parameters()

        # Double check
        enabled = set()
        for name, param in named_parameters:
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if self._cur_task==0:
            optimizer = optim.SGD(network_params, momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            self.run_epoch = self.init_epoch
        else:
            optimizer = optim.SGD(network_params, momentum=0.9,lr=self.lrate,weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            self.run_epoch = self.epochs
        self.train_function(train_loader, test_loader, optimizer, scheduler)
    
    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        self._network.eval()
        for _, epoch in enumerate(prog_bar):
            if self.distributed and self.args.world_size > 1:
                train_loader.sampler.set_epoch(epoch)
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self._known_classes

                logits = self._network(inputs)['logits']
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy_domain(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc, test_acc)
            # prog_bar.set_description(info)

        logging.info(info)

    def clustering(self, dataloader):
        features = []
        for i, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                feature = self._network.extract_vector(inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))

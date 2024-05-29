import copy
from distutils.log import info
from importlib.abc import MetaPathFinder
from locale import normalize
from pickle import FALSE
from random import random, shuffle
import torch
import os

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.dist_utils import get_rank, get_world_size, is_main_process
import torch.distributed as dist
from models.apil_vit import APrompt

batch = [128,64,32,32,32,16,16,16]

class APIL(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = APrompt(args)
        self._network_ori = None
        self._old_network = None

        self.args = args
        self.EPSILON = args.EPSILON
        
        self.num_workers = args.num_workers
        self.init_epoch = args.init_epoch
        self.init_weight_decay = args.init_weight_decay
        self.epochs = args.epochs
        self.weight_decay = args.weight_decay
        self.T = args.T

        self.anchor_lr = args.anchor_lr
        self.anchor_epochs = args.anchor_epochs
        self.anchor_num = args.anchor_num
        self.sigma = args.sigma
        self.sample_num = args.sample_num
        self.cluster_num = args.cluster_num

        self.prompt = None
        self._class_means = None
        self.sel_data = None
        self.cluster_idx = None

        if self.args.unscale_lr:
            self.batch_size = args.batch_size
            self.batch_size_ori = args.batch_size
        else:

            self.batch_size = batch[get_world_size()-1]
            self.batch_size_ori = args.batch_size
        self.init_lr = args.init_lr
        self.lrate = args.lrate
        self.sched = args.sched
        self.local_rank = get_rank()
        
        
    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

        self._old_class_mean = self._class_means

    def incremental_train(self, train_loader, test_loader, data_manager, task_id):
        self._cur_task = task_id
        self._total_classes = self._known_classes + data_manager.get_task_size(task_id)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # finetune MLP
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if 'fc2' in name:
                param.requires_grad_(True)
            if "fc.weight" == name:
                param.requires_grad_(True)
            if "fc.bias" == name:
                param.requires_grad_(True)

        # train prompt
        self.anchor_train_distributed(data_manager)

        # settings for DDP
        self._network.to(self._device)
        if self.distributed:
            self._network = DDP(self._network, device_ids=[self.local_rank],find_unused_parameters=False,broadcast_buffers=False)

        # training
        self._train(train_loader, test_loader)
        if self.args.calculate_distributed:
            torch.distributed.barrier()

        if self.distributed:
            self._network = self._network.module

        # construct means
        self.construct_mean_distributed(data_manager)
        if self.args.calculate_distributed:
            torch.distributed.barrier()
        
        self.prompt, self.sel_data, self.cluster_idx = None, None, None

    def _train(self, train_loader, test_loader):
        # Double check
        enabled = set()
        if self.distributed:
            named_parameters = self._network.module.named_parameters()
            network_params = self._network.module.parameters()
        else:
            named_parameters = self._network.named_parameters()
            network_params = self._network.parameters()

        for name, param in named_parameters:
            if param.requires_grad:
                enabled.add(name)
        logging.info(f"Parameters to be updated: {enabled}")

        if self._cur_task == 0:
            optimizer = optim.SGD(network_params, momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            self._init_train(train_loader,test_loader,optimizer,scheduler)
        else:
            if self.sched:
                lr = self.lrate
            else:
                lr = self.lrate * 0.95 ** (self._cur_task - 1)
            optimizer = optim.SGD(network_params, momentum=0.9,lr=lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            self._balance_update(train_loader, test_loader, optimizer, scheduler)

    def extract_all_fea(self, loader, network):
        network.eval()
        network.to(self._device)
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            _vectors = tensor2numpy(network.extract_vector(_inputs.to(self._device)))

            vectors.append(_vectors)
            targets.append(_targets)

        vectors = np.concatenate(vectors, axis=0)
        targets = np.concatenate(targets, axis=0)
        return vectors, targets

    def extract_all_fea_with_anchor(self, loader, network):
        network.eval()
        network.to(self._device)
        vectors = []
        cluster_idx_list = []
        for _, _inputs, cluster_idx in loader:
            prompt = self.prompt[cluster_idx]
            _vectors = tensor2numpy(network.forward_with_anchor(_inputs.to(self._device), prompt.to(self._device))['fea'])

            vectors.append(_vectors)
            cluster_idx_list.append(cluster_idx)

        vectors = np.concatenate(vectors, axis=0)
        cluster_idx_list = np.concatenate(cluster_idx_list, axis=0)
        return vectors, cluster_idx_list
    
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.init_epoch))
        self._network.eval()         # do not update the network normalizer statistics
        for epoch in prog_bar:
            if self.distributed and self.args.world_size>1:
                train_loader.sampler.set_epoch(epoch)
            losses = 0.
            correct, total = 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device,non_blocking=True), targets.to(self._device,non_blocking=True)
                logits = self._network(inputs)['logits']
                loss=F.cross_entropy(logits,targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if self.sched:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
            self._cur_task, epoch+1, self.init_epoch, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)
        logging.info(info)

    def _balance_update(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        self._network.eval()         # do not update the network normalizer statistics
        for epoch in prog_bar:
            if self.distributed and self.args.world_size>1:
                train_loader.sampler.set_epoch(epoch)
            losses, kd_losses, clf_losses, inter_losses = 0., 0., 0., 0.
            correct, total = 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device,non_blocking=True), targets.to(self._device,non_blocking=True)
                outputs = self._network(inputs)
                logits = outputs['logits']
                fea = outputs['fea']
                loss_clf = F.cross_entropy(logits[:, self._known_classes: ], targets - self._known_classes)

                outputs_prev = self._old_network(inputs)
                logits_prev = outputs_prev['logits']
                loss_kd = self._KD_loss(logits[:, : self._known_classes], logits_prev, self.T)

                one_hot_targets = F.one_hot(targets, self._total_classes).float()
                mask = torch.matmul(one_hot_targets, one_hot_targets.t())
                mask = mask - torch.eye(mask.size(0)).to(mask)

                fea_prev = outputs_prev['fea']
                disp = fea - fea_prev
                cdis = torch.exp(torch.matmul(fea_prev, fea_prev.t()) / self.sigma)   
                cdis = cdis * mask
                cdis = cdis / (torch.sum(cdis, dim=1, keepdim=True) + self.args.EPSILON)
                disp = torch.matmul(cdis, disp)
                fea_syn = F.normalize(fea_prev + disp, p=2, dim=1)
                loss_inter = torch.mean(1.0 - torch.sum(fea_syn*fea, dim=1))

                loss = loss_clf + loss_kd + loss_inter
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                kd_losses += loss_kd.item()
                clf_losses += loss_clf.item()
                inter_losses += loss_inter.item()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if self.sched:     
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, CLS_Loss {:.3f}, KD_Loss {:.3f}, Inter_Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                        self._cur_task, epoch+1, self.epochs, losses/len(train_loader), clf_losses/len(train_loader),
                        kd_losses/len(train_loader), inter_losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)
        logging.info(info)
                
    def _get_curr_data(self, data_manager):
        # only access the current task data        
        data_curr, targets_curr, idx_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                    mode='test', ret_data=True)
        idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size_ori, shuffle=False, num_workers=self.num_workers)

        # use idx to get data
        fea_list = []
        for idx, inputs, _ in idx_loader:
            inputs = inputs.to(self._device)
            fea = self._old_network(inputs)['fea']
            fea_list.append(fea)

        fea_all = torch.cat(fea_list, dim=0)
        return data_curr, targets_curr, fea_all

    def _anchor_train_multi(self, data_manager,known_classes, data_curr, targets_curr, fea_all):
        if self.args.calculate_distributed:
            logging.disable(logging.NOTSET)
        
        anchor_list = None
        sel_data_list = None
        cluster_idx_list = None
        for class_idx in known_classes:
            # logging.info("prompt train for class {}".format(class_idx))
            class_mean = self._class_means[class_idx*self.cluster_num:(class_idx+1)*self.cluster_num]
            class_mean = torch.from_numpy(class_mean).to(self._device)

            if self.args.scenario == 'domain':
                cluster_idx = []
                sel_data = None
                isel_data = data_curr[targets_curr % self.class_num == class_idx % self.class_num]
                for cluster in range(0, self.cluster_num):
                    for i in targets_curr:
                        if i % self.class_num == class_idx % self.class_num:
                            cluster_idx.append(cluster)
                    if sel_data is None:
                        sel_data = isel_data
                    else:
                        sel_data = np.concatenate([sel_data, isel_data], axis=0)
                cluster_idx = np.array(cluster_idx)
            else:
                cos_dis = torch.mm(class_mean, fea_all.t())
                sel_idx = torch.argsort(cos_dis, dim=-1, descending=True)[:, :self.sample_num].cpu().numpy().flatten()
                sel_data = data_curr[sel_idx]

                cluster_idx = []
                for i in range(self.cluster_num):
                    cluster_idx.extend([i] * self.sample_num)
                cluster_idx = np.array(cluster_idx)

            if sel_data_list is None:
                sel_data_list = sel_data
            else:
                sel_data_list = np.concatenate([sel_data_list, sel_data], axis=0)

            if cluster_idx_list is None:
                cluster_idx_list = cluster_idx + class_idx * self.cluster_num
            else:
                cluster_idx_list = np.concatenate([cluster_idx_list,
                                                cluster_idx + class_idx * self.cluster_num], axis=0)

            assert len(sel_data) == len(cluster_idx)

            sel_dset = data_manager.get_dataset([], source='train', mode='train',
                                    appendent=(sel_data, cluster_idx))
            train_loader = DataLoader(sel_dset, batch_size=self.batch_size_ori, shuffle=True,
                                        num_workers=self.num_workers,)
            prompt = torch.nn.Parameter(torch.rand((self.cluster_num, self.anchor_num,
                                                    self._old_network.feature_dim), device=self._device), requires_grad=True)
            optimizer = optim.Adam([prompt],lr=self.anchor_lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.anchor_epochs)
            prog_bar = tqdm(range(self.anchor_epochs))

            for _, epoch in enumerate(prog_bar):
                losses = 0.
                kd_losses = 0.
                clf_losses = 0.
                div_losses = 0.
                correct, total = 0, 0
                for _, inputs, cluster_i in train_loader:
                    inputs = inputs.to(self._device)
                    targets = torch.ones(len(inputs), dtype=torch.long).to(self._device) * class_idx           # transfer label to adaversary target class
                    sel_anchor = prompt[cluster_i]
                    outputs = self._old_network.forward_with_anchor(inputs, sel_anchor)   # forward with prev network
                    logits = outputs['logits']
                    fea = outputs['fea']

                    dis_mitrix = torch.mm(fea, fea.t())
                    dis_mitrix -= 0.95
                    dis_mitrix -= torch.diag(dis_mitrix.diag())
                    dis_mitrix = dis_mitrix[dis_mitrix > 0]

                    loss_div = torch.sum(dis_mitrix) / (len(dis_mitrix)+1e-8)

                    loss_clf = F.cross_entropy(logits, targets)
                    sel_mean = class_mean[cluster_i]
                    loss_kd = torch.mean(1.0 - torch.sum(fea * sel_mean, dim=-1))
                    loss = loss_kd + loss_clf + loss_div

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()
                    kd_losses += loss_kd.item()
                    clf_losses += loss_clf.item()
                    div_losses += loss_div.item()

                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

                scheduler.step()
                train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)

                info = 'Local Rank:{}--Anchor Class {}, Epoch {}/{} => Loss {:.3f}, CLS_Loss {:.3f}, KD_Loss {:.3f}, Div_Loss {:.3f}, Train_accy {:.2f}'.format(
                            self.local_rank, class_idx, epoch+1, self.anchor_epochs, losses/len(train_loader), clf_losses/len(train_loader),
                            kd_losses/len(train_loader), div_losses/len(train_loader) ,train_acc)
                prog_bar.set_description(info)
                logging.info(info)

            # logging.info(info)
    
            if anchor_list is None:
                anchor_list = prompt.detach().cpu()
            else:
                anchor_list = torch.cat([anchor_list, prompt.detach().cpu()], dim=0)

        del train_loader
            
        if self.args.calculate_distributed:
            if self.local_rank != 0:
                logging.disable(logging.CRITICAL)
        return sel_data_list, cluster_idx_list, anchor_list
    
    def update_previous_mean(self, data_manager, total_classes):
        with torch.no_grad():   
            mean_dis = []
            for i in total_classes:
                bool_idx = (self.cluster_idx == i)
                sel_data = self.sel_data[bool_idx]
                sel_idx = self.cluster_idx[bool_idx]
                sel_dset = data_manager.get_dataset([], source='train', mode='test',
                                                    appendent=(sel_data, sel_idx))
                train_loader_test = DataLoader(sel_dset, batch_size=self.batch_size_ori, shuffle=False,
                                                num_workers = self.num_workers, persistent_workers=True)

                fea_old, _ = self.extract_all_fea_with_anchor(train_loader_test,
                                                                self._old_network)  # feature of the previous task
                fea_new, _ = self.extract_all_fea_with_anchor(train_loader_test, self._network)
                distance = self.displacement(fea_new, fea_old, self._class_means[i][None, :], self.sigma)
                mean_dis.append(distance)
            del train_loader_test
            mean_dis = np.concatenate(mean_dis, axis=0).astype(np.float32)
            return mean_dis
        
    def construct_curr_mean(self, data_manager, total_classes):
        with torch.no_grad():   
            cur_mean = []
            for class_idx in total_classes:
                sel_dset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train', mode='test')
                train_loader_test = DataLoader(sel_dset, batch_size=self.batch_size_ori, shuffle=False,
                                                num_workers = self.num_workers, persistent_workers=True)
                fea_new, label_new = self.extract_all_fea(train_loader_test, self._network) # feature of the current task
                k_means = KMeans(n_clusters=self.cluster_num, random_state=0, n_init='auto').fit(fea_new)
                cur_mean.append(k_means.cluster_centers_)
            cur_mean = np.concatenate(cur_mean, axis=0).astype(np.float32)
            del train_loader_test
            return cur_mean
    
    def construct_mean_distributed(self,data_manager):
        if self.args.calculate_distributed:
            """update class means for each class in the previous task."""
            if self._class_means is not None and  self._old_network is not None:
                logging.info("previous means are updated on {} gpus".format(self.world_size))
                total_classes = list(range(len(self._class_means)))
                splits_total_classes = np.array_split(total_classes, self.world_size)
                total_classes = list(splits_total_classes[self.local_rank])

                total_mean_dis = [None for _ in range(self.world_size)]
                mean_dis = self.update_previous_mean(data_manager, total_classes)
                dist.all_gather_object(total_mean_dis, mean_dis)

                total_mean_dis = np.concatenate([idata for idata in total_mean_dis],0)
                self._class_means += total_mean_dis
                logging.info('finish update mean')
            
            """Construct class means for each class in the current task."""
            total_classes = list(range(self._known_classes, self._total_classes))
            splits_total_classes = np.array_split(total_classes, self.world_size)
            total_classes = list(splits_total_classes[self.local_rank])

            total_cur_mean = [None for _ in range(self.world_size)]
            cur_mean = self.construct_curr_mean(data_manager, total_classes)
            dist.all_gather_object(total_cur_mean, cur_mean)
            total_cur_mean = np.concatenate([idata for idata in total_cur_mean],0)

            if self._class_means is None:
                self._class_means = total_cur_mean
            else:
                self._class_means = np.concatenate([self._class_means, total_cur_mean], axis=0)
        else:
            if self._class_means is not None and  self._old_network is not None:
                total_classes = list(range(len(self._class_means)))
                mean_dis = self.update_previous_mean(data_manager, total_classes)
                self._class_means += mean_dis
                logging.info('finish update mean')
            
            total_classes = list(range(self._known_classes, self._total_classes))
            cur_mean = self.construct_curr_mean(data_manager, total_classes)

            if self._class_means is None:
                self._class_means = cur_mean
            else:
                self._class_means = np.concatenate([self._class_means, cur_mean], axis=0)

        self._class_means = self._class_means / (np.linalg.norm(self._class_means , axis=1, keepdims=True) + self.args.EPSILON)
        logging.info("finish construct mean, shape is {}".format(self._class_means.shape))

    def anchor_train_distributed(self,data_manager):
        if self._old_network is not None:
            if not self.args.calculate_distributed :
                self._old_network.to(self._device)
                data_curr, targets_curr, fea_all = self._get_curr_data(data_manager)
                self.sel_data, self.cluster_idx, self.prompt = self._anchor_train_multi(data_manager, np.arange(self._known_classes), data_curr, targets_curr, fea_all)
            else:
                logging.info("prompt train is distributed on {} gpus".format(self.world_size))
                self._old_network.to(self._device)
                data_curr, targets_curr, fea_all = self._get_curr_data(data_manager)
    
                known_classes = list(range(self._known_classes))
                splits_known_classes = np.array_split(known_classes, self.world_size)
                known_classes = list(splits_known_classes[self.local_rank])
                
                # prepare for distributed               
                anchor_list = [torch.zeros(
                    (self.cluster_num*len(classes), self.anchor_num,self._old_network.feature_dim), dtype=torch.float32).to(self._device) 
                            for _,classes in enumerate(splits_known_classes)]
                sel_data_list = [None for _ in range(self.world_size)]
                cluster_idx_list = [None for _ in range(self.world_size)]

                # anchor_train
                sel_data, cluster_idx, prompt = self._anchor_train_multi(data_manager, known_classes, data_curr, targets_curr, fea_all)
                del data_curr, targets_curr, fea_all
                torch.distributed.barrier()

                all_anchor = self.all_gather_nd(prompt.to(self._device),self.world_size)
                all_anchor = torch.cat(all_anchor)
                dist.all_gather_object(sel_data_list,sel_data)
                dist.all_gather_object(cluster_idx_list,cluster_idx)
                
                self.prompt = torch.cat([ianchor.detach().cpu() for ianchor in anchor_list],0)
                self.sel_data = np.concatenate([idata for idata in sel_data_list],0)
                self.cluster_idx = np.concatenate([idata for idata in cluster_idx_list],0)

    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred/T, dim=1)
        soft = torch.softmax(soft/T, dim=1)
        return -1 * torch.mul(soft, pred).sum()/pred.shape[0]

    def displacement(self, new_fea, old_fea, old_mean, sigma):
        disp = new_fea-old_fea
        cos_dis = np.matmul(old_mean, old_fea.T)
        cos_dis = np.clip(cos_dis, -1, 1)
        W = np.exp(cos_dis/sigma)
        normalize = np.sum(W, axis=1, keepdims=True)
        W = W / (normalize + self.args.EPSILON)
        mean_disp = np.matmul(W, disp)
        return mean_disp
                    
    def all_gather_nd(self,tensor,world_size):
        """
        Ref:https://stackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of
        Gathers tensor arrays of different lengths in a list.
        The length dimension is 0. This supports any number of extra dimensions in the tensors.
        All the other dimensions should be equal between the tensors.
        """
        world_size = dist.get_world_size()
        local_size = torch.tensor(tensor.size(), device=tensor.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)

        max_length = max(size[0] for size in all_sizes)

        length_diff = max_length.item() - local_size[0].item()
        if length_diff:
            pad_size = (length_diff, *tensor.size()[1:])
            padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat((tensor, padding))

        all_tensors_padded = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(all_tensors_padded, tensor)
        all_tensors = []
        for tensor_, size in zip(all_tensors_padded, all_sizes):
            all_tensors.append(tensor_[:size[0]])
        return all_tensors
  
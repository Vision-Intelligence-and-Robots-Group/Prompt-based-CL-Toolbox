from torch import optim
import timm
import logging
from methods.base import BaseLearner
from torch.nn.parallel import DistributedDataParallel as DDP
from models.dualp import _create_vision_transformer
import torch
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
from utils.toolkit import tensor2numpy

from timm.optim import create_optimizer

class dualp_engine(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        prompt_kwargs = dict(
            prompt_length=args.prompt_length,
            embedding_key=args.embedding_key,
            prompt_init=args.prompt_key_init,
            prompt_pool=args.prompt_pool,
            prompt_key=args.prompt_key,
            pool_size=args.size,
            top_k=args.top_k,
            batchwise_prompt=args.batchwise_prompt,
            prompt_key_init=args.prompt_key_init,
            head_type=args.head_type,
            use_prompt_mask=args.use_prompt_mask,
            use_g_prompt=args.use_g_prompt,
            g_prompt_length=args.g_prompt_length,
            g_prompt_layer_idx=args.g_prompt_layer_idx,
            use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
            use_e_prompt=args.use_e_prompt,
            e_prompt_layer_idx=args.e_prompt_layer_idx,
            use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
            same_key_value=args.same_key_value,)


        model_kwargs = dict(num_classes=100, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                            drop_rate=0., drop_path_rate=0., )
        self._network_ori = _create_vision_transformer('vit_base_patch16_224_base', pretrained=args.pretrained, **model_kwargs)

        model_kwargs = dict(num_classes=100, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                            drop_rate=0., drop_path_rate=0., **prompt_kwargs)
        self._network = _create_vision_transformer('vit_base_patch16_224_base', pretrained=args.pretrained, **model_kwargs)

        self.args = args
        self.EPSILON = args.EPSILON
        
        self.num_workers = args.num_workers
        self.init_epoch = args.init_epoch
        self.init_weight_decay = args.init_weight_decay
        self.epochs = args.epochs
        self.weight_decay = args.weight_decay

        self.batch_size = args.batch_size
        self.batch_size_ori = args.batch_size
        self.init_lr = args.init_lr
        self.lrate = args.lrate

        if self.args.unscale_lr:
            global_batch_size = args.batch_size
        else:
            global_batch_size = args.batch_size * args.world_size
        self.init_lr = args.init_lr * global_batch_size / 256.0
        self.lrate = args.lrate * global_batch_size / 256.0

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, train_loader, test_loader, data_manager, task_id):
        self._cur_task = task_id
        self._total_classes = self._known_classes + data_manager.get_task_size(task_id)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        self._network.to(self._device)
        self._network_ori.to(self._device)

        if self.args.freeze:
            # all parameters are frozen for original vit model
            for p in self._network_ori.parameters():
                p.requires_grad = False
            
            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in self._network.named_parameters():
                if n.startswith(tuple(self.args.freeze)):
                    p.requires_grad = False

        # settings for DDP
        if self.distributed:
            self._network = DDP(self._network, device_ids=[self.local_rank])
        
        # training
        self._train(train_loader, test_loader)

        if self.distributed:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):

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
        if self._cur_task == 0:
            optimizer = optim.Adam(network_params, lr=self.init_lr, eps=1e-8, betas=(0.9, 0.999)) 
            self.run_epoch = self.init_epoch
        else:
            optimizer = optim.Adam(network_params, lr=self.lrate, eps=1e-8, betas=(0.9, 0.999))
            self.run_epoch = self.epochs
        self.train_function(train_loader, test_loader, optimizer)
    
    def train_function(self, train_loader, test_loader, optimizer):
        # Transfer previous learned prompt params to the new prompt
        if self.args.prompt_pool and self.args.shared_prompt_pool:
            if self._cur_task > 0:
                prev_start = (self._cur_task - 1) * self.args.top_k
                prev_end = self._cur_task * self.args.top_k

                cur_start = prev_end
                cur_end = (self._cur_task + 1) * self.args.top_k

                if (prev_end > self.args.size) or (cur_end > self.args.size):
                    pass
                else:
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if self.args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if self.args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if self.args.distributed:
                            self._network.module.e_prompt.prompt.grad.zero_()
                            self._network.module.e_prompt.prompt[cur_idx] = self._network.module.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = self._network.module.parameters()
                        else:
                            self._network.e_prompt.prompt.grad.zero_()
                            self._network.e_prompt.prompt[cur_idx] = self._network.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = self._network.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if self.args.prompt_pool and self.args.shared_prompt_key:
            if self._cur_task > 0:
                prev_start = (self._cur_task - 1) * self.args.top_k
                prev_end = self._cur_task * self.args.top_k

                cur_start = prev_end
                cur_end = (self._cur_task + 1) * self.args.top_k

                with torch.no_grad():
                    if self.args.distributed:
                        self._network.module.e_prompt.prompt_key.grad.zero_()
                        self._network.module.e_prompt.prompt_key[cur_idx] = self._network.module.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = self._network.module.parameters()
                    else:
                        self._network.e_prompt.prompt_key.grad.zero_()
                        self._network.e_prompt.prompt_key[cur_idx] = self._network.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = self._network.parameters()
   
        prog_bar = tqdm(range(self.run_epoch))
        self._network.train(True)
        self._network_ori.eval()
        for _, epoch in enumerate(prog_bar):
            if self.distributed and self.args.world_size > 1:
                train_loader.sampler.set_epoch(epoch)
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)

                with torch.no_grad():
                    if self._network_ori is not None:
                        output = self._network_ori(inputs)
                        cls_features = output['pre_logits']
                    else:
                        cls_features = None

                output = self._network(inputs, task_id=self._cur_task, cls_features=cls_features, train=True)
                logits = output['logits']

                mask = range(self._known_classes, self._total_classes)
                not_mask = np.setdiff1d(np.arange(logits.shape[1]), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self._device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                loss = F.cross_entropy(logits, targets)

                if self.args.pull_constraint and 'reduce_sim' in output:
                    loss = loss - self.args.pull_constraint_coeff * output['reduce_sim']

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), self.args.clip_grad)
                optimizer.step()
                losses += loss.item()
                
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)
            
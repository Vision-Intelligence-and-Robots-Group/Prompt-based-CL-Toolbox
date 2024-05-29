import logging
import numpy as np
import torch
from torch import nn
from utils.toolkit import tensor2numpy, accuracy,accuracy_domain
from scipy.spatial.distance import cdist

class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = 0
        self._known_classes = 0
        self._total_classes = 0
        self.increment = args.increment
        self._network = None
        self._old_network = None
        self.topk = 1

        self.num_tasks = args.num_tasks
        self.acc_table_cnn = np.zeros([self.num_tasks, self.num_tasks])
        self.acc_table_nme = np.zeros([self.num_tasks, self.num_tasks])

        self.result_curve = dict()
            
        self._device = torch.device(args.device)
        self.world_size = args.world_size

        self.distributed = args.distributed
        if self.distributed and self.world_size > 1:
            self.calculate_distributed = args.calculate_distributed
        else:
            self.calculate_distributed = False

    def incremental_train(self):
        pass
        
    def after_task(self):
        pass

    def eval_task(self, test_loader):
        y_pred, y_true = self._eval_cnn(test_loader)
        cnn_accy = self._evaluate(y_pred, y_true, True)
        self._logging_result(cnn_accy, 'CNN')
        
        if hasattr(self, '_class_means'):
            y_pred, y_true = self._eval_nme_multiple(test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true, False)
            self._logging_result(nme_accy, 'NME')     
        print('\n')
    
    def _logging_result(self, result, prefix):
        if self.result_curve.get(prefix) == None:
            self.result_curve[prefix] = {'top1': []}

        logging.info('{}: {}'.format(prefix, result['grouped']))
        self.result_curve[prefix]['top1'].append(result['top1'])
        logging.info('{} top1 curve: {}'.format(prefix, self.result_curve[prefix]['top1']))

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            with torch.no_grad():
                if self.args.extra_eval == 'S-Prompt':
                    feature = self._network.extract_vector(inputs)
                    taskselection = []
                    for task_centers in self.all_keys:
                        tmpcentersbatch = []
                        for center in task_centers:
                            tmpcentersbatch.append((((feature - center) ** 2) ** 0.5).sum(1))
                        taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0])

                    selection = torch.vstack(taskselection).min(0)[1]
                    outputs = self._network.interface(inputs, selection)
                elif self.args.extra_eval in ['L2P','DualP']:
                    if self._network_ori is not None:
                        self._network_ori.eval()
                        output = self._network_ori(inputs)
                        cls_features = output['pre_logits']
                    else:
                        cls_features = None
                    outputs = self._network(inputs, task_id=self._cur_task, cls_features=cls_features)['logits'] #[bs,n]
                else :
                    outputs = self._network(inputs)['logits'] #[bs,n]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader, self._network)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + self.args.EPSILON)).T

        dists = cdist(class_means, vectors, 'sqeuclidean')  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, :self.topk], y_true  # [N, topk]

    def _eval_nme_multiple(self, loader, class_means):
        cluster_num = self.cluster_num
        vectors, y_true = self._extract_vectors(loader, self._network)

        cos_dis = np.matmul(vectors, class_means.T)
        cos_dis = np.clip(cos_dis, -1, 1)
        cos_dis = np.exp(cos_dis/self.sigma)
        cos_dis = cos_dis / np.sum(cos_dis, axis=1, keepdims=True)
        cos_dis = cos_dis.reshape(len(vectors), -1, cluster_num)
        cos_dis = np.sum(cos_dis, axis=-1)
        return np.argsort(cos_dis, axis=1)[:, ::-1][:, :self.topk], y_true  # [N, topk]
    
    def _evaluate(self, y_pred, y_true , is_cnn):
        ret = {}
        if self.args.scenario == 'domain':
            grouped = accuracy_domain(y_pred.T[0], y_true, self._known_classes, class_num=self.class_num)
        else:
            if is_cnn:
                grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self.increment, self.acc_table_cnn, self._cur_task)            
            else:
                grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self.increment, self.acc_table_nme, self._cur_task)

        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _compute_accuracy(self, model, loader):
        self._network.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                if self.args.extra_eval == 'S-Prompt':
                    feature = self._network.extract_vector(inputs)
                    taskselection = []
                    for task_centers in self.all_keys:
                        tmpcentersbatch = []
                        for center in task_centers:
                            tmpcentersbatch.append((((feature - center) ** 2) ** 0.5).sum(1))
                        taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0])
                    selection = torch.vstack(taskselection).min(0)[1]
                    outputs = self._network.interface(inputs, selection)
                elif self.args.extra_eval in ['L2P','DualP']:
                    if self._network_ori is not None:
                        self._network_ori.eval()
                        output = self._network_ori(inputs)
                        cls_features = output['pre_logits']
                    else:
                        cls_features = None
                    outputs = self._network(inputs, task_id=self._cur_task, cls_features=cls_features)['logits'] #[bs,n]
                else :
                    outputs = self._network(inputs)['logits'] #[bs,n]

            predicts = torch.max(outputs, dim=1)[1]
            if self.args.scenario == 'domain':
                correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            else:
                correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct)*100 / total, decimals=4)
    
    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _extract_vectors(self, loader, model):
        model.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(model, nn.DataParallel):
                _vectors = tensor2numpy(model.module.extract_vector(_inputs.to(self._device)))
            else:
                _vectors = tensor2numpy(model.extract_vector(_inputs.to(self._device)))

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
    
    def save_checkpoint(self, model, filename):
        model.cpu()
        save_dict = {
            'tasks': self._cur_task,
            'model_state_dict': model.state_dict(),
        }
        torch.save(save_dict, '{}_{}.pkl'.format(filename, self._cur_task))

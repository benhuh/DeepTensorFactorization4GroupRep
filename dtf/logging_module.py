import matplotlib
import numpy as np
import torch
# from dtf.tensor_operations import compute_effective_rank
from dtf.model import Deep_Tensor_Net


PROG_BAR_LIST = ["loss/train", "loss/val"] 

class Logging_Module():

    def aggregate_info(self, info_dicts, key, pop=False):
        if pop:
            info_list =  [x.pop(key) for x in info_dicts]
        else:
            info_list =  [x[key] for x in info_dicts]
        # else:
        return info_list

    def aggregate_loss_acc(self, info_dicts):
        total_batch = sum(self.aggregate_info(info_dicts,"batch", pop=True))
        loss = torch.stack(self.aggregate_info(info_dicts,"loss/reconst", pop=True)).sum() / total_batch
        accuracy = torch.stack(self.aggregate_info(info_dicts,"accuracy", pop=True)).sum() / total_batch if "accuracy" in info_dicts[0].keys() else None
        return loss, accuracy        

    def log_epoch_end(self, info_dicts, train_or_test):

        with torch.no_grad():
            loss_reconst, accuracy = self.aggregate_loss_acc(info_dicts)

        logs = { f"loss/{train_or_test}": loss_reconst,  }

        if accuracy is not None:
            logs[f"accuracy/{train_or_test}"] = accuracy

        for key in info_dicts[0].keys():
            logs[key] = sum(self.aggregate_info(info_dicts, key)) #.mean()

        if train_or_test == 'train':
            logs['scheduler/counter'] = self.scheduler_counter

        elif train_or_test == 'val':
            
            if isinstance(self.model,(Deep_Tensor_Net)):
                logs.update(self.log_weight_norm())
                logs.update({'orth_loss/all': compute_AA(self.model.factor_list, factor_norm=logs["norm/mean"])[1]**2})
                logs.update({'orth_loss/indiv': compute_individual_AA(self.model.factor_list, factor_norm=logs["norm/mean"])**2})

                if self.hparams.log_svd_max>0:
                    logs.update(self.log_svd_factors(self.hparams.log_svd_max)) #, weight_norm=logs["norm/mean"]))

                if self.hparams.log_imbalance:

                    if 'imbalance2/mean' in self.hparams.scheduler_criterion:
                        logs.update(self.log_imbalance2(weight_norm=logs["norm/mean"])) #self.hparams.log_svd_max, svd=True))

        for k, v in logs.items():
            self.log(k, v, prog_bar=True if k in PROG_BAR_LIST + self.hparams.scheduler_criterion else False)
            
        ## log values before_annealing weight decay        # skip logging singular values
        self.last_logs.update({k:v for k, v in logs.items() if not k.startswith('sing')}) #(logs)
            
        if self.scheduler_counter == 0:
            for key in ['accuracy/train', 'accuracy/val', 'loss/total', 'loss/train', 'loss/val', 'grad_norm/mean', 'imbalance/mean', 'imbalance2/mean', 'erank/mean', 'orth_loss/all', 'orth_loss/indiv']:
                self.last_logs[key+'/before_anneal'] = self.last_logs.get(key,0.0)
            self.last_logs['time'+'/before_anneal'] = self.current_epoch - self.hparams.counter_threshold[0]
        
        self.last_logs['_accuracy/val'] = 100 - self.last_logs.get('accuracy/val',0.0)
        return logs
    
    def log_svd_factors(self, max_num=10, weight_norm=None): 
        with torch.no_grad():
            logs={}
            svd_dict = self.model.get_svd()

            if len(svd_dict)>0:
                for idx, err in svd_dict.items():
                    for j, s in enumerate(err['sig']):
                        if j<max_num: 
                            logs[f"singular_val/{idx}/{j}"] = s.item()
                            
            return logs 
        

    def log_imbalance2(self, max_num=10, svd=True, weight_norm=None): 
        with torch.no_grad():
            logs={}
            imbalance2 = self.model.get_imbalance2()

            if len(imbalance2)>0:
                norm_2_sum = 0
                numel_sum = 0

                for idx, err in imbalance2.items():
                    norm = err.norm().item() 
                    numel = err.numel()
                    norm_2_sum += norm**2
                    numel_sum += numel

                logs[f"imbalance2/mean"]  = (norm_2_sum/numel_sum)**0.5

                if weight_norm is not None:
                    logs[f"imbalance2/mean"] /= weight_norm ** 4

            return logs 
    


    def record_param_grad(self):
        if getattr(self,'netW_hist',None) is None:
            self.netW_hist = []
        self.netW_hist.append(self.model._net_Weight.detach()+0.0)

        if getattr(self,'w_hist',None) is None:
            self.w_hist = []
        w = [p.detach() for p in self.model.factor_list]
        w_shape = [p.shape for p in w]
        if check_all_equal(w_shape):
            w = torch.stack(w)
        self.w_hist.append(w)


    @staticmethod
    def get_layer_name(name):
        split = name.split('.')
        layer_num = split[-2] if split[-1]=='weight' else split[-1]
        name_ =   layer_num
        # name_ = 'layer' + layer_num
        return name_, layer_num
    
    def log_weight_norm(self): 
        logs = {}
        norm_2_sum = 0
        numel_sum = 0
        for name, param in self.named_parameters():
            name_, layer_num = self.get_layer_name(name)
            # get the l2 norm of the parameter
            # norm = param.pow(2).mean().sqrt().item()  #torch.norm(param,2).item() / np.sqrt(param.numel())
            norm = torch.norm(param,2).item() 
            numel = param.numel()
            norm_2_sum += norm**2
            numel_sum += numel

        logs["norm/mean"] = (norm_2_sum/numel_sum)**0.5
        logs["norm/NetW"] = self.model._net_Weight.norm().item()  / np.sqrt(self.model._net_Weight.numel())
        return logs  
        
def check_all_equal(list):
    return all(i == list[0] for i in list)
     

    
def compute_AA(factor_list_opt, factor_norm, imshow=False):
    AA_all = []
    for A in factor_list_opt:
        AAt = []
        AtA = []
        for A_ in A:
            AAt.append(A_@A_.T)
            AtA.append(A_.T@A_)
        AA_all.append(torch.stack(AAt).sum(0))
        AA_all.append(torch.stack(AtA).sum(0))
    AA_all = torch.stack(AA_all)

    diag_mean = AA_all.diagonal(dim1=1,dim2=2).mean()
    orth_loss = AA_all - diag_mean*torch.eye(AA_all[0].shape[0], device = AA_all.device)
    orth_loss_norm = orth_loss.norm() / AA_all.numel()**0.5

    return AA_all, orth_loss_norm / factor_norm **2


def compute_individual_AA(factor_list_opt, factor_norm):
    orth_loss_all = []
    for A in factor_list_opt:
        for A_ in A:
            orth_loss = orthogonal_loss(A_)
            orth_loss_all.append(orth_loss) # normalize by diag_mean
    
    orth_loss_indiv_norm = torch.stack(orth_loss_all).norm() / sum([A.numel() for A in orth_loss_all])**0.5

    return orth_loss_indiv_norm / factor_norm **2

def orthogonal_loss(A):
    AA = A@A.T
    diag_mean = AA.diag().mean()
    orth_loss = AA - diag_mean*torch.eye(AA.shape[0], device = AA.device)
    # orth_loss = AA/diag_mean - torch.eye(AA.shape[0])
    return orth_loss

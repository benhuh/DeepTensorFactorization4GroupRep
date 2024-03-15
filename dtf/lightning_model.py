# import time
from typing import Any, Dict, List, Optional, Tuple, Union
from argparse import Namespace #, ArgumentParser
from pytorch_lightning import LightningModule
import torch
import numpy as np

from dtf.model import get_model_parser, Deep_Tensor_Net, Deep_Tensor_Net_conv #Transformer, Simple_MLP, Simple_MLP2, Simple_CNN, Simple_Resnet
from dtf.logging_module import Logging_Module
from dtf.lr_scheduler import  Reduce_WeightDecayCoeff_OnPlateau
from torch.optim import SGD

torch_inf = torch.tensor(np.Inf)

class LITmodel(LightningModule, Logging_Module):
    """
    Adds training methods to train a generic transformer on arithmetic equations
    """

    def __init__(self, hparams: Namespace): #, device) -> None:
        """
        :param hparams: An argparse.Namespace with parameters defined in
                        self.add_model_specific_args().
        """
        super().__init__()
        self.save_hyperparameters(hparams)

        if isinstance(hparams,dict):
            hparams = Namespace(**hparams)
        hparams.model = 'Deep_Tensor_Net_conv'
        if hparams.model == 'Transformer':            # Make sure d_model, n_heads, and d_key are compatible   # hparams.d_key = hparams.d_model / hparams.n_heads
            assert ( hparams.d_model % hparams.n_heads == 0 ), "n_heads=%s does not evenly divide d_model=%s" % ( hparams.n_heads,  hparams.d_model,)
            model_class = Transformer
            model_args = dict(  dim=hparams.d_model,
                                num_layers=hparams.n_layers,
                                num_heads=hparams.n_heads,
                                seq_len=hparams.max_context_len,
                                num_tokens=hparams.tensor_width,) #len(arithmetic_tokenizer), )

        elif hparams.model in  ['Deep_Tensor_Net', 'Deep_Tensor_Net_conv']:
            model_class = Deep_Tensor_Net if hparams.model == 'Deep_Tensor_Net' else Deep_Tensor_Net_conv
            model_args = dict(  N=hparams.tensor_width,
                                r=hparams.model_rank,
                                decomposition_type=hparams.decomposition_type,
                                init_scale=hparams.init_scale,
                                layer_type = hparams.layer_type,
                                )
        else:
            raise ValueError

        model_args.update(loss_fn=hparams.loss_fn,)
                        #   weight_decay_form = getattr(hparams,'weight_decay_form','default')  )
        self.model = model_class(**model_args)

        self.next_epoch_to_eval = -1
        self.next_train_epoch_to_log = 0

        self.last_logs = {criterion: torch_inf for criterion in self.hparams.scheduler_criterion}
        self.A_hist = None

        self.validation_step_outputs = []
        self.train_step_outputs = []

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    @property
    def current_epoch(self) -> int:
        return int(super().current_epoch/2)  # Hacky fix for factor of 2 in epoch ...

    @property
    def current_step(self) -> int:
        return int(super().global_step)

    @property
    def current_wd_coeff(self):
        return self.trainer.optimizers[0].param_groups[0].get('wd_coeff', None)

    @property
    def current_weight_decay(self):
        return self.current_wd_coeff * self.hparams.weight_decay

    @property
    def current_lr(self):
        return self.trainer.optimizers[0].param_groups[0]['lr']

    @property
    def scheduler_counter(self):
        default_count = -1
        schedulers = self.lr_schedulers()
        scheduler = schedulers[0] if isinstance(schedulers, (list,tuple)) else schedulers
        return scheduler.scheduler_counter if hasattr(scheduler,'scheduler_counter') else default_count

    @property
    def scheduler_criterion(self):
        if self.hparams.scheduler_criterion is not None: # and len(self.hparams.scheduler_criterion)>0:
            assert len(self.hparams.scheduler_criterion) in [1,2]
            return self.last_logs[self.hparams.scheduler_criterion[0]], self.last_logs[self.hparams.scheduler_criterion[-1]]
        else:
            return 0.0, 0.0


    def configure_optimizers(self) -> Tuple[List[Any], List[Dict]]:
        lr = self.hparams.lr
        WD_scheduler = Reduce_WeightDecayCoeff_OnPlateau

        if not (self.optimizers() and self.trainer.lr_scheduler_configs):  # define new optimizers if not already set..  (if lr_scheduler_configs is None)

            param_group_1 = {'params': self.model.T_param_list}
            param_group_2 = {'params': self.model.conv_weight, 'lr': 1 * lr, 'weight_decay': 0.1}
            param_groups = [param_group_1, param_group_2]

            if self.hparams.optim == 'SGD':
                optimizer = SGD(param_groups, momentum=self.hparams.betas[0], lr=lr, weight_decay=0)
            else:
                raise ValueError

            optimizers = [optimizer]
            schedulers = [
                {"scheduler": WD_scheduler(optimizer, mode='min', factor= self.hparams.scheduler_decay, trigger_threshold = self.hparams.scheduler_threshold, counter_threshold = self.hparams.counter_threshold[0], min_lr=self.hparams.weight_decay_min, eps=0),
                    'monitor': 'scheduler/criterion wd',     "interval": "step", "frequency": 1,  'name': 'monitor',},                      ]
        else:
            optimizers = self.trainer.optimizers
            schedulers = [  config.__dict__  for config in self.trainer.lr_scheduler_configs]
        return optimizers, schedulers


    def on_after_backward(self, *args):

        logs={}
        for k, v in logs.items():
            self.log(k, v)

        self.last_logs.update(logs)

        if self.hparams.record_wg_hist>0 :
            record_condition = not (self.current_step % self.hparams.record_wg_hist) #50)
            if record_condition:
                self.record_param_grad()


    def _step( self, data, batch_idx: int, train_or_test: str = 'train'):
        """
        Performs one forward pass on a training or validation batch

        :param data: The batch of data (equations) to process
        :param batch_idx: which batch this is in the epoch.
        """

        if self.hparams.model == 'Transformer':
            x, = data
            batch = len(x)
            data = x
            # import pdb; pdb.set_trace()
        else:
            x, y = data
            batch = x.shape[0]

        other_losses = {}
        loss_reconst, acc, out, other_outputs = self.model.evaluate(data, train_or_test=train_or_test)

        if getattr(self.hparams, 'manual_L2', False):
            other_losses['regularization'] = self.model.manual_L2_loss()

        if getattr(self.hparams, 'custom_L2', False):
            other_losses['regularization'] = self.model.Custom_L2_loss()

        info = {"loss/reconst":  loss_reconst.detach(),
                "batch": batch,
                }
        if acc is not None:
            info[ "accuracy"] = acc.detach()*batch

        return loss_reconst, info, other_losses


    def training_step(self, data, batch_idx):
        """
        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        """
        loss_reconst, info, other_losses = self._step(data=data, batch_idx=batch_idx, train_or_test='train')


        if 'regularization' in other_losses.keys():
            reg_loss = self.current_weight_decay * other_losses['regularization']
            loss_train = loss_reconst + reg_loss
            info["loss/reg"] = reg_loss.item() / self.model._net_Weight.numel()
        else:
            loss_train = loss_reconst

        info["loss"] = loss_train

        self.log('scheduler/criterion wd', self.scheduler_criterion[0], prog_bar=False) #True)
        self.log('scheduler/criterion lr', self.scheduler_criterion[-1], prog_bar=False) #True)

        self.train_step_outputs.append(info)

        return info

    def on_train_epoch_end(self):
        """
        Used by pytorch_lightning: Accumulates results of all forward training passes in this epoch
        """
        info_dicts = self.train_step_outputs
        if len(info_dicts) != 0:
            self.log_epoch_end(info_dicts, train_or_test='train')
        self.train_step_outputs.clear()

    def validation_step(self, data, batch_idx):
        with torch.no_grad():
            loss_main, info, other_losses = self._step(data=data, batch_idx=batch_idx, train_or_test='val')

        self.validation_step_outputs.append(info)
        return info


    def on_validation_epoch_end(self):
        info_dicts = self.validation_step_outputs
        # if len(info_dicts) == 0:
        #     import pdb; pdb.set_trace()

        validation_is_real = len(info_dicts[0]) != 0 if len(info_dicts)>0 else False

        if validation_is_real:
            logs = self.log_epoch_end(info_dicts, train_or_test='val')
            self.validation_step_outputs.clear()
            return logs


    @staticmethod
    def get_model_parser(model): # -> ArgumentParser:
        return get_model_parser(model)

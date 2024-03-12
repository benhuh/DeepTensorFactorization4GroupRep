from torch import inf, is_tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLROnPlateau_orig
from torch.optim.lr_scheduler import EPOCH_DEPRECATION_WARNING

class ReduceLROnPlateau(ReduceLROnPlateau_orig):
    key = 'lr'  # For lr_scheduling

    def __init__(self, *args, trigger_threshold = None, counter_threshold = 0, **kwargs):
        self.trigger_threshold = trigger_threshold
        self.counter_threshold = counter_threshold
            
        super().__init__(*args, **kwargs)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

        self.triggered = 0 #False
        self.counter = 0
        self.scheduler_counter = 0

    def step(self, metrics, epoch=None):
        if not is_tensor(metrics):  # Huh: Bug fix (avoids the small (non-tensor) metrics given during validation.. for some reason)
            metrics = inf if self.mode == 'min' else -inf

        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.trigger_threshold is not None:
            if self.is_better(current, self.trigger_threshold): # Huh
                self.triggered += 1   # Huh
        else:
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown
                
        if self.triggered>self.counter_threshold or self.num_bad_epochs > self.patience:        # if self.triggered or self.num_bad_epochs > self.patience:

            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            self.scheduler_counter += 1
            self.triggered += 1

        if self.key == 'lr':
            self._last_lr = [group[self.key] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group[self.key])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            param_group[self.key] = new_lr


class Reduce_WeightDecayCoeff_OnPlateau(ReduceLROnPlateau):
    key = 'wd_coeff'    # For manual_L2 weight_decay_scheduling 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group[self.key]=1

from dtf.training import get_hparams, get_model_pkg, get_datamodule
from argparse import Namespace
from dtf.save_load import plot_scalars

default_kwargs = dict()

def get_args_str(args):
    args_str = f' --model {args.model}'
    if args.gpus is not None:
        args_str += f' --gpus {args.gpus}'

    args_str += f' --optim {args.optim} --random_seed {args.random_seed} --weight_decay_min {args.weight_decay_min}' #' --constant_lr' 
    args_str += f' --exp_name {args.exp_name} --task_name {args.task_name} --tensor_width {args.tensor_width} --task_rank {args.task_rank} --model_rank {args.model_rank} --loss_fn {args.loss_fn} '
    args_str += f' --train_frac {args.train_frac} --init_scale {args.init_scale} --weight_decay {args.weight_decay} --lr {args.lr} --betas {args.momentum} 0'        
    args_str += f' --record_wg_hist {args.record_wg_hist}'

    scheduler_criterion =  'imbalance2/mean'
    args_str += f' --earlystop {args.earlystop} --scheduler_criterion {scheduler_criterion} --scheduler_threshold {args.scheduler_threshold} --counter_threshold {args.counter_threshold} --scheduler_decay {args.scheduler_decay}' #  --scheduler_decay 0.8
    args_str += f' --log_imbalance --log_svd_max {args.log_svd_max} --val_check_interval {args.val_check_interval}'
    return args_str

def get_args(**kwargs):
    default_args = dict(model = 'Deep_Tensor_Net',
                        gpus = 0,
                        exp_name = 'test',
                        task_name = 'binary/sym3_xy',
                        loss_fn = 'mse_loss',  # 'Lagrange', 'cross_entropy'
                        optim = 'SGD',
                        train_frac = 100,
                        tensor_width = 0,
                        task_rank = 0,
                        model_rank = 0, #40
                        init_scale = 1 ,
                        weight_decay = 0.1, #/5 #*4
                        weight_decay_min = 0, 
                        lr = 1/2,
                        momentum = 0.9,
                        earlystop = 'loss/train',
                        scheduler_criterion = None, 
                        scheduler_threshold = 1e-5, #3e-5 #10
                        scheduler_decay = 0.0, 
                        counter_threshold = '0 50',
                        record_wg_hist = 0,
                        val_check_interval = 10,
                        random_seed = 0,
                        log_svd_max = 12,
                        step_max = 2500,
                        )
    default_args.update(kwargs)
    return Namespace(**default_args)

def plot_all(model, skip_list = None, **kwargs):
    skip_list = [] if skip_list is None else skip_list
    skip_list0 = ['grad_align', 'monitor', 'scheduler', 'erank' ]  # , 'norm'
    skip_list += skip_list0

    plot_scalars(model.hparams.log_dir, skip_list=skip_list, **kwargs) #, 'sing_val j', 'sing_val k'])#, save_name = "grokking_in_matrix_completion3")
    
def run_exp(train_frac, extra_args_str = None, train_flag = True, **kwargs):  # , model0=None
    args = get_args(train_frac=train_frac, **kwargs)
    args_str = get_args_str(args) 
    if extra_args_str is not None:
        args_str += extra_args_str #' --use_different_logger'        # args_str += ' --record_wg_hist 1' 
    hparams = get_hparams(args_str.split(), default_kwargs=default_kwargs)

    model, datamodule, trainer = get_model_pkg(hparams)

    if train_flag:
        trainer.fit(model=model, datamodule=datamodule) 
        plot_all(model) #     plot_scalars(trainer.logger.log_dir, skip_list=['monitor', 'norm'])    

    return model, datamodule, trainer

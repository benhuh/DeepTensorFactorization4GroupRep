#!/usr/bin/env python

import os

from argparse import ArgumentParser, Namespace

import torch
from pytorch_lightning import Trainer #, LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping as EarlyStopping_orig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback

from dtf.lightning_model import LITmodel
from dtf.data import ( DEFAULT_DATA_DIR, get_default_N, create_mask, ArithmeticDataModule, set_random_seed)
from dtf.data import ( VALID_OPERATORS, ALGORITHMIC_OPERATORS, TOY_OPERATORS)
# from dtf.monitor import All_Monitor
from dtf.progress_bar import CustomProgressBar as TQDMProgressBar


import logging

logging_verbose_level = 0 #if not hparams.verbose else 'WARNING'
logging.getLogger('pytorch_lightning').setLevel(logging_verbose_level)  # hide printing of GPU available: True (cuda), used: True

DEFAULT_LOG_DIR = os.path.join(os.environ.get("LOGDIR", "."),"lightning_logs")
DEFAULT_PARSER_KWARGS = dict(logdir=DEFAULT_LOG_DIR)


def get_model_pkg(ckptpath_or_hparams, M=None):
    if isinstance(ckptpath_or_hparams, str):
        ckpt_path_list = get_ckptpath(ckptpath_or_hparams)
        print(f'loading: {ckpt_path_list}')
        # assert len(ckpt_path_list) == 1
        model = LITmodel.load_from_checkpoint(ckpt_path_list[0])
        hparams = model.hparams
        datamodule = get_datamodule(hparams, M)
        train_batch = len(datamodule.train_dataset)
        trainer = get_trainer(hparams, train_batch, ckpt_path_list[0]) #, ckpt_path)
    else:
        hparams = ckptpath_or_hparams

        seed = hparams.random_seed
        if getattr(hparams,'trial_num',None) is not None:  # seed > 0 and
            seed += hparams.trial_num

        set_random_seed(seed)
        model = LITmodel(hparams) #.float()        # Create the model
        datamodule = get_datamodule(hparams, M)
        train_batch = len(datamodule.train_dataset)
        trainer = get_trainer(hparams, train_batch)
        logger_dir = trainer.logger.log_dir
        model.hparams.logger_version = trainer.logger.version
        model.hparams.log_dir = logger_dir
        os.makedirs(logger_dir, exist_ok=True)

    model.train_batch = train_batch
    # model.Target = datamodule.train_dataset.M.permute(1,2,0).to_dense() + 0.0
    # model.Mask = create_mask(datamodule)

    return model, datamodule, trainer


def get_trainer(hparams, train_batch, ckpt_path=None):
    version = getattr(hparams, 'logger_version', None)
    logger = TensorBoardLogger(hparams.logdir, name = get_log_name(hparams), default_hp_metric=False, version=version)

    if hparams.verbose:
        print('log_dir:', logger.log_dir)

    # default earlystop
    hparams.earlystop += ["scheduler/counter"]

    earlystop_callback_dict = {
        "scheduler/counter":
            EarlyStopping(monitor="scheduler/counter", min_delta = -1, mode="max", stopping_threshold = hparams.counter_threshold[-1], check_on_train_epoch_end=True),
        "accuracy/val":
            EarlyStopping(monitor="accuracy/val", patience=int(2000/hparams.val_check_interval), min_delta = 1e-4, mode="max", stopping_threshold=99, check_on_train_epoch_end=False),
        "loss/train":
            EarlyStopping(monitor="loss/train", min_delta = -1, mode="min", stopping_threshold= 1e-7, divergence_threshold=1e2, check_on_train_epoch_end=True),
    }

    earlystop_callback = [earlystop_callback_dict[key] for key in [*set(hparams.earlystop)]] # set removes duplicates

    trainer_args = {
        "max_steps": 5000,#hparams.max_steps,
        "max_epochs": int(1e8),
        "val_check_interval": hparams.val_check_interval,
        "check_val_every_n_epoch": None,
        "num_sanity_val_steps": -1,  # run full validation in the beginning
        "profiler": False,
        "logger": logger,
        "log_every_n_steps": 1,
        "callbacks": [TQDMProgressBar(refresh_rate=1, enable_val=hparams.enable_val_progress),
                      # All_Monitor(logging_interval='step'),
                      Validation_On_Start_Callback(),
                      *earlystop_callback ] ,
        "gradient_clip_algorithm": "norm",
        "profiler": "pytorch" if hparams.profiler else None,
        }

    trainer_args.update({"gradient_clip_val": hparams.grad_clip * train_batch})

    if ckpt_path is not None:
        trainer_args.update({'ckpt_path': ckpt_path})  # For Next version of Trainer.

    # if torch.cuda.is_available() and len(hparams.gpus) > 0:
    #     trainer_args.update({ "accelerator": "gpu", "devices": hparams.gpus,        # "strategy": "ddp",
    #                           })
    trainer_args.update({"accelerator": "cpu"})
    if torch.backends.mps.is_available() and len(hparams.gpus) > 0:
        trainer_args.update({ "accelerator": "mps", "devices": hparams.gpus,        # "strategy": "ddp",
                              })

    trainer = Trainer(**trainer_args)      # trainer = Trainer(**trainer_args)
    return trainer

class Validation_On_Start_Callback(Callback):
    def on_train_start(self, trainer, pl_module):
        trainer.checkpoint_callback.on_train_start(trainer, pl_module)
        # return trainer.validate()
    ## commented out: assert self.evaluating in trainer._run_evaluate



def get_datamodule(hparams, M=None):

    seed = hparams.random_seed
    if getattr(hparams,'trial_num',None) is not None: # seed > 0
        seed += hparams.trial_num

    dm_args = { "data_dir": hparams.datadir,
                "prepare_data_per_node":  getattr(hparams, 'prepare_data_per_node', 'False'),
                # "loss_type": hparams.loss_type,
                "data_type": 'text' if hparams.model == 'Transformer' else 'tuple',
                "train_frac": hparams.train_frac,
                "batchsize_hint": hparams.batchsize,
                "seed": seed,
                "task": hparams.task_name,
                "M": M,
                }

    optional = ['task_rank', 'tensor_width', 'total_batch', 'teacher_arch', 'nonlinearity', ]
    for k in optional:
        dm_args[k] = getattr(hparams, k, None)

    datamodule =  ArithmeticDataModule(**dm_args)
    datamodule.setup()

    return datamodule


def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
    """
    Defines the hyperparameter arguments needed by instances of this
    class. This is intended to be called when parsing command line
    arguments.

    :param parser: an argparse.ArgumentParser created by the caller
    :returns: the argument parser with the command line arguments added
                for this class.
    """
    parser.add_argument("--model", type=str, default="Deep_Tensor_Net", choices=["Deep_Tensor_Net", "Deep_Tensor_Net_conv"])

    parser.add_argument("--optim", choices=["SGD"], default="SGD")
    parser.add_argument("--loss_fn", choices=["mse_loss"], default="mse_loss")
    parser.add_argument("--batchsize", type=float, default=1, help=" 0<N<=1 -> fraction of dataset",)

    ## for data
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--task_rank", type=int, default=None , nargs='+')
    parser.add_argument("--modulus", type=int, default=None)

    parser.add_argument("--trial_num", type=int, default=None)
    parser.add_argument("--train_frac", type=float, default=100) #25)

    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=2e-1) # default=3e-3)

    parser.add_argument("--betas", nargs='+', type=float, default=[0.9, 0.0]) #default=[0.8, 0.9])

    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--scheduler_criterion", nargs='+', type=str, default=['imbalance2/mean']) #None)
    parser.add_argument("--scheduler_threshold", type=float, default=1e-7)
    parser.add_argument("--scheduler_decay", type=float, default=0.0)
    parser.add_argument("--counter_threshold", nargs='+', type=int, default=[1,200])

    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--conv_weight_decay", type=float, default=0.1)
    parser.add_argument("--weight_decay_min", type=float, default=0)

    parser.add_argument("--use_scale", choices=[None, 'log', 'square', 'sqrt', 'exp'], default=None)  #, 'p1_reg', 'max'          # parser.add_argument("--use_scale", dest="use_scale", action="store_true", default=False) # log-scale for weight decay scheduler

    parser.add_argument("--custom_L2",  action="store_true", default=False)
    parser.add_argument("--manual_L2",  action="store_true", default=False)

    parser.add_argument("--log_imbalance", action="store_true", default=False)
    parser.add_argument("--log_svd_max", type=int, default=10)

    parser.add_argument("--record_wg_hist", type=int, default=0)

    parser.add_argument("--logdir", type=str, default=DEFAULT_LOG_DIR,)
    parser.add_argument("--datadir", type=str, default=DEFAULT_DATA_DIR,)
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)

    parser.add_argument("--earlystop", type=str, nargs='+', default=[])

    return parser

###############################################

def get_hparams(*args, default_kwargs=None, parser=None) -> Namespace:
    """
    Parses the command line arguments
    :returns: an argparse.Namespace with all of the needed arguments
    """
    if parser is None:
        parser = ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpus", nargs='+', type=int, default=[])
    parser.add_argument("--val_check_interval", type=int, default=10)

    parser.add_argument("--enable_val_progress", dest="enable_val_progress", action="store_true", default=False)
    parser.add_argument("--profiler", dest="profiler", action="store_true", default=False)

    # add model specific args
    parser = add_model_specific_args(parser)    #  LITmodel.add_model_specific_args(parser)
    # parser = Trainer.add_argparse_args(parser)

    if default_kwargs is not None:
        DEFAULT_PARSER_KWARGS.update(default_kwargs)
    parser.set_defaults(**DEFAULT_PARSER_KWARGS)
    hparams, unknown = parser.parse_known_args(*args)

    model_parser = LITmodel.get_model_parser(hparams.model)
    hparams2, unknown_ = model_parser.parse_known_args(unknown)
    hparams = Namespace(**vars(hparams), **vars(hparams2))

    if len(unknown_)>0:
        print('unknown args keys:', unknown_)

    # Process the args
    if hparams.logdir is None:
        hparams.logdir = DEFAULT_LOG_DIR # os.environ.get("LOGDIR", ".")

    hparams.datadir = os.path.abspath(hparams.datadir)
    hparams.logdir = os.path.abspath(hparams.logdir)
    hparams.checkpoint_path = hparams.logdir + "/checkpoints"

    hparams = get_tensor_width(hparams)

    if hparams.model in ["Deep_Tensor_Net"]:
        hparams.custom_L2 = not hparams.manual_L2
        hparams.decomposition_type = 'FC'

        hparams.tensor_width = extract_first_element(hparams.tensor_width)
        hparams.model_rank = extract_first_element(hparams.model_rank)
        hparams.task_rank = extract_first_element(hparams.task_rank)

    return hparams


def extract_first_element(n):
    if isinstance(n, (list,tuple)) and len(n)==1:
        n = n[0]
    if n is None:
        n = 0
    # assert isinstance(n,int)
    return n

def get_tensor_width(hparams):
    if hparams.model in ["Deep_Tensor_Net", "Deep_Tensor_Net_Lie", "Transformer"]:
        if getattr(hparams,'tensor_width',None) in [None, 0]:
            hparams.tensor_width = get_default_N(hparams.task_name, default_modulus = hparams.modulus)
    if hparams.verbose:
        print("Task:", hparams.task_name, "tensor_width", hparams.tensor_width) #, "Total batch:", hparams.total_batch

    return hparams

#######################################




##################################


def get_print_str(hparams):
    if isinstance(hparams,dict):
        hparams = Namespace(**hparams)
    log_name = get_log_name(hparams)
    print(log_name)


def get_log_name(hparams): #, print_flag=False):
    # task = hparams.task_name.replace('**','^').replace('/*','/xy').replace('*','').replace('//','/x_div_y')
    task = VALID_OPERATORS.get(hparams.task_name)
    if task is None:
        task = hparams.task_name
        # raise ValueError(f"Unknown task: {hparams.task_name}")
    # print('task:', hparams.task_name, task)

    if getattr(hparams,'modulus') is not None:
        task += f'_mod_{hparams.modulus}'

    if hparams.task_name in (ALGORITHMIC_OPERATORS|TOY_OPERATORS):
        if hparams.tensor_width is not None and hparams.tensor_width != 1:
            task += f"_{hparams.tensor_width}"
    else:
        if hparams.tensor_width is not None and hparams.tensor_width != 1:
            task += f"_{hparams.tensor_width}"
        if hparams.task_rank is not None and ("completion" in hparams.task_name) and hparams.task_rank is not None:
            task += f"rank{hparams.task_rank}"
        if getattr(hparams,'teacher_arch',None) is not None:
            task += f"_teacher{hparams.teacher_arch}"

    # model_str = hparams.model #'' if hparams.model == 'Deep_Tensor_Net ' else hparams.model+" "

    if hparams.model == 'Transformer':
        model_str = hparams.model
    else:
        assert hparams.model in ['Deep_Tensor_Net', 'Deep_Tensor_Net_Lie', 'Deep_Tensor_Net_conv']
        model_str = f"{hparams.decomposition_type}"

        if hasattr(hparams,'model_rank') and hparams.model_rank != 0:
            model_str += f"_rank{hparams.model_rank}"

    optim_str = f"{hparams.optim}"

    assert len(hparams.betas)==1 or hparams.betas[-1]==0
    optim_str += f" momentum={str(hparams.betas[0])}"

    if getattr(hparams,'custom_L2',False):
        optim_str += " customL2"
    if getattr(hparams,"manual_L2",False):
        optim_str +=  " manual_L2"

    optim_str += f"/lr={str(hparams.lr)}"
    if hparams.weight_decay != 0:
        optim_str += f" wd={str(hparams.weight_decay)}"
    if hasattr(hparams,'init_scale') and hparams.init_scale != 1:
        optim_str += f" init={hparams.init_scale}"

    loss_fn_str = hparams.loss_fn

    misc = ""
    if hparams.random_seed != 0:
        misc += f"/seed{int(hparams.random_seed)}"
    if hparams.batchsize != 1:
        misc += " batch"+str(hparams.batchsize)

    train_frac_str = "frac"
    train_frac = hparams.train_frac if hparams.train_frac>1 else hparams.train_frac*100
    train_frac_str += f"={train_frac}"

    exp_task_name = f"{hparams.exp_name}/{loss_fn_str}/{task}/"
    log_name = f"{model_str}/{optim_str + misc}/{train_frac_str}"
    return exp_task_name+log_name

from pathlib import Path

def get_ckptpath(path):  #, version_limit=None  # Recursive!!
    path = Path(path).expanduser()
    ckptpath_list = [path] if path.suffix=='.ckpt'  else  list(path.glob('*.ckpt'))     # if path.suffix=='.ckpt':   # path.stem /  path.name

    if len(ckptpath_list) == 0:
        sub_path_list = list(path.glob('*'))        # sub_path_list.sort() #(key=os.path.getmtime)  # sort by the time of last modification
        for sub_path in sub_path_list:
            ckptpath_list += get_ckptpath(sub_path)   # Recursive!!

    return ckptpath_list

#######################################


class EarlyStopping(EarlyStopping_orig):
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        current_step = trainer.fit_loop.total_batch_idx + 1
        if current_step <= trainer.val_check_interval: #20: # to avoid the bug in which the validation logging doesn't work for first few epochs.
            return

        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return

        self._run_early_stopping_check(trainer)

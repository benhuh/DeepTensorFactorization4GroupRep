#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt

from dtf.train_helper_script import run_exp, plot_all

from dtf.visualization_new import plot_tensor_slices2, show_netWeight_hist, show_sparse_hist, get_regular_representation, get_regular_representation_XYZ, normalize_factor_list, reorder_tensor, plot_netWeight_with_train_data

def train(task_name, train_frac, seed=1, loss_fn = 'mse_loss', scheduler_threshold = 1e-5, L2_reg=False, gpus=None, val_check_interval=5, tensor_width=0, weight_decay=0.1, no_regularization = False, add_str=None, train_flag=True):
    # Shouldn't we have a way to control here if it is a transformer or a DFN?

    # Ben said we should have weight decay for the filters, probably
    # don't add it here because it will also add weight decay to T.
    # Maybe have two optimizers?
    weight_decay = 0 if no_regularization else weight_decay
    weight_decay_min =  0
    lr, momentum, lr_Lagrange, counter_threshold = (0.5/2, 0.5, 0.05*400, '30 90')  # SGD

    save_name = get_save_name(task_name,train_frac,seed,no_regularization, add_str)
    extra_args_str = ''# None
    if L2_reg:
        extra_args_str += ' --decomposition_type FC_embed0 --manual_L2'
        weight_decay = 1.5
        save_name += '_L2'
        scheduler_threshold = 100
        counter_threshold = '250 100' #'300 50'

    optim = 'SGD'
    out = run_exp(train_frac=train_frac, extra_args_str = extra_args_str,
                       optim = optim, val_check_interval=val_check_interval, loss_fn = loss_fn,
                       task_name = task_name, tensor_width = tensor_width or 6,
                       lr = lr, momentum=momentum, weight_decay = weight_decay,
                       scheduler_threshold = scheduler_threshold, weight_decay_min = weight_decay_min, lr_min = 1,
                       random_seed=seed, record_wg_hist=10, grad_clip=1e-1,
                       counter_threshold = counter_threshold, gpus=gpus, train_flag=train_flag)
    return out, save_name

def get_save_name(task_name,train_frac,seed,no_regularization, add_str):
    task = task_name.split('/')[1]
    task = task_name.replace('/','_')
    save_name = f'figure_ICML/{task}_{train_frac}_seed{seed}'
    if no_regularization:
        save_name += '_noReg'
    if add_str is not None:
        save_name += '_'+add_str
    return save_name

def reorder_factor_list(factor_list, new_order, **kwargs):
    factor_list_reordered=[]
    for A in factor_list:
        factor_list_reordered.append(reorder_tensor(A, new_order).squeeze())
    return torch.stack(factor_list_reordered)
    # plot_tensor_slices2(factor_list_reordered, idx_name_cols=['A','B','C'], **kwargs)
    #
def generate_figures(model, datamodule, save_name=None, new_order=None, XYZ_prev=None, normalize_1 = None, which_tensor=[0], show_num0=3, skip=2, show_steps = 4, t_init=0, plot_all_weights=False, normalize_later=False, ABC_or_A = 'A'):

    plot_all(model, skip_list = ['norm','grad_norm'], save_fig=True, save_name=save_name)

    if plot_all_weights:
        plot_netWeight_with_train_data(model, datamodule, save_name=save_name, M_lim=8, )
        if ABC_or_A=='ABC':
            plot_tensor_slices2(model.model.factor_list, idx_name_cols=['A','B','C'], save_name=save_name, save_name_extra='ABC_raw')
        else:
            plot_tensor_slices2(model.model.factor_list[0], idx_name_cols=['A'], M_lim=8, save_name=save_name, save_name_extra='A_raw')

    # show_steps = 4 #5 # 6
    show_num=[show_num0,skip*show_steps+1+t_init]

    if save_name is not None and ('noReg' in save_name): # or 'L2' in save_name):
        sparsity_type = None
        sparsity_type_str = 'raw'
    else:
        sparsity_type = 'block_diag'
        sparsity_type_str = sparsity_type # or 'raw'

        if plot_all_weights:
            if ABC_or_A=='ABC':
                plot_tensor_slices2(normalize_factor_list(model.model.factor_list), idx_name_cols=['A','B','C'], save_name=save_name, save_name_extra='ABC_normalized')
            else:
                plot_tensor_slices2(normalize_factor_list(model.model.factor_list)[0], idx_name_cols=['A'], M_lim=8, loss_type=sparsity_type, save_name=save_name, save_name_extra=f'A_normalized')

    XYZ, factor_list_sparse = get_regular_representation_XYZ(model, datamodule, plot_flag=False, XYZ=XYZ_prev, normalize_1=normalize_1, steps=1500, lr=1, loss_type=sparsity_type, idx_name_cols=['A','B','C'], )

    if new_order is not None:
        assert XYZ_prev == None
        factor_list_sparse = reorder_factor_list(factor_list_sparse, new_order)
        XYZ = XYZ[:,:,new_order]
    if plot_all_weights:
        if ABC_or_A=='ABC':
            if normalize_later:
                plot_tensor_slices2(normalize_factor_list(factor_list_sparse), idx_name_cols=['A','B','C'], loss_type=sparsity_type, save_name=save_name, save_name_extra=f'ABC_{sparsity_type_str}_normalized')
            else:
                plot_tensor_slices2(factor_list_sparse, idx_name_cols=['A','B','C'], loss_type=sparsity_type, save_name=save_name, save_name_extra=f'ABC_{sparsity_type_str}')
        else:
            plot_tensor_slices2(factor_list_sparse[0], idx_name_cols=['A'], M_lim=8, loss_type=sparsity_type, save_name=save_name, save_name_extra=f'A_{sparsity_type_str}')

    A_hist_, XYZ_ = show_sparse_hist(model, datamodule, which_tensor=which_tensor, new_order=None, XYZ = XYZ, animate=False, skip=skip, t_init=t_init, show_num=show_num, M_lim=8, loss_type=sparsity_type, save_name=save_name, save_name_extra=f'A_hist_{sparsity_type_str}')
    show_netWeight_hist(model, datamodule, hist_name='netW_hist', animate=False, skip=skip, t_init=t_init, show_num=show_num, M_lim=8, z_pow=0.6, save_name=save_name)

    return XYZ

seed = 2
train_frac = 61
task_name = 'binary/sym3_xy_vec'

# change task name to '*_vec' to get a vectorized version

out, save_name_sym3_61_seed2 = train(task_name, train_frac, seed=seed,  val_check_interval=5, no_regularization=False)
# generate figures doesn't really work
# (model_sym3_61_seed2, datamodule_sym3_61_seed2, trainer) = out
# XYZ_sym3_61_seed2 = generate_figures(model_sym3_61_seed2, datamodule_sym3_61_seed2, save_name_sym3_61_seed2, skip=15, t_init=0, show_steps=5, new_order=[5,4,3,2,1,0], plot_all_weights=True, ABC_or_A = 'ABC')

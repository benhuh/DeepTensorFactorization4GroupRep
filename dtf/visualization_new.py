#!/usr/bin/env python

import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick
import numpy as np
import torch

# def take_diag(ABC):
#     ABC_new = []
#     for T in ABC:
#         T_ = []
#         for t in T:
#             T_.append(t.diag())
#         ABC_new.append( torch.stack(T_))
#     return ABC_new

plt.rcParams['text.usetex'] = False #True
plt.rcParams['savefig.transparent'] = True


def check_group(datamodule, plot_flag=False): #anti=False):
    M = datamodule.train_dataset.M.to_dense()+0      # cab
    M = M.permute(1,2,0)      # cab -> abc

    type_list = ['group', 'antigroup']
    err_dict = {}

    for type in type_list:
        if type == 'group':
            M_ = M     #  abc
            A,B,C = M_.permute(0,2,1), M_.permute(0,2,1), M_
        elif type == 'antigroup':
            M_ = M.permute(2,1,0)    # abc -> cba
            A,B,C = M_.permute(0,2,1), M_.permute(0,2,1), M_
            # M_ = M     #  abc
            # A,B,C =  M.permute(2,1,0), M.permute(2,0,1), M.permute(2,0,1)            # A,B,C =  M_, M_.permute(0,2,1), M_.permute(0,2,1),
        else:
            raise ValueError

        T = torch.einsum('aij, bjk, cki->abc', A,B,C)
        T = T/T.shape[0]
        err = M_ - T
        err_dict[type] = err.norm().item()
        if plot_flag:
            plot_tensor_slices2(M_)
            plot_tensor_slices2(T)
    return err_dict


def normalize_factor_list(factor_list, ABC0 = None, normalize_0=True, normalize_1=True):
    if isinstance(factor_list, (list,tuple)):
        factor_list = torch.stack(factor_list)
    if len(factor_list.shape)==5 and factor_list.shape[1]==3:  # time x ABC: T x 3 x (6,6,6)
        factor_list = factor_list.permute(1,0,2,3,4)    # 3 x T x (6,6,6)
    else:
        assert len(factor_list.shape)==4   #  ABC:  3 x (6,6,6)

    factor_list = factor_list.detach()

    assert factor_list.shape[0]==3

    if normalize_0:  # scale by sqrt of group_order
        n = factor_list.shape[-1]

    A,B,C = factor_list
    if normalize_1:
        A0,B0,_ = factor_list[:,0] if len(factor_list.shape)==4 else factor_list[:,-1,0]  # 0th element and last time_step
        # return [A0.inverse()@A,   B@B0.inverse(),  B0@C@A0]           # return [A0.T@A,   B@B0.T,  B0@C@A0]
        return [A0.T@A,   B@B0.T,  B0@C@A0]
    else:
        return [A, B, C]


def get_loss_fn(loss_type):
    def regular_loss(A_,M):
        assert A_.shape[-2:] == M.shape[-2:]
        # if A_.shape[1] != M.shape[0]:
        #     q = M.shape[0] // A_.shape[1]
        #     M = M[::q]
        return (A_-M).pow(2).mean()
    def sparse_loss(A_,M):
        loss = A_.abs().pow(1).mean()      # loss = ABC_.relu().pow(2).mean() * 1
        # loss += (1-A_).relu().pow(2).mean()  * 3
        # loss += 10 * identity_loss(A_,M)
        return loss


    from scipy.linalg import toeplitz
    def offdiag_loss(ABC_, log_scale=True):
        shape = ABC_.shape
        n = shape[-1]
        # loss = 0
        # vec = [i for i in range(n)]
        vec = [i for i in range(1,n+1)]
        mask = toeplitz(vec, vec)
        mask = torch.tensor(mask, dtype=ABC_.dtype, device=ABC_.device).unsqueeze(0)
        if log_scale:
            mask = mask.log()
        else:
            mask = mask - 1
        loss = (mask*ABC_).abs().pow(2).mean()

        return loss

    def offdiagonal_loss(A_,M):
        return offdiag_loss(A_)

    def identity_loss(A_,M):
        A_0 = A_[:,0]
        Id = torch.eye(A_.shape[-1], device=A_.device, dtype=A_.dtype)
        return (A_0-Id).pow(2).mean()

    loss_dict = dict(regular=regular_loss, sparse=sparse_loss, block_diag=offdiagonal_loss, identity=identity_loss)
    return loss_dict[loss_type]

def diagonalize(ABC, V, loss_type, fit_index=None):  #fit_index=[0,1,2]

    if fit_index is not None:        # fit_index = [0,1,2]
        ABC = ABC[fit_index]

    if ABC.dtype != V.dtype:
        ABC = ABC.to(V.dtype)

    if loss_type=='regular':  # if inv_or_trans=='trans':
        ABC_ = V.T @ ABC @ V
    else:
        ABC_ = V.inverse() @ ABC @ V        # ABC_ = torch.linalg.pinv(V) @ ABC @ V
    return ABC_

def diagonalize_T(T, V, loss_type, fit_index=None):  #fit_index=[0,1,2]

    if fit_index is not None:        # fit_index = [0,1,2]
        T = T[fit_index]

    if T.dtype != V.dtype:
        T = T.to(V.dtype)

    if loss_type=='regular':  # if inv_or_trans=='trans':
        T_ = torch.einsum('ijk,jl->ilk', T, V.T)
    else:
        T_ = torch.einsum('ijk,jl->ilk', T, V.inverse())
    return T_

def optimize_T(T, V, M, lr, loss_type, steps=100, reg_coeff=0.1, fit_index=None, loss_all=None, idx_sort=None, idx_sign=None):
    if loss_type == 'exact':
        # T = V @ M
        # V = T @ M ^{-1}
        # permute the indices first
        # T = T.permute(0, 2, 1)
        # T = T.view(-1, T.shape[-1])
        V = M[:, 0, :] @ torch.linalg.pinv(T[:, 0, :]) # should be the same for every slice
        return V
    loss_all = loss_all or []
    V = torch.nn.Parameter(V)
    optim = torch.optim.SGD([V], lr=lr, momentum=0.9)
    loss_fn = get_loss_fn(loss_type)

    for _ in range(steps):
        optim.zero_grad()
        T_ = diagonalize_T(T, V, loss_type, fit_index=fit_index)             # ABC_ = diagonalize(ABC, V, loss_type='sparse', fit_index=fit_index)
        loss = loss_fn(T_,M) + reg_coeff * (V.T @ V - torch.eye(V.shape[0]).to(V.device)).pow(2).mean()
        loss.backward()
        optim.step()
        loss_all.append(loss.item())

    if idx_sort is not None:
        V = V[:,idx_sort]

    T_ = diagonalize_T(T, V, loss_type, fit_index=None)
    return V.detach(), T_.detach(), loss_all

def optimize_V(ABC, V, M, lr, loss_type, steps=100, reg_coeff=0.1, fit_index=None, loss_all=None, idx_sort=None, idx_sign=None):
    loss_all = loss_all or []
    V = torch.nn.Parameter(V)
    optim = torch.optim.SGD([V], lr=lr, momentum=0.9)
    loss_fn = get_loss_fn(loss_type)

    for _ in range(steps):
        optim.zero_grad()
        ABC_ = diagonalize(ABC, V, loss_type, fit_index=fit_index)             # ABC_ = diagonalize(ABC, V, loss_type='sparse', fit_index=fit_index)
        loss = loss_fn(ABC_,M)   +  reg_coeff * (V.T @ V - torch.eye(V.shape[0]).to(V.device)).pow(2).mean()
        loss.backward()
        optim.step()
        loss_all.append(loss.item())


    if idx_sort is not None:
        V = V[:,idx_sort]

    # if idx_sign is not None:
    #     V[:,idx_sign] *= -1

    ABC_ = diagonalize(ABC, V, loss_type, fit_index=None)
    return V.detach(), ABC_.detach(), loss_all

def get_regular_representation(model_or_factor_list, datamodule, V=None, plot_flag=False, steps=500, lr=1, loss_type='block_diag', trans_list=[0,1], anti_group=False, normalize_0=True, normalize_1=True, fit_index=None, show_error=False, idx_sort = None, idx_sign=None, **kwargs):
    M_factors = datamodule.train_dataset.factors
    if M_factors is not None:
        M = torch.stack(M_factors,dim=0)
    else:
        M = datamodule.train_dataset.M.permute(1,2,0).to_dense()+0   if hasattr(datamodule, 'train_dataset') else datamodule   # cab -> abc  (for addition / sym)
        if anti_group:
            M = M.permute(2,1,0)    # abc -> cba

    if isinstance(model_or_factor_list, (list,tuple, torch.Tensor)):
        factor_list = model_or_factor_list
    else:
        factor_list = model_or_factor_list.model.factor_list

    if loss_type is None:
        pass
    else:
        factor_list = normalize_factor_list(factor_list, normalize_0=normalize_0, normalize_1=normalize_1)

    for i in trans_list:
        factor_list[i] = factor_list[i].permute(0,2,1)      # factor_list = [A.permute(0,2,1),B.permute(0,2,1),C]     # Huh? Transpose A,B
    ABC = torch.stack(factor_list) #.detach()



    if V is not None:
        ABC_ = diagonalize(ABC, V, loss_type, fit_index=None)
        factor_list = [A_ for A_ in ABC_]

        # return V.detach(), factor_list_
    elif loss_type is None :
        pass
    else:
        # if loss_type != 'regular':  # if inv_or_trans == 'inv':
        V = torch.eye(ABC.shape[-1]);
        # else:
        # V = torch.randn(ABC.shape[-2:]);

        loss_all = []
        V, ABC_, loss_all = optimize_V(ABC, V, M, lr, loss_type, steps, loss_all=loss_all, fit_index=fit_index, idx_sort=idx_sort, idx_sign=idx_sign)

        if show_error and loss_type == 'regular':
            plot_tensor_slices2(ABC_ - M)

        if plot_flag:
            fig, ax = plt.subplots(1, 1, figsize=(3*1,2), sharex=False, sharey=False)
            ax.semilogy(loss_all,'-');    plt.show()
        factor_list = [A_ for A_ in ABC_]
    for i in trans_list:
        factor_list[i] = factor_list[i].permute(0,2,1)

    factor_list = torch.stack(factor_list).detach()

    if plot_flag:
        plot_tensor_slices2(factor_list.squeeze(), modify_dim=-1, **kwargs)
        # plot_tensor_slices2(M)

    return V.detach(), factor_list


def optimize_XYZ(ABC, XYZ, M, lr, loss_type, steps=100, reg_coeff=0.0, fit_index=None, loss_all=None):
    loss_all = loss_all or []
    XYZ = torch.nn.Parameter(XYZ)
    optim = torch.optim.SGD([XYZ], lr=lr, momentum=0.9)
    loss_fn = get_loss_fn(loss_type)

    for _ in range(steps):
        optim.zero_grad()
        ABC_ = diagonalize_XYZ(ABC, XYZ, loss_type)
        loss = loss_fn(ABC_,M)

        if reg_coeff > 0:
            # import pdb; pdb.set_trace()
            for X in XYZ:
                loss += reg_coeff * (X.T @ X - torch.eye(X.shape[0]).to(X.device)).pow(2).mean()

        loss.backward()
        optim.step()
        loss_all.append(loss.item())


    ABC_ = diagonalize_XYZ(ABC, XYZ, loss_type)
    return XYZ.detach(), ABC_.detach(), loss_all


def diagonalize_XYZ(ABC, XYZ, loss_type):
    X,Y,Z = XYZ
    if len(ABC.shape) == 4:
        pass
    elif len(ABC.shape) == 5:
        ABC = ABC.permute(1,0,2,3,4)
    else:
        raise ValueError
    A,B,C = ABC

    if loss_type=='regular':  # if inv_or_trans=='trans':
        A_ = X.T @ A @ Y
        B_ = Y.T @ B @ Z
        C_ = Z.T @ C @ X
    else:
        A_ = X.inverse() @ A @ Y
        B_ = Y.inverse() @ B @ Z
        C_ = Z.inverse() @ C @ X

    ABC_ = torch.stack([A_, B_, C_])
    if len(ABC.shape) == 5:
        ABC_ = ABC_.permute(1,0,2,3,4)
    return ABC_

from pytorch_lightning import LightningDataModule

def get_regular_representation_XYZ(model_or_factor_list, datamodule = None, normalize_1 = None, XYZ=None, plot_flag=False, steps=500, lr=1, loss_type='block_diag', trans_list=[0,1], anti_group=False, fit_index=None, **kwargs):

    if normalize_1 is None:
        normalize_1 = True if XYZ is None else False

    if isinstance(model_or_factor_list, (list,tuple, torch.Tensor)):
        factor_list0 = model_or_factor_list
    else:
        factor_list0 = model_or_factor_list.model.factor_list
    factor_list = normalize_factor_list(factor_list0, normalize_0=True, normalize_1=normalize_1)

    ABC = torch.stack(factor_list) #.detach()

    if datamodule is None:
        M_factors = None
    elif isinstance(datamodule, LightningDataModule) and hasattr(datamodule,'train_dataset'):
        M_factors = datamodule.train_dataset.factors
    else:
        M_factors = datamodule

    if M_factors is not None:
        M = torch.stack(M_factors,dim=0)
    else:
        M = datamodule.train_dataset.M.permute(1,2,0).to_dense()+0   if hasattr(datamodule, 'train_dataset') else datamodule   # cab -> abc  (for addition / sym)
        if M is not None:
            if anti_group:
                M = M.permute(2,1,0)    # abc -> cba
            M = M.unsqueeze(0).repeat(ABC.shape[0],1,1,1) #.clone()
            M_list = [M_ for M_ in M]
            for i in trans_list:
                M_list[i] = M_list[i].permute(0,2,1)
            M = torch.stack(M_list)


    if XYZ is not None:
        assert normalize_1 == False
        ABC_ = diagonalize_XYZ(ABC, XYZ, loss_type)
        factor_list = [A_ for A_ in ABC_]
    elif loss_type is None :
        pass
    else:
        # assert normalize_1 == True
        loss_all = []
        XYZ = torch.eye(ABC.shape[-1]).unsqueeze(0).repeat(ABC.shape[0],1,1)
        XYZ, ABC_, loss_all = optimize_XYZ(ABC, XYZ, M, lr, loss_type, steps, reg_coeff=0.1, fit_index=fit_index, loss_all=loss_all)

        if normalize_1:
            XYZ = get_effective_XYZ(XYZ, factor_list0)

        if plot_flag:
            fig, axes = plt.subplots(1, 2, figsize=(3*2,2), sharex=False, sharey=False)
            axes[0].semilogy(loss_all,'-');    plt.show()

        factor_list = [A_ for A_ in ABC_]

    factor_list = torch.stack(factor_list).detach()

    if plot_flag:
        plot_tensor_slices2(factor_list.squeeze(), modify_dim=-1,)

    return XYZ.detach(), factor_list

def get_effective_XYZ(XYZ, factor_list0):
    A,B,C = normalize_factor_list(factor_list0, normalize_0=True, normalize_1=False)
    X,Y,Z = XYZ

    X_ = A[0] @ X
    Y_  = Y
    Z_ = B[0].T @ Z
    return torch.stack([X_,Y_,Z_])

def return_to_factors(ABC_, V_all):
    V = V_all#[0]
    # ABC = V @ ABC_ @ V.inverse()
    V_inv = V.inverse()
    ABC = V_inv.T @ ABC_ @ V_inv
    return ABC

def get_default_save_name(log_dir=None, add_str=None, save_fig=False, save_name=None):

    save_name_ = ''
    if save_name is not None:
        save_name_ = save_name + save_name_
    if add_str is not None:
        save_name_ += '_' + add_str
    if log_dir is not None:
        save_name_ += log_dir.split('mse_loss/')[-1].replace('/', '_').replace(' ', '_')
    return save_name_

from dtf.tensor_operations import tensor_prod
def process_train_data_and_netW_hist(netW_hist, datamodule, idx_order): #'abc'):
    ab,c = datamodule.train_dataset.tensors #.permute(1,2,0)
    # train_data = torch.cat((ab,c.unsqueeze(1)),dim=1)

    if idx_order=='abc':
        if len(netW_hist.shape) == 3:
            netW_hist = netW_hist.permute(1,2,0)  # cab -> abc
        elif len(netW_hist.shape) == 4:
            netW_hist = netW_hist.permute(0,2,3,1)  # t cab -> t abc
        else:
            raise ValueError
        idx_name = ['T']
    elif idx_order=='cab':
        if len(netW_hist.shape) == 3:
            netW_hist = netW_hist.permute(0,2,1)  # cab -> cba  # so that y-axis: a, x-axis: b
        elif len(netW_hist.shape) == 4:
            netW_hist = netW_hist.permute(0,1,3,2)  # t cab -> t cba  # so that y-axis: a, x-axis: b
        else:
            raise ValueError
        # train_data = train_data[:,[2,0,1]]          # train_data = train_data[:,[2,0,1]]
        idx_name = ['T_{\cdot \cdot}']

    return netW_hist, train_data, idx_name

def show_netWeight_hist(model, datamodule, idx_order='cab', hist_name = 'netW_hist', animate=False, t_init=0, **kwargs):
    netW_hist = torch.stack(getattr(model,hist_name)).detach().cpu()
    netW_hist, train_data, idx_name = process_train_data_and_netW_hist(netW_hist, datamodule, idx_order)
    t0=model.hparams.record_wg_hist
    save_name_extra = kwargs.pop('save_name_extra', None) or hist_name
    log_dir = None #model.hparams.log_dir

    if animate:
        ani = plot_tensor_slices_animation(netW_hist, idx_name_rows=None, idx_name_cols=idx_name, t0=t0, train_data=train_data, transpose = False, log_dir = log_dir, save_name_extra=save_name_extra, **kwargs)
        return ani
    else:
        netW_hist = netW_hist.permute(1,0,2,3)  # t on the 1st idx.
        plot_tensor_slices2(netW_hist, idx_name_rows=idx_name, idx_name_cols=['t='], t0=t0, t_init=t_init, train_data=train_data, transpose = False, log_dir = log_dir, save_name_extra=save_name_extra, **kwargs)
        # return netW_hist

def plot_netWeight_with_train_data(model, datamodule, idx_order='cab', **kwargs):
    W = model.model._net_Weight.detach()
    W, train_data, idx_name = process_train_data_and_netW_hist(W, datamodule, idx_order)
    plot_tensor_slices2(W,  idx_name_cols=['T_{\cdot \cdot}'], train_data=train_data, transpose = False, save_name_extra='T', abc_dim=1, **kwargs)


import math

def show_grad_hist(model, datamodule, idx_order='abc', **kwargs):
    t0=model.hparams.record_wg_hist

    # ABC_hist = torch.stack(model.w_hist,dim=0).cpu()
    ABC_hist = show_sparse_hist(model, datamodule, t_ref = kwargs.get('t_ref',-1))[0]; ABC_hist = ABC_hist / (ABC_hist.shape[2])**(0.5)

    # ABC_hist.requires_grad = True
    A_hist, B_hist, C_hist = ABC_hist.permute(1,0,2,3,4)
    netW_hist = torch.einsum('taij, tbjk, tcki->tcab',  A_hist, B_hist, C_hist) * math.sqrt(A_hist.shape[1])

    error = netW_hist - (datamodule.train_dataset.M.to_dense()+0)
    assert idx_order=='abc'
    error = error.permute(0,2,3,1)  # t cab -> t abc
    error = error.permute(1,0,2,3)  # t on the 1st idx. -> a t bc

    ab,c = datamodule.train_dataset.tensors #.permute(1,2,0)
    abc = torch.cat((ab,c.unsqueeze(1)),dim=1)
    N = netW_hist.shape[1]
    mask = torch.sparse_coo_tensor(ab.T,torch.ones(ab.shape[0]), size=(N,)*ab.shape[1]).to_dense()
    mask = mask.unsqueeze(1).unsqueeze(-1)

    save_name_extra = 'netgrad'
    # plot_tensor_slices2(error.detach(), idx_name_rows='T', idx_name_cols='t=', t0=t0, train_data=abc, log_dir = model.hparams.log_dir, save_name_extra=save_name_extra, **kwargs)

    netT_grad = mask*error
    plot_tensor_slices2(netT_grad.detach(), idx_name_rows='T', idx_name_cols='t=', t0=t0, train_data=abc, log_dir = model.hparams.log_dir, save_name_extra=save_name_extra, **kwargs)

    # loss = (mask*error).pow(2).mean()/2
    # loss.backward()
    # A_grad_hist, B_grad_hist, C_grad_hist = ABC_hist.grad.permute(1,0,2,3,4)
    # A_grad_hist = A_grad_hist.permute(1,0,2,3)  # t on the 1st idx. -> a t bc
    # plot_tensor_slices2(A_grad_hist, idx_name_rows='A', idx_name_cols='t=', t0=t0, train_data=None, log_dir = model.hparams.log_dir, save_name_extra=save_name_extra, **kwargs)

    A_grad = torch.einsum('atbc, tbjk, tcki->atij',  netT_grad, B_hist, C_hist)
    plot_tensor_slices2(A_grad.detach(), idx_name_rows='A', idx_name_cols='t=', t0=t0, train_data=None, log_dir = model.hparams.log_dir, save_name_extra=save_name_extra, **kwargs)
    return netT_grad, A_grad

def show_sparse_hist(model, datamodule, datamodule_regular = None, which_tensor=[0,1,2], factor_list = None, t_ref=None, V=None, XYZ = None, loss_type = 'block_diag', animate=False, normalize_1=True, show_netWeight=True, new_order=None, fit_index = None, **kwargs): #, ABC0=None, lr=0.01, reference_step=-1, loss_type='regular', inv_or_trans='inv', plot_flag=False, anti_group=False, trans_list=[]):  #, trans_list=[0,1]
    if hasattr(model, 'trainer'):
        print(model.hparams.log_dir)
    factor_list = factor_list or model.model.factor_list  if t_ref is None  else model.w_hist[t_ref].detach().cpu()  #reference_step]

    if XYZ is None:
        datamodule_regular = datamodule_regular or datamodule
        if fit_index == None:
            XYZ, ABC_ = get_regular_representation_XYZ(factor_list, datamodule_regular, loss_type=loss_type, normalize_1=normalize_1, XYZ=XYZ, **kwargs) #, plot_flag=plot_flag, lr=lr, loss_type=loss_type, anti_group=anti_group, trans_list=trans_list, fit_index=fit_index) #, inv_or_trans=inv_or_trans)
        else:
            XYZ, ABC_ = get_regular_representation(factor_list, datamodule_regular, loss_type=loss_type, normalize_1=normalize_1, XYZ=XYZ, fit_index=fit_index, **kwargs) #, plot_flag=plot_flag, lr=lr, loss_type=loss_type, anti_group=anti_group, trans_list=trans_list, fit_index=fit_index) #, inv_or_trans=inv_or_trans)

        if new_order is not None:
            ABC_ = reorder_tensor(ABC_, new_order)
            XYZ = XYZ[:,:,new_order]  if fit_index == None else XYZ[:, new_order]
    else:
        assert new_order == None
        normalize_1 = False


    ABC_hist = torch.stack(model.w_hist,dim=0).cpu()
    if XYZ is not None:
        ABC_hist_normalized = torch.stack(normalize_factor_list(ABC_hist, ABC0 = None, normalize_0=True, normalize_1=False), dim=1)
        if fit_index == None:
            ABC_hist_ = diagonalize_XYZ(ABC_hist_normalized, XYZ, loss_type=loss_type)
        else:
            ABC_hist_ = diagonalize(ABC_hist_normalized, XYZ, loss_type=loss_type, fit_index=None)
    else:
        ABC_hist_ = ABC_hist
        if new_order is not None:
            ABC_hist_ = reorder_tensor(ABC_hist_, new_order)


    idx_name_all = ['A','B','C']         # for cyclic group (addition)        # idx_name = ['e','(1,2)','(2,3)','(1,2,3)','(1,3,2)','(1,3)'] # for sym3_xy
    idx_name = [idx_name_all[i] for i in which_tensor]

    t0 = model.hparams.record_wg_hist

    save_name_extra = kwargs.pop('save_name_extra', None) or ('factors' + '_' + loss_type if loss_type is not None else 'factors_native')
    log_dir = None # model.hparams.log_dir

    if animate:
        if show_netWeight:
            # hist_all, train_data = ABC_hist_, None
            netW_hist = torch.stack(getattr(model,'netW_hist')).detach().cpu()
            netW_hist, train_data, netW_idx_name = process_train_data_and_netW_hist(netW_hist, datamodule, idx_order='cab')  #='abc')

            ABC_hist_ = ABC_hist_[:,which_tensor]
            hist_all = torch.cat((netW_hist.unsqueeze(1), ABC_hist_), dim=1)
            # hist_all = [netW_hist, ABC_hist_]
            idx_name = netW_idx_name + idx_name
        else:
            train_data = None
            hist_all = ABC_hist_

    # ABC_hist_: t x [a,b,c] x 3Tensor
        ani = plot_tensor_slices_animation(hist_all, idx_name_rows=None, idx_name_cols= idx_name, t0=t0, train_data=train_data, log_dir = log_dir, save_name_extra=save_name_extra, **kwargs)
        return ani #, hist_all, V
    else:
        for which in which_tensor:
            A_hist = ABC_hist_[:,which].permute(1,0,2,3)        # A_hist = ABC_hist_[:,which].permute(1,0,3,2)
            plot_tensor_slices2(A_hist, modify_dim=-1, idx_name_rows=idx_name_all[which] , idx_name_cols=['t='], t0=t0, train_data=None, log_dir = log_dir, save_name_extra = save_name_extra, **kwargs)
        return ABC_hist_, V

def show_eig_hist(model, which_tensor=[0,1,2], factor_list = None, t_ref=None, skip=1, panel_dim = 'elements', animate_flag=False, log_dir = None, save_fig=False, save_name=None, save_name_extra='', **kwargs):
    if hasattr(model, 'trainer'):
        print(model.hparams.log_dir)

    factor_list = factor_list or model.model.factor_list  if t_ref is None  else model.w_hist[t_ref].detach().cpu()  #reference_step]

    ABC_hist = torch.stack(model.w_hist,dim=0).cpu()
    ABC_hist_normalized = torch.stack(normalize_factor_list(ABC_hist, ABC0 = None, normalize_0=True, normalize_1=True), dim=0)
    ABC_hist_ev, V_all = eig_diagonalize(ABC_hist_normalized) #, fit_index=None)
    # idx_name = ['A','B','C']         # for cyclic group (addition)        # idx_name = ['e','(1,2)','(2,3)','(1,2,3)','(1,3,2)','(1,3)'] # for sym3_xy

    # import pdb; pdb.set_trace()
    # ABC_hist_ = ABC_hist_[:,which_tensor]
    # idx_name = [idx_name[i] for i in which_tensor]

    t0 = model.hparams.record_wg_hist

    A_hist_ev = ABC_hist_ev[0] # plot just A
    if panel_dim == 'elements':
        idx_name_cols=['A']
        pass
    elif panel_dim == 'freq':
        idx_name_cols=['n=']
        A_hist_ev = A_hist_ev.permute(0,2,1)

    l = A_hist_ev.shape[1]
    fig, axes = plt.subplots(1,l, figsize=(18,3))
    fig.tight_layout()      # fig.tight_layout(h_pad=0.5, w_pad=0.5)
    title_cols, title_rows = get_titles_list(fig_size=(1,l),  idx_name_rows=None, idx_name_cols=idx_name_cols, skip=skip, t0=t0)

    lines = []
    t=0

    for i, ax in enumerate(axes):
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

        if title_cols is not None:
            title = get_local_title(title_cols,0,i, axes.shape[0])
            ax.set_title(title, y=1.0, pad=-14)

        if not animate_flag:
            ax.set_prop_cycle('color',plt.cm.Spectral(np.linspace(0,1,len(A_hist_ev))))
            ax.plot(A_hist_ev[:,i].real.T, A_hist_ev[:,i].imag.T, '.:')
        else:
            line = ax.plot(A_hist_ev[t,i].real.T, A_hist_ev[t,i].imag.T, '.:')
            lines.append(line[0])

    if not animate_flag:
        plt.show()
        return A_hist_ev, V_all

    else:
        t_list = [(skip*t+0)*t0 for t in range(len(A_hist_ev))]
        titles = ['t = {0:03d}'.format(t) for t in t_list]

        def animate(t):
            fig.suptitle(titles[t], fontsize=12) #, y=0.0) #, pad=14)
            for i, line in enumerate(lines):
                    line.set_xdata(A_hist_ev[t,i].real.T)
                    line.set_ydata(A_hist_ev[t,i].imag.T)

        ani = animation.FuncAnimation(fig, animate, frames=len(A_hist_ev), interval=150, blit=False, repeat_delay=3000)
        plt.close()

        add_str = panel_dim + '_' + save_name_extra
        log_dir = model.hparams.log_dir
        save_name = get_default_save_name(log_dir, add_str, save_fig, save_name)
        if save_name is not None:
            ani.save(f'{save_name}.mp4',
                    savefig_kwargs={"transparent": True}, #, "facecolor": "none"},
                    )

        return ani


def eig_diagonalize(ABC_hist):

    ABC_hist_ev = []
    V_all = []
    for A_hist in ABC_hist:
        A_last = A_hist[-1]

        ev, eV = torch.linalg.eig(A_last[1])
        ## Sort first by eigenvalue angles:
        theta1 = ev.log().imag
        _, sort_idx1 = ((theta1) % (2*np.pi)).sort()
        eV = eV[:,sort_idx1]

        A_hist_ev = diagonalize(A_hist, eV, loss_type=None)
        A_hist_ev = A_hist_ev.diagonal(dim1=-1, dim2=-2)

        ABC_hist_ev.append(A_hist_ev)
        V_all.append(eV)

    return ABC_hist_ev, V_all

def get_fig_size(T_shape):
    if len(T_shape)==3 :
        fig_size = 1, T_shape[0]
    elif len(T_shape)==4:
        fig_size = T_shape[:2]
    else:
        raise ValueError
    return fig_size

def get_titles_list(fig_size, idx_name_rows=None, idx_name_cols=None, skip=1, t0=1, t_init=0):
    def helper(idx_names, L):
        # idx_names = idx_name if isinstance(idx_name, (tuple,list)) else [idx_name]
        # if isinstance(idx_name, (tuple,list)):
        #     return [idx_name]
        # else:
        #
        if idx_names is None:
            titles = None
            titles_list = [titles]
        else:
            titles_list = []
            for idx_name in idx_names:
                assert isinstance(idx_name, str)
                # if idx_name == 't':   # time index
                if idx_name.endswith('='):   # time index
                    if idx_name.startswith('t'):
                        # t_init_ = t_init or skip
                        # t_list = [0] + [(skip*j+t_init_) for j in range(L-1)]
                        t_list =  [(skip*j+0) for j in range(L)]
                        titles = [f't = {t*t0}' for t in t_list]
                        # titles = [f't = {(skip*j+0)*t0}' for j in range(L)]                   # titles = [f'{idx_name[:-1]} = {(skip*j+0)*t0}' for j in range(L)]
                    else:
                        titles = [f'${idx_name[:-1]} = {i}$' for i in range(L)]
                else:
                    titles = [f'${{{idx_name}}}_{{{i}}}$' for i in range(L)]  # titles = [f'{idx_name} = {i}' for i in range(L)]
                titles_list.append(titles)
        return titles_list

    N,M = fig_size
    title_cols = helper(idx_name_cols, M)
    title_rows = helper(idx_name_rows, N)
    return title_cols, title_rows

def get_local_title(title_list, i, j, n):
    def get_title(title_list, j):
        # if title_list is None:
        #     title = ''
        if isinstance(title_list, (tuple,list)):
            title = f'{title_list[j]}'
        else:
            title = f'{title_list} = {j}'
        return title

    # import pdb; pdb.set_trace()
    if len(title_list)==1:
    # assert len(title_list)==1
        title_list = title_list[0]
    elif isinstance(title_list[0], (tuple,list)):
        temp = []
        for titles in title_list:
            temp += titles
            title_list = temp

    if title_list is None:
        title = ''
    elif len(title_list) == n:
        if i==0:
            title = get_title(title_list, j)
        else:
            title = ''
    else:
        j_ = j + n*i
        title = get_title(title_list, j_)
    # else:
    #     title = get_title(title_list[i], j)
    return title


def pre_process_tensor(T, show_num=[0], skip=1, skip_dim=1, t_init=0, z_pow=1, softmax_dim=None):
    if isinstance(T, (list,tuple)):
        T = torch.stack(T)

    if softmax_dim is not None:
        T = torch.nn.Softmax(dim=softmax_dim)(T)

    if not isinstance(show_num, (list,tuple)):
        show_num = [show_num]

    if len(show_num)==1:
        if show_num[0]!=0:
            T = T[:show_num[0]]
    else:
        T = T[:show_num[0],:show_num[1]]

    if skip>1:
        if skip_dim==0:
            # T = torch.cat((T[0:1],T[t_init::skip]), dim=0)  # T[1::skip]
            T = T[0::skip]
        else:
            T = T[:,0::skip]

    if z_pow != 1:
        T = T.abs().pow(z_pow) * T.sign()

    return T.detach()

def reshape_T(T, modify_dim, M_lim):
    def helper(T):
        fig_size  = (M//M_lim, M_lim)
        if len(T.shape)==5:
            new_shape = T.shape[:modify_dim] + fig_size + T.shape[modify_dim+2:]
        elif len(T.shape)==4:
            new_shape = T.shape[:modify_dim] + fig_size + T.shape[modify_dim+1:]
        elif len(T.shape)==3:
            new_shape = fig_size + T.shape[1:]
        else:
            print(T.shape, fig_size)
            raise ValueError
        T = T.view(new_shape)
        return T, fig_size

    if modify_dim<0:
        fig_size_orig = get_fig_size(T.shape[0:])
        fig_size = fig_size_orig
        # print(T.shape)
        # import pdb; pdb.set_trace()
        wrap_around_flag = False
        return T, fig_size_orig, fig_size, wrap_around_flag
    else:
        fig_size_orig = get_fig_size(T.shape[modify_dim:])
        N, M = fig_size_orig
        wrap_around_flag = M_lim is not None and (N==1 and M>M_lim)

        if wrap_around_flag:
            T, fig_size = helper(T)
        else:
            fig_size = fig_size_orig
        return T, fig_size_orig, fig_size, wrap_around_flag

def plot_tensor_slices2(T, fig = None, axes = None, ims = None, idx_name_rows=None, idx_name_cols=None, skip=1, t0=0, t_init=0, train_data=None, abc_dim=0, z_pow=1, show_num=[0], log_dir = None, save_fig=False, save_name=None, save_name_extra='', softmax_dim=None, modify_dim=0, M_lim = None, transpose=False, **kwargs): # , color_fix=True
    T = pre_process_tensor(T, show_num=show_num, skip=skip, skip_dim=1, t_init=t_init, z_pow=z_pow, softmax_dim=softmax_dim)    # time dimension = 1 (skip_dim)
    T, fig_size_orig, fig_size, wrap_around_flag = reshape_T(T, modify_dim, M_lim)    # fig_size_orig = get_fig_size(T.shape)
    title_cols, title_rows = get_titles_list(fig_size_orig,  idx_name_rows=idx_name_rows, idx_name_cols=idx_name_cols, skip=skip, t0=t0, t_init=t_init)

    fig, axes, ims, T = init_axes(fig_size, ims,  T, train_data, abc_dim=abc_dim, unsqueeze_dim=0, title_cols=title_cols, title_rows=title_rows)

    for i, (T_, row) in enumerate(zip(T, axes)):
        for j, (a_, ax) in enumerate(zip(T_, row)):
            if transpose:
                a_ = a_.T
            ims[i,j].set_data(a_)  #.set_data(a_[1:4,4:7])

    plt.show()

    save_fig = True if save_name is not None else save_fig
    add_str = save_name_extra
    save_name = get_default_save_name(log_dir, add_str, save_fig, save_name)
    if save_name is not None:
        fig.savefig(f'{save_name}.pdf')

    # return fig, axes, ims

def init_axes(fig_size, ims, T, abc, abc_dim=0, unsqueeze_dim=0, title_cols=None, title_rows=None, cmap='bwr', suptitle=None):
    # cmap='seismic' 'viridis' 'PRGn' 'coolwarm'

    N, M = fig_size
    fig, axes = plt.subplots(N, M, figsize=(1.5*M,1.5*N), sharex=True, sharey=True, constrained_layout=True)  #figsize=(8*num_axes,5))
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    if N==1: # and not wrap_around_flag:
        axes = axes[np.newaxis,...]        # axes = [axes]
        T = T.unsqueeze(dim=unsqueeze_dim)                 #  T = [T]
    # T_max = T.abs().max()
    # v_range = [-T_max, T_max]     # v_range = [T.min(), T.max()]
    v_range = [-1, 1]

    zeros = np.zeros(T.shape[-2:])
    ims = ims or np.empty(axes.shape, dtype=object)

    train_marker = '*' #'o'   if T.shape[-1]>10    else '*'
    markersize   = 3   if T.shape[-1]>7    else 5
    alpha_not = 0.00  if T.shape[-1]>7    else 0.04 #0.07

    if abc is not None:
        if abc_dim==1:
            a_max_flag = abc[:,0].max() > axes.shape[abc_dim]
        else:
            a_max_flag = False #True

    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            assert ims[i,j] == None
            im = ax.imshow(zeros, vmin=v_range[0], vmax=v_range[1], cmap=cmap)
            ims[i,j]=im

            if abc is not None:
                if a_max_flag:
                    i_ = i + j*axes.shape[0] if abc_dim==0 else j + i*axes.shape[1]
                else:
                    i_ = i if abc_dim==0 else j

                idx = abc[:,0] == i_
                idx_not = abc[:,0] != i_

                if a_max_flag or i==0 or abc_dim==0:                 # if abc_dim==0 or i==0:
                    ax.plot(abc[idx,1],abc[idx,2], train_marker, color='k', markersize=markersize, alpha=0.9, fillstyle='none') #, edgecolors='r')
                    ax.plot(abc[idx_not,1],abc[idx_not,2], 'o', color='k', markersize=markersize, alpha=alpha_not, fillstyle='none') #, edgecolors='r')

            # ax.set_axis_off()
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            if title_rows is not None:
                ylabel = get_local_title(title_rows,j,i, axes.shape[0])
                ax.set_ylabel(ylabel+'    ', rotation=0, size='large')

            if title_cols is not None:
                title = get_local_title(title_cols,i,j, axes.shape[1])
                ax.set_title(title)

    # cbar = fig.colorbar(ims[-1,-1], shrink=0.8)    # fig.tight_layout()    # fig.tight_layout(h_pad=0, w_pad=0)

    return fig, axes, ims, T

import matplotlib.animation as animation

def plot_tensor_slices_animation(T, idx_name_rows=None, idx_name_cols=None, skip=1, t0=0, train_data=None, z_pow=1, show_num=[0], log_dir = None, save_fig=False, save_name=None, save_name_extra='', softmax_dim=None, M_lim = None, transpose=False, **kwargs): # , color_fix=True
    T = pre_process_tensor(T, skip=skip, skip_dim=0, z_pow=z_pow, softmax_dim=softmax_dim)     # time dimension = 0 (skip_dim)
    T, fig_size_orig, fig_size, wrap_around_flag = reshape_T(T, modify_dim=1, M_lim = M_lim)    # fig_size_orig = get_fig_size(T.shape[1:])
    title_cols, title_rows = get_titles_list(fig_size_orig,  idx_name_rows=idx_name_rows, idx_name_cols=idx_name_cols, skip=skip, t0=t0)

    fig, axes, ims, T = init_axes(fig_size, None,  T, train_data, abc_dim=1, unsqueeze_dim=1, title_cols=title_cols, title_rows=title_rows)
    t_list = [(skip*t+0)*t0 for t in range(len(T))]
    titles = ['t = {0:03d}'.format(t) for t in t_list]
    # titles = [f't = {(skip*t+0)*t0}' for t in range(len(T))]


    def animate(t):
        fig.suptitle(titles[t], fontsize=12)

        for i, row in enumerate(axes):
            for j, ax in enumerate(row):
                (i_, j_) = (1, M_lim*i+j)  if wrap_around_flag else i, j
                image = T[t,i_,j_]
                if transpose:
                    image = image.T
                ims[i,j].set_data(image)

                # ax.set_title(title)
        # return ims

    ani = animation.FuncAnimation(fig, animate, frames=len(T), interval=150, blit=False, repeat_delay=3000)
    # ani = animation.ArtistAnimation(fig, ims_all, interval=50, blit=True, repeat_delay=1000)
    plt.close()

    add_str = save_name_extra
    save_name = get_default_save_name(log_dir, add_str, save_fig, save_name)
    if save_name is not None:
        ani.save(f'{save_name}.mp4')

    return ani

# def getImageFromList(x):
#     return imageList[x]

# # fig = plt.figure(figsize=(10, 10))
# fig, axes = plt.subplots(2,3)
# ax = axes[0][0]
# ax2 = axes[1][1]
# ims = []
# ims2 = []
# for i in range(len(imageList)):
#     im = ax.imshow(getImageFromList(i), animated=True)
#     im2 = ax2.imshow(getImageFromList(i), animated=True)
#     ims.append([im,im2])

# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
# plt.close()
# ani.save('test.mp4')
# # Show the animation
# # HTML(ani.to_html5_video())
# HTML(ani.to_jshtml())



def reorder_tensor(A_, idx_sort):
    return A_[...,idx_sort,:][...,:,idx_sort]

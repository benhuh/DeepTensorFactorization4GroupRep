#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib import cm, rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
sns.set_theme()
sns.set_context('paper')
sns.set(font_scale=1.4)
import numpy as np

from dtf.train_helper_script import run_exp, plot_all
from dtf.visualization_new import plot_tensor_slices2, show_netWeight_hist, show_sparse_hist, get_regular_representation, get_regular_representation_XYZ, normalize_factor_list, reorder_tensor, plot_netWeight_with_train_data
from dtf.visualization_new import optimize_T
from dtf.training import get_model_pkg

def plot_heatmaps(trained, desired, opt_V):
    """
    Plots heatmaps of the trained and desired products between the product structure
    tensor and the convolution weights, as well as the orthogonal matrix that best
    aligns the product tensor and the generating data.

    Parameters
    ----------
    trained: PyTorch tensor of size (D, D, D) (or (D, D))
        Learned products.
    desired: PyTorch tensor of size (D, D, D) (or (D, D))
        Desired products.
    opt_V: PyTorch tensor of size (D, D)
        Orthogonal matrix that best aligns the product tensor and the generating data.

    Returns
    -------
    fig: Matplotlib figure
        The figure containing the heatmaps.
    """

    # make subplots
    base_w = 1
    base_h = 1
    if len(trained.shape) == 2:
        fig = plt.figure(figsize=(int(6 * base_w), int(2 * base_h))) # 1 for annot, 1 for trained, 2 for opt_V, 2 for opt_V.T @ opt_V
        gs = gridspec.GridSpec(int(2 * base_h), int(6 * base_w))
        p = 1
    else:
        fig = plt.figure(figsize=(int(11 * base_w), int(2 * base_h))) # 1 for annot, 6 for trained, 2 for opt_V, 2 for opt_V.T @ opt_V
        gs = gridspec.GridSpec(int(2 * base_h), int(11 * base_w))
        p = 6

    # color maps
    cmap = sns.cubehelix_palette(reverse=True, rot=-0.2, as_cmap=True)
    cmap_r = sns.cubehelix_palette(reverse=True, start=0,rot=0.2, as_cmap=True)
    cmap_y = sns.cubehelix_palette(reverse=True, start=0, rot=0.6, as_cmap=True)

    # side labels
    ax = plt.subplot(gs[0, 0])
    ax.annotate('Desired', xy=(0.5, 0.5), xytext=(0.5, 0.5), textcoords='axes fraction', ha='center', va='center', fontsize=14, fontweight="bold")
    ax.grid(False)
    ax.set_facecolor('white')
    #plt.axis("off")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax = plt.subplot(gs[1, 0])
    ax.annotate('Trained', xy=(0.5, 0.5), xytext=(0.5, 0.5), textcoords='axes fraction', ha='center', va='center', fontsize=14, fontweight="bold")
    ax.grid(False)
    ax.set_facecolor('white')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for idx in range(1, p + 1):
        ax = plt.subplot(gs[0, idx])
        desired_norm = desired[:, idx - 1, :].detach().numpy()
        sns.heatmap(desired_norm, ax=ax, cmap=cmap, cbar=False, square=True, xticklabels=False, yticklabels=False)

        ax = plt.subplot(gs[1, idx])
        trained_norm = trained[:, idx - 1, :].detach().numpy()
        sns.heatmap(trained_norm, ax=ax, cmap=cmap_y, cbar=False, square=True, xticklabels=False, yticklabels=False)

    # orthogonal plot
    ax = plt.subplot(gs[:, p+1:p+3])
    ax.set_title(r'$\boldsymbol{V}$ matrix', fontweight="bold", fontsize=14)
    opt_V_norm = opt_V.detach().numpy()
    sns.heatmap(opt_V_norm, ax=ax, cmap=cmap_r, cbar=False, square=True, xticklabels=False, yticklabels=False)

    ax = plt.subplot(gs[:, p+3:])
    ax.set_title(r'$\boldsymbol{V}^T\boldsymbol{V}$ (Orthogonal?)', fontweight="bold", fontsize=14)
    opt_O_norm = (opt_V.T @ opt_V).detach().numpy()
    sns.heatmap(opt_O_norm, ax=ax, cmap=cmap_r, cbar=False, square=True, xticklabels=False, yticklabels=False)

    return fig


def check_model(model, datamodule):
    """
    Checks if the learned convolution weight matches the one used for training
    (in terms of products with the tensor, has basis ambiguity).

    Parameters
    ----------
    model: PyTorch model
        The model to be checked.
    datamodule: PyTorch datamodule
        The datamodule used for training the model.

    Returns
    -------
    trained: PyTorch tensor
        The convolution weight recovered by training.
    desired: PyTorch tensor
        The true convolution weight.
    """

    if len(model.model.net_Weight.shape) == 2:
        trained = torch.einsum('ijk,j->ik', model.model.net_Weight, model.model.conv_weight)
        w_data = torch.Tensor(np.arange(6) / 100)
        desired = torch.einsum('ijk,j->ik', datamodule.train_dataset.M.to_dense() + 0.0, w_data)
    elif len(model.model.net_Weight.shape) == 3:
        trained = torch.einsum('ijk,jc->ick', model.model.net_Weight, model.model.conv_weight)
        torch.manual_seed(2)
        w_data = w = torch.randn(6, 6) / np.sqrt(6)
        desired = torch.einsum('ijk,jc->ick', datamodule.train_dataset.M.to_dense() + 0.0, w_data)
    else:
        raise ValueError('net_Weight has an unexpected shape')

    return trained, desired

seed = 2
train_frac = 61
task_name = 'binary/sym3_xy_vec'

checkpoint = r'lightning_logs/test/mse_loss/binary/sym3_xy_vec_6/FC/SGD momentum=0.5 customL2/lr=0.005 wd=0.1/seed2/frac=61.0/version_3/checkpoints/epoch=7998-step=4000.ckpt'
model, datamodule, trainer = get_model_pkg(checkpoint)

trained, desired = check_model(model, datamodule)

V = torch.eye(model.model.net_Weight.shape[-1])#.unsqueeze(0).repeat(model.model.net_Weight.shape[0],1,1)
model_weight = model.model.net_Weight.detach()

# first, rotate M and try to recover
gaus = torch.randn(6, 6)
svd = torch.linalg.svd(gaus)
orth = svd[0] @ svd[2]
print(orth.T @ orth) # sanity check
train_M = datamodule.train_dataset.M.to_dense() + 0.0
rot_train_M = torch.einsum('ijk,jl->ilk', train_M, orth)
opt_V, opt_T, losses = optimize_T(rot_train_M / (rot_train_M.norm()) * train_M.norm(), V, train_M, lr=1e-2, reg_coeff=0.1, loss_type='regular', steps=1000)
print((opt_V - orth).norm() ** 2) # works

# we should also be able to do this with linear algebra
opt_V_e = optimize_T(rot_train_M / (rot_train_M.norm()) * train_M.norm(), V, train_M, lr=1e-2, reg_coeff=0.1, loss_type='exact', steps=1000)
print((opt_V_e - orth).norm() ** 2) # doesn't work

fig = plot_heatmaps(trained, desired, opt_V)
plt.show()
plt.close()
# import pdb; pdb.set_trace()
# XYZ = generate_figures(model, datamodule, save_name, skip=15, t_init=0, show_steps=5, plot_all_weights=True, ABC_or_A = 'ABC')
#
# T = T.permute(0, 2, 1).view(-1, T.shape[-1])

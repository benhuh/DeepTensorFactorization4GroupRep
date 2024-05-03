#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib import cm, rc

rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")
sns.set_theme()
sns.set_context("paper")
# sns.set(font_scale=1.4)
import numpy as np

from dtf.train_helper_script import run_exp, plot_all
from dtf.visualization_new import (
    plot_tensor_slices2,
    show_netWeight_hist,
    show_sparse_hist,
    get_regular_representation,
    get_regular_representation_XYZ,
    normalize_factor_list,
    reorder_tensor,
    plot_netWeight_with_train_data,
)
from dtf.visualization_new import (
    optimize_T,
    plot_heatmaps,
    plot_heatmaps_h,
    check_model,
)
from dtf.training import get_model_pkg

seed = 2
train_frac = 61
task_name = "binary/sym3_xy_vec"

# checkpoint = r'lightning_logs/test/mse_loss/binary/sym3_xy_vec_6/FC/Adam momentum=0.5 customL2/lr=0.001 wd=0.1/seed2/frac=61.0/version_19/checkpoints/epoch=7998-step=4000.ckpt'
checkpoint = r"lightning_logs/test/mse_loss/layer/FC_vec_6/FC/Adam momentum=0.5 customL2/lr=0.01 wd=0.1/seed2/frac=61.0/version_0/checkpoints/epoch=1238-step=620.ckpt"
model, datamodule, trainer = get_model_pkg(checkpoint)

# trained, desired = check_model(model, datamodule)

model_weight = model.model.net_Weight.detach()
conv_weight = model.model.conv_weight.detach()
train_M = datamodule.train_dataset.M.to_dense() + 0.0

V = torch.eye(model.model.net_Weight.shape[1])
opt_V, opt_T, losses = optimize_T(
    model_weight / (model_weight.norm()) * train_M.norm(),
    V,
    train_M,
    lr=1e-1,  # conv: 1e-2, fc: 1e-1
    reg_coeff=2,  # conv: 1.0, fc: 0.5
    loss_type="sparse_inv",
    steps=1000,  # conv: 1000, fc: 10000
)

# original
fig = plot_heatmaps_h(model_weight, train_M)
plt.show()
plt.close()

# optimized
fig = plot_heatmaps_h(opt_T, train_M, opt_V)
plt.show()
plt.close()

# plot the singular values
unfolded_B = model_weight.permute(0, 2, 1)
unfolded_B = unfolded_B.reshape(-1, model_weight.shape[1])
u, s, vh = torch.linalg.svd(unfolded_B)
plt.scatter(range(len(s)), s.cpu().numpy())
plt.ylabel("Singular Value")
plt.xlabel("Index")
plt.title("Singular Values of the learnt tensor")
plt.show()
plt.close()

u, s, vh = torch.linalg.svd(conv_weight)
plt.scatter(range(len(s)), s.cpu().numpy())
plt.ylabel("Singular Value")
plt.xlabel("Index")
plt.title("Singular Values of the learnt weights")
plt.show()
plt.close()

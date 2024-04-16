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
# sns.set(font_scale=1.4)
import numpy as np

from dtf.train_helper_script import run_exp, plot_all
from dtf.visualization_new import plot_tensor_slices2, show_netWeight_hist, show_sparse_hist, get_regular_representation, get_regular_representation_XYZ, normalize_factor_list, reorder_tensor, plot_netWeight_with_train_data
from dtf.visualization_new import optimize_T, plot_heatmaps, check_model
from dtf.training import get_model_pkg

seed = 2
train_frac = 61
task_name = 'binary/sym3_xy_vec'

checkpoint = r'lightning_logs/test/mse_loss/binary/sym3_xy_vec_6/FC/Adam momentum=0.5 customL2/lr=0.001 wd=0.1/seed2/frac=61.0/version_14/checkpoints/epoch=7998-step=4000.ckpt'
model, datamodule, trainer = get_model_pkg(checkpoint)

trained, desired = check_model(model, datamodule)

model_weight = model.model.net_Weight.detach()
train_M = datamodule.train_dataset.M.to_dense() + 0.0

V = torch.eye(model.model.net_Weight.shape[-1])
opt_V_e, opt_T_e, loss_e = optimize_T(model_weight / (model_weight.norm()) * train_M.norm(), V, train_M, lr=1e-2, reg_coeff=1, loss_type='sparse')

V = torch.eye(model.model.net_Weight.shape[-1])
opt_V, opt_T, losses = optimize_T(model_weight / (model_weight.norm()) * train_M.norm(), V, train_M, lr=1e-2, reg_coeff=0.0, loss_type='sparse_inv')

# original
fig = plot_heatmaps(model_weight, train_M)
plt.show()
plt.close()

# plot T and M
fig = plot_heatmaps(opt_T_e, train_M, opt_V_e)
plt.show()
plt.close()

fig = plot_heatmaps(opt_T, train_M, opt_V)
plt.show()
plt.close()

# import pdb; pdb.set_trace()
# XYZ = generate_figures(model, datamodule, save_name, skip=15, t_init=0, show_steps=5, plot_all_weights=True, ABC_or_A = 'ABC')
#
# T = T.permute(0, 2, 1).view(-1, T.shape[-1])

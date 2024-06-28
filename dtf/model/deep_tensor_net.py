#!/usr/bin/env python
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dtf.tensor_operations import (
    tensor_prod,
    compute_svd_all,
    compute_imbalance2,
    compute_Custom_L2_all,
    compute_Manual_L2_all,
    get_einsum_str,
    get_tensor_size,
)


class Tensor_Layer(nn.Module):
    def __init__(self, siz, init_scale=1):  # , random_init=True):  # , permute_dim=None
        super().__init__()
        assert len(siz) == 3
        # import pdb; pdb.set_trace()
        self.base = (
            init_scale * torch.randn(*siz) / math.sqrt(siz[0])
        )  # self.base *= math.sqrt(siz[0])
        self.base = nn.Parameter(self.base)

    @property
    def tensor(self):
        return self.base


class Tensor_Layer2d(nn.Module):
    def __init__(self, siz, init_scale=1):  # , random_init=True):  # , permute_dim=None
        super().__init__()
        assert len(siz) == 3
        # import pdb; pdb.set_trace()
        self.base_x = (
            init_scale * torch.randn(*siz) / math.sqrt(siz[0])
        )  # self.base *= math.sqrt(siz[0])
        self.base_x = nn.Parameter(self.base_x)

        self.base_y = (
            init_scale * torch.randn(*siz) / math.sqrt(siz[0])
        )  # self.base *= math.sqrt(siz[0])
        self.base_y = nn.Parameter(self.base_y)

    @property
    def tensor_x(self):
        return self.base_x

    @property
    def tensor_y(self):
        return self.base_y


def get_W_shape_cum(W):
    w_shape = list(W.shape[1:]) + [1]
    w_shape.reverse()
    p = 1
    cum_shape = [p := p * n for n in w_shape]
    cum_shape.reverse()
    return torch.Tensor(cum_shape)


class Base_Model(nn.Module):
    def __init__(self, *args, **kwargs):  #  loss_fn='mse_loss'):
        super().__init__()
        self.loss_fn = kwargs.get("loss_fn", "mse_loss")
        self.layer_type = kwargs.get("layer_type", None)

    def get_label(self, out, y):
        if self.loss_fn == "cross_entropy":
            assert y.dtype in [torch.int64]
            if len(y.shape) == len(out.shape):
                y = y.squeeze()
            label = y

        elif self.loss_fn in ["mse_loss"]:
            if y.dtype in [torch.int64]:
                if len(y.shape) == len(out.shape):
                    y = y.squeeze()
                label = F.one_hot(y, num_classes=out.shape[1]).float()
            elif y.dtype in [
                torch.float,
                torch.bool,
                torch.complex64,
            ]:  # , torch.float64]:
                if len(y.shape) == len(out.shape) + 1:
                    y = y.squeeze()
                label = y.to(out.dtype)
            else:
                raise ValueError
        return label

    def evaluate(self, data, *args, **kwargs):
        x, y = data
        out, other_outputs = self.forward(x, *args, **kwargs)
        label = self.get_label(out, y)

        if self.loss_fn == "cross_entropy":
            loss = F.cross_entropy(out, label, reduction="sum")
        elif self.loss_fn in ["mse_loss", "Lagrange"]:
            loss = F.mse_loss(out, label, reduction="sum")  # / 2
        elif self.loss_fn == "mse_loss_complex":
            loss = ((out - label).abs() ** 2).sum()  # / 2
        else:
            raise ValueError(f"No loss function named {self.loss_fn}")

        acc = self.get_accuracy(out, y)
        return loss, acc, out, (other_outputs, x, out, label)

    def get_accuracy(self, out, y):
        if len(y.shape) == len(out.shape) - 1:
            with torch.no_grad():
                acc = self._accuracy(out, y)
                acc = acc.mean()
        else:
            acc = None  # torch.zeros(1)
        return acc

    def _accuracy(self, y_hat: Tensor, y: Tensor) -> Tensor:
        row_accuracy = y_hat.argmax(1).eq(y)
        return row_accuracy.float() * 100  # shape: batchsize

class HyperCube(nn.Module):
    def __init__(self, N, r, init_scale, decomposition_type, layer_type):  #  loss_fn='mse_loss'):
        super().__init__()
        self.initialize_weights(N, r, init_scale, decomposition_type, layer_type)

    def initialize_weights(self, N, r, init_scale, decomposition_type, layer_type):
        r = r or (min(N) if isinstance(N,(tuple,list)) else N)
        einsum_str, shared_idx0 = get_einsum_str(decomposition_type)
        tensor_size_list, idx_appearance_dict, input_str_list, ext_indices, _ = get_tensor_size(N, r, einsum_str)
        if layer_type == 'FC': # Hack for now
            tensor_size_list[0] = [N**2, N, N]

        # Should have a trainable convolution filter here
        # self.filter = nn.Parameter(init_scale * torch.randn(3, 3) / math.sqrt(2))
        # (I actually think the filter is incorporate into W)
        #
        # Separate class: subclass of Deep_Tensor_Net(_w)

        self.einsum_str = einsum_str
        self.idx_appearance_dict = idx_appearance_dict
        self.input_str_list = input_str_list
        self.ext_indices = ext_indices
        self.decomposition_type=decomposition_type

        self.layers = self.initialize_Tensors(tensor_size_list, init_scale) #, random_init)

        self.N = N
        W = self.net_Weight
        self.W_shape_cum = get_W_shape_cum(W)
        # self.store_init_weights()

    def initialize_Tensors(self, tensor_size_list, init_scale0=1):
        tensor_depth = len(tensor_size_list)
        init_scale = init_scale0**(1/tensor_depth)

        layers = []
        for siz in tensor_size_list:
            layer =  Tensor_Layer(siz, init_scale=init_scale)
            layers.append(layer)

        return nn.ModuleList(layers)

    @property
    def T_param_list(self):
        return [layer.tensor for layer in self.layers]

    @property
    def T_list_no_grad(self):
        return [T.detach() for T in self.factor_list]

    @property
    def factor_list(self):
        return self.T_param_list

    @property
    def net_Weight(self):
        self._net_Weight = tensor_prod(self.factor_list, self.einsum_str)
        return self._net_Weight #.permute(2,0,1)

    def get_svd(self):
        return compute_svd_all(self.factor_list, self.idx_appearance_dict)

    def get_imbalance2(self):
        return compute_imbalance2(self.T_list_no_grad)

    def manual_L2_loss(self):
        return compute_Manual_L2_all(self.factor_list)

    def Custom_L2_loss(self): #mix_coeff=0.00):
        loss = compute_Custom_L2_all(self.factor_list, self.idx_appearance_dict)
        return loss


class Deep_Tensor_Net(Base_Model):
    def __init__(self, *args, **kwargs):  # init_val = 1) -> None:
        super().__init__(*args, **kwargs)
        # self.initialize_weights(*args, **kwargs)
        N, r = kwargs.get("N", None), kwargs.get("r", None)
        init_scale = kwargs.get("init_scale", 1)
        layer_type = kwargs.get("layer_type", None)
        decomposition_type = kwargs.get("decomposition_type", 'FC')

        self.hypercubes = nn.ModuleList([HyperCube(N, r, init_scale, decomposition_type, layer_type)])

    def evaluate(self, data, *args, **kwargs):
        loss, acc, out, (other_outputs, x, out, label) = super().evaluate(
            data, *args, **kwargs
        )
        return loss, acc, out, other_outputs

    # def initialize_weights(self, *args, **kwargs):
    #     N, r, decomposition_type = kwargs.get("N", None), kwargs.get("r", None), "FC"
    #     init_scale = kwargs.get("init_scale", 1)
    #     r = r or (min(N) if isinstance(N, (tuple, list)) else N)
    #     einsum_str, shared_idx0 = get_einsum_str(decomposition_type)
    #     tensor_size_list, idx_appearance_dict, input_str_list, ext_indices, _ = (
    #         get_tensor_size(N, r, einsum_str)
    #     )

    #     if self.layer_type == "FC":
    #         tensor_size_list[0] = [N**2, N, N]

    #     self.einsum_str = einsum_str
    #     self.idx_appearance_dict = idx_appearance_dict
    #     self.input_str_list = input_str_list
    #     self.ext_indices = ext_indices
    #     self.decomposition_type = decomposition_type

    #     self.layers = self.initialize_Tensors(
    #         tensor_size_list, init_scale
    #     )  # , random_init)

    #     self.N = N
    #     W = self.net_Weight
    #     self.W_shape_cum = get_W_shape_cum(W)
    #     # self.store_init_weights()

    # def initialize_Tensors(self, tensor_size_list, init_scale0=1):
    #     tensor_depth = len(tensor_size_list)
    #     init_scale = init_scale0 ** (1 / tensor_depth)

    #     layers = []
    #     for siz in tensor_size_list:
    #         layer = Tensor_Layer(siz, init_scale=init_scale)
    #         layers.append(layer)

    #     return nn.ModuleList(layers)

    # @property
    # def T_param_list(self):
    #     return [layer.tensor for layer in self.layers]

    def forward(
        self, x: Tensor, W=None, *args, **kwargs
    ):  # -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        if W is None:
            if kwargs.get("train_or_test") == "train":
                W = self.net_Weight
            else:
                W = self.net_Weight  # existing weight
        out = self.read_from_Tensor(W, x)
        return out, (W,)

    def read_from_Tensor(self, W, x):  # Modify this to handle tuple of weight tensors
        if x.dtype == torch.int64:  # x is tensor of indices
            out = self.index_select(W,x)
        else:
            if isinstance(W, tuple):
                raise ValueError("Deep_Tensor_Net: read_from_Tensor: W is tuple of tensors. This should be used with Deep_Tensor_Net_conv2d.")
            else:
                x1,x2 = (tensor.squeeze(dim=1) for tensor in x.split(1,dim=1))
                out = torch.einsum('ijk,bi,bj->bk',W,x1,x2)
        return out

    def index_select(self, W, x):
        index_1d = (
            x.float() @ self.W_shape_cum[-x.shape[1] :].to(x.device)
        ).int()  #    index_1d = W.shape[-2] * idx1 + idx2
        # if len(W.shape) != 2:
        numel = math.prod(list(W.shape[-x.shape[1] :]))
        out = W.reshape(-1, numel).index_select(
            -1, index_1d
        )  # out = torch.index_select(W.view(-1, numel), -1, index_1d)
        return out.T

    @property
    def factor_list(self):
        return  (hypercube.factor_list for hypercube in self.hypercubes)
        # return self.T_param_list

    @property
    def T_list_no_grad(self):
        return (hypercube.T_list_no_grad for hypercube in self.hypercubes)
        # return [T.detach() for T in self.factor_list]

    @property
    def net_Weight(self): # returns a tuple of net_Weight tensors
        self._net_Weight = (hypercube.net_Weight for hypercube in self.hypercubes)
        # self._net_Weight = tensor_prod(self.factor_list, self.einsum_str)
        return self._net_Weight  # .permute(2,0,1)

    def get_svd(self):
        return torch.stack([hypercube.get_svd() for hypercube in self.hypercubes])
        # return compute_svd_all(self.factor_list, self.idx_appearance_dict)

    def get_imbalance2(self):
        return sum(hypercube.get_imbalance2() for hypercube in self.hypercubes)
        # return compute_imbalance2(self.T_list_no_grad)

    def manual_L2_loss(self):
        return sum(hypercube.manual_L2_loss() for hypercube in self.hypercubes)
        # return compute_Manual_L2_all(self.factor_list)

    def Custom_L2_loss(self):  # mix_coeff=0.00):
        return sum(hypercube.Custom_L2_loss() for hypercube in self.hypercubes)
        # loss = compute_Custom_L2_all(self.factor_list, self.idx_appearance_dict)
        # return loss


class Deep_Tensor_Net_conv(Deep_Tensor_Net):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.initialize_weights(*args, **kwargs)
        init_scale = kwargs.get("init_scale", 1)
        n_vectors = kwargs.get("n_vectors", 1)
        # convolution weights
        # self.conv_weight = nn.Parameter(init_scale * torch.randn(3, 3) / math.sqrt(2))
        if self.layer_type == "FC":
            self.conv_weight = nn.Parameter(
                init_scale * torch.randn(6**2, n_vectors)
            )  # should match data.py line 309
        else:
            self.conv_weight = nn.Parameter(init_scale * torch.randn(6, n_vectors))

    def read_from_Tensor(self, W, x):
        if x.dtype == torch.int64:  # x is tensor of indices
            out = self.index_select(W, x)
        else:
            # (this split is weird, data should have 3 tensors, x,y,z)
            # import pdb; pdb.set_trace()
            # x1,x2 = (tensor.squeeze(dim=1) for tensor in x.split(1,dim=1))
            # out = torch.einsum('ijk,bi,bj->bk',W,x1,x2)
            # here I should performing the convolution with the filter?
            # out = torch.einsum('ijk,bi,j->bk',W,x1,self.conv_weight)
            # out = torch.einsum('ijk,bi,j->bk',W,x,self.conv_weight)
            if isinstance(W, tuple):
                raise ValueError("Deep_Tensor_Net: read_from_Tensor: W is tuple of tensors. This should be used with Deep_Tensor_Net_conv2d.")
            else:
                print(W)
                print(W.shape, x.shape, self.conv_weight.shape)
                out = torch.einsum("ijk,bi,jc->bck", W, x, self.conv_weight)
        return out

    def normalize(self):
        # normalize by diving the total norm instead
        self.conv_weight.data = F.normalize(
            self.conv_weight.data, p=2, dim=0
        )  # conw_weight is 6x36
        # self.net_Weight.data = F.normalize(self.net_Weight.data, p=2, dim=1) # net_Weight is 6x6x6
        # we want to update the factor weights. Divide the factor weights by  Frob(net_Weight) ^ 1/3
        return self


class Deep_Tensor_Net_conv2d(Base_Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_weights(*args, **kwargs)
        init_scale = kwargs.get("init_scale", 1)
        n_vectors = kwargs.get("n_vectors", 1)
        # convolution weights
        # self.conv_weight = nn.Parameter(init_scale * torch.randn(3, 3) / math.sqrt(2))
        if self.layer_type == "FC":
            self.conv_weight = nn.Parameter(
                init_scale * torch.randn(6**2, n_vectors)
            )  # should match data.py line 309
        else:
            self.conv_weight = nn.Parameter(init_scale * torch.randn(6, n_vectors))

    def initialize_weights(self, *args, **kwargs):
        N, r, decomposition_type = kwargs.get("N", None), kwargs.get("r", None), "FC"
        init_scale = kwargs.get("init_scale", 1)
        r = r or (min(N) if isinstance(N, (tuple, list)) else N)
        einsum_str, shared_idx0 = get_einsum_str(decomposition_type)
        tensor_size_list, idx_appearance_dict, input_str_list, ext_indices, _ = (
            get_tensor_size(N, r, einsum_str)
        )

        if self.layer_type == "FC":
            tensor_size_list[0] = [N**2, N, N]

        self.einsum_str = einsum_str
        self.idx_appearance_dict = idx_appearance_dict
        self.input_str_list = input_str_list
        self.ext_indices = ext_indices
        self.decomposition_type = decomposition_type

        self.layers = self.initialize_Tensors(
            tensor_size_list, init_scale
        )  # , random_init)
        self.N = N
        W = self.net_Weight
        self.W_shape_cum = get_W_shape_cum(W)
        # self.store_init_weights()

    def initialize_Tensors(self, tensor_size_list, init_scale0=1):
        tensor_depth = len(tensor_size_list)
        init_scale = init_scale0 ** (1 / tensor_depth)
        import pdb

        pdb.set_trace
        layers = []
        for siz in tensor_size_list:
            layer = Tensor_Layer2d(siz, init_scale=init_scale)
            layers.append(layer)

        return nn.ModuleList(layers)

    @property
    def T_param_list_x(self):
        return [layer.tensor_x for layer in self.layers]

    @property
    def T_param_list_y(self):
        return [layer.tensor_y for layer in self.layers]

    # @property
    def net_Weight2d(self, update):  # you can remove update and return both
        if update == "x":
            self._net_Weight_x = tensor_prod(self.T_param_list_x, self.einsum_str)
            return self._net_Weight_x  # .permute(2,0,1)
        elif update == "y":
            self._net_Weight_y = tensor_prod(self.T_param_list_y, self.einsum_str)
            return self._net_Weight_y  # .permute(2,0,1)
        else:
            raise ValueError("update should be x or y")

    def forward(
        self, x: Tensor, W=None, *args, **kwargs
    ):  # -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        if W is None:
            if kwargs.get("train_or_test") == "train":
                W_x = self.net_Weight2d("x")
                W_y = self.net_Weight2d("y")
            else:
                W_x = self._net_Weight_x  # existing weight
                W_y = self._net_Weight_y  # existing weight
        out = self.read_from_Tensor2d(W_x, W_y, x)
        return out, (W,)

    def read_from_Tensor2d(self, W_x, W_y, x):
        if x.dtype == torch.int64:  # x is tensor of indices
            out = self.index_select(W_x, x)
        else:
            # (this split is weird, data should have 3 tensors, x,y,z)
            # import pdb; pdb.set_trace()
            # x1,x2 = (tensor.squeeze(dim=1) for tensor in x.split(1,dim=1))
            # out = torch.einsum('ijk,bi,bj->bk',W,x1,x2)
            # here I should performing the convolution with the filter?
            # out = torch.einsum('ijk,bi,j->bk',W,x1,self.conv_weight)
            # out = torch.einsum('ijk,bi,j->bk',W,x,self.conv_weight)
            out = torch.einsum("iko,jlp,bij,kl->bop", W_x, W_y, x, self.conv_weight)
        return out

    def normalize(self):
        # normalize by diving the total norm instead
        self.conv_weight.data = F.normalize(
            self.conv_weight.data, p=2, dim=0
        )  # conw_weight is 6x36
        # self.net_Weight.data = F.normalize(self.net_Weight.data, p=2, dim=1) # net_Weight is 6x6x6
        # we want to update the factor weights. Divide the factor weights by  Frob(net_Weight) ^ 1/3
        return self

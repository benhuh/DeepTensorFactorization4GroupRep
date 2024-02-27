import torch
import math

#########################################
def tensor_prod(tensors, einsum_str='aki,bij,cjk->cab', normalization='1/n' ): #, noise=0.0
    if normalization == '1/n':
        return torch.einsum(einsum_str, *tensors) / tensors[0].shape[0] 
    else:
        print(normalization)
        raise ValueError
    
def unfold_tensors(tensor_list, idx_appearances):
    unfolded_tensors = []
    for idx_appearance in idx_appearances:
        tensor_id, idx_loc = idx_appearance['tensor_id'], idx_appearance['idx_loc']
        T = tensor_list[tensor_id]
        T_fold = unfold_tensor(T,idx_loc)
        unfolded_tensors.append(T_fold)
    return unfolded_tensors

import itertools

def compute_Custom_L2_all(tensor_list, idx_appearance_dict, expensive_version=False):
    loss = 0
    n = tensor_list[0].shape[0]
    for idx_char, idx_appearances in idx_appearance_dict.items():
        if len(idx_appearances) == 2:    # only look for doubly repeated indices
            M1, M2 = unfold_tensors(tensor_list, idx_appearances)
            loss += torch.einsum('ij,ji->', M1@M1.T, M2@M2.T)            
        else:
            pass

    return loss / n 

def compute_Manual_L2_all(factor_list):
    n = factor_list[0].shape[0]
    p_norms = [p.norm().pow(2) for p in factor_list]
    loss = sum(p_norms) / n 
    return loss

def compute_svd_all(tensor_list, idx_appearance_dict):
    svd_all = {}
    for idx_char, idx_appearances in idx_appearance_dict.items():
        if len(idx_appearances) > 1:    # only look for repeated indices
            CP = len(idx_appearances) > 2
            M_list = unfold_tensors(tensor_list, idx_appearances)

            svd_list = [compute_svd(M1, CP = CP) for (M1) in M_list]

            if len(svd_list)==1:
                svd_all[idx_char] = svd_list[0]
            else: 
                for id, svd  in zip(('i','j','k','l'), svd_list):    # for id, imb in enumerate(svd_list):
                    key = id 
                    svd_all[key] = svd
    return svd_all

def compute_imbalance2(tensor_list): 
    def helper(A, B, C):
        BB_T = torch.einsum('bxy,bzy->xz', B, B)
        ABBA = torch.einsum('aij,jl,akl->ik', A, BB_T, A)
        B_TB = torch.einsum('bxy,bxz->yz', B, B)
        CBBC = torch.einsum('cij,il,clk->jk', C, B_TB, C)
        return (ABBA-CBBC) / factor
    
    A,B,C=tensor_list

    factor = A.shape[0] **2

    imbalance2 = {'k': helper(A, B, C)/factor, 'i': helper(B, C, A)/factor, 'j': helper(C, A, B)/factor}
    return imbalance2


#########################################


def compute_svd(M1, CP = False):
    # M1, M2: tensors in matrix (folded) form

    if CP: 
        sig1, _ = M1.pow(2).sum(dim=1).sqrt().sort(descending=True)
        L1 = None
    else:
        try:
            L1, sig1, _ = M1.svd() 
        except:
            L1, sig1, _ = None, torch.zeros(M1.shape[0]), None

    sig1 /= M1.shape[0] ** 0.5
    return dict(sig=sig1, L=L1) 


def unfold_tensor(T,idx):
    if idx==0:
        if len(T.shape)>1:
            T = T.flatten(start_dim=1,end_dim=-1)
    else:
        if idx<len(T.shape)-1:
            T = T.flatten(start_dim=idx+1,end_dim=-1)
        T = T.flatten(start_dim=0,end_dim=idx-1)

        if len(T.shape)==2:
            T = T.T
        elif len(T.shape)==3:
            T = T.permute(1,0,2).flatten(1,-1)
        else:
            raise ValueError
    return T


########################################

def get_einsum_str(decomposition_type):

    idx0=0

    if decomposition_type in ["FC", "FCTrans"]:
        einsum_str = 'aki,bij,cjk->cab'
    else:
        raise ValueError(f"decomposition_type={decomposition_type} is not supported")
    return einsum_str, idx0


def get_tensor_size(N, r, einsum_str):
    input_str_list = ['aki', 'bij', 'cjk'] 
    int_indices = 'kij'
    ext_indices = 'cab'
    idx_appearance_dict = {'a': [{'tensor_id': 0, 'idx_loc': 0}], 'k': [{'tensor_id': 0, 'idx_loc': 1}, {'tensor_id': 2, 'idx_loc': 2}], 'i': [{'tensor_id': 0, 'idx_loc': 2}, {'tensor_id': 1, 'idx_loc': 1}], 'b': [{'tensor_id': 1, 'idx_loc': 0}], 'j': [{'tensor_id': 1, 'idx_loc': 2}, {'tensor_id': 2, 'idx_loc': 1}], 'c': [{'tensor_id': 2, 'idx_loc': 0}]}

    ext_range = [N] * 3
    int_range = [r] * 3
    idx_range_dict = {key: val for key, val in zip(ext_indices+int_indices, [*ext_range,*int_range])}
    range_dict = dict(ext = ext_range, int = int_range)

    tensor_size_list = get_size_from_str(input_str_list, idx_range_dict)
    return tensor_size_list, idx_appearance_dict, input_str_list, ext_indices, range_dict


def get_size_from_str(input_str_list, idx_range_dict):
    
    size_list = []
    for string in input_str_list:
        temp = []
        for idx_char in string:
            siz = idx_range_dict[idx_char]
            temp.append(siz)
        size_list.append(temp)
    return size_list    

#########################################


import itertools
import math

import torch
from torch import Tensor, LongTensor

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader, TensorDataset

import numpy as np
from typing import Tuple, List, Dict, Any, Union, Optional

from sympy.combinatorics.permutations import Permutation
from mod import Mod
import math
from copy import deepcopy

NUM_WORKERS = 16
MODULUS = 97 # 59 #
PERMUTATIONS = 5

from math import factorial

def get_perm(task):
    perm = int(task.split("sym")[-1].split("_")[0]  or PERMUTATIONS) # 3 if "sym3_xxx" 
    return perm

def get_default_N(task_name, default_modulus=None):
    if "sym" in task_name:
        perm = get_perm(task_name)
        return factorial(perm)
    else:
        return default_modulus or MODULUS
    
def set_random_seed(seed):

    # Set up the RNGs for repeatability
    if seed > 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

SHOWN_IN_GROK = ["addition", "subtraction", "division", "mix1", "quad1", "quad2", "quad3", "cube1", "cube2", "sym_xy", "sym_xyx_inv", "sym_xyx"]
SHOWN_IN_GROK_short = ["addition", "mix1", "quad2", "cube1", "sym_xy", "sym_xyx_inv"]

ALGORITHMIC_OPERATORS = {
    "binary/+": "addition",
    "binary/-": "subtraction",
    "binary/*": "muliplication",
    "binary//": "division",
    "binary/+*": "even-addition_odd-multiplication",
    "binary/+-": "even-addition_odd-subtraction",
    "binary/(x._value//y)if(y._value%2==1)else(x-y)_mod": "mix1",
    "binary/x**2+y**2_mod": "quad1",
    "binary/x**2+x*y+y**2_mod": "quad2",
    "binary/x**2+x*y+y**2+x_mod": "quad3",
    "binary/x**3+x*y_mod": "cube1",
    "binary/x**3+x*y**2+y_mod": "cube2",
    # 
    # "unary/sort": "sort",
    # "unary/reverse": "reverse",
    # "unary/copy": "copy",
# 
# GROUP_OPERATORS = {
    "binary/sym_xy": "sym_xy",
    "binary/sym_xyx_inv": "sym_xyx_inv",
    "binary/sym_xyx": "sym_xyx",
}

ALGORITHMIC_OPERATORS_GROK = {key:val for key,val in ALGORITHMIC_OPERATORS.items() if val in SHOWN_IN_GROK}
ALGORITHMIC_OPERATORS_GROK_short = {key:val for key,val in ALGORITHMIC_OPERATORS.items() if val in SHOWN_IN_GROK_short}

TOY_OPERATORS = {
    "binary/sym3_xy": "sym3_xy",
    "binary/sym3_xyx_inv": "sym3_xyx_inv",
    "binary/sym3_xyx": "sym3_xyx",
    "binary/sym4_xy": "sym4_xy",
    "binary/sym4_xyx_inv": "sym4_xyx_inv",
    "binary/sym4_xyx": "sym4_xyx",
}


VALID_OPERATORS = ALGORITHMIC_OPERATORS | TOY_OPERATORS 

# used only for text input data type
EOS_TOKEN = "<|eos|>"
EQ_TOKEN = "="
S5_range = 120 

DEFAULT_DATA_DIR = "data"

def calc_split_len(train_frac, total_data_batch): 
    assert train_frac <= 100.0
    if train_frac > 1:
        train_frac /= 100.0
    train_data_batch = round(total_data_batch * train_frac )
    val_data_batch = total_data_batch - train_data_batch
    return train_data_batch, val_data_batch
        
        
class ArithmeticDataset(TensorDataset):
    """A Dataset of arithmetic equations"""

    def __init__(self,  tensor_width = [0,0,0],
                        total_batch: int = 0, 
                        task: str = 'binary/+',
                        task_rank: int = None, 
                        operand_length: Optional[int] = None,
                        loss_type: str = 'classification',
                        data_type: str = 'tuple'  ,
                        M = None,
                        seed=0,
                        noise_level=0,
                        shuffle=True,
                        )  -> None:
        """
        :param data: A list of equations strings. Each equation must have an '=' in it.
        """
        self.tensor_width = tensor_width
        self.total_batch = total_batch
        self.name = self.get_dsname(task, operand_length)

        if seed<=0:
            seed = None
        self.rng = np.random.RandomState(seed=seed)

        vector_task = '_vec' in task
        if vector_task:
            task = task.replace('_vec','')

        if M is not None:
            assert data_type=='tuple'
            data = get_data_from_M(M)
            self.M = M
        else:
            data, self.M, self.factors = self.make_data(task, task_rank, operand_length, data_type) #, loss_type

        if vector_task:
            data = self._get_vectorized_data(self.M, total_batch=2.0)

        data = shuffle_data(data, self.rng, data_type, noise_level=noise_level, shuffle=shuffle)

        if data_type=='text':
            arithmetic_tokenizer = ArithmeticTokenizer(N = self.tensor_width) 
            data = arithmetic_tokenizer.encode(data)  
            super().__init__(data)

        elif isinstance(data, torch.Tensor):
            dim=1
            data_splitted =(tensor.squeeze(dim=dim) for tensor in data.split((2,1), dim=dim))
            super().__init__(*data_splitted)
        else:
            data, target = convert_list2tuple(data, loss_type, data_type)
            super().__init__(data, target)
        

    def _make_binary_operation_data(self, operator: str, data_type: str, operands=None) -> List[str]:
        N = self.tensor_width
        operator = operator.split("binary/")[1]
        if operator == "+-+-":  # composit
            operator = ["+", "-", "-&+", "-&-"]
        else:
            assert isinstance(N,int)
            
        if operator.startswith("sym"): 
            if "xyx" in operator: # in ["xyx_inv", "xyx"]:
                fnc = Permutation
            else: #if operator in ["xy"]:
                fnc = np.array
            inverse_factorial = {1:1, 2:2, 6:3, 24:4, 120:5, 720:6}                
            n_range = inverse_factorial[N]  # inverting factorial..
            
            operands = operands or list(range(n_range))
            elems = map(fnc, itertools.permutations(operands))
            tuples = itertools.product(elems, repeat=2)
            tuples_idx = itertools.product(range(N), repeat=2)            # N = TASK_INPUT_NUM["sym5"]
            
            elems_str = map(str,map(fnc, itertools.permutations(operands)))
            elems_dict = {v:k for k,v in enumerate(elems_str)}
            
        elif "_mod" in operator:
            modulo = N 
            elems = [Mod(i, modulo) for i in range(modulo)]
            tuples = itertools.product(elems, repeat=2)
            tuples_idx = itertools.product(range(modulo), repeat=2)            # N = TASK_INPUT_NUM["modulo"]
            
            elems_str = map(str, elems)
            elems_dict = {v:k for k,v in enumerate(elems_str)}
        else:
            operands = operands or list(range(N))
            tuples = itertools.product(operands, repeat=2)
            tuples_idx = deepcopy(tuples)
            
        eqs = []
        coo = []
        
        MODULUS = N
        
        operators = operator if isinstance(operator,(tuple,list)) else [operator]
        for (t, operator) in enumerate(operators): 
            for (a, b), (i,j) in zip(tuples, tuples_idx):
                if operator == "/":
                    if b == 0: # not part of eqs
                        continue
                    else:
                        c = a
                        a = (b * c) % MODULUS
                elif operator.startswith("sym"):
                    if "xyx_inv" in operator:  #  in [ "sym3_xyx_inv", "sym5_xyx_inv"]
                        c = a * b * (a.__invert__())
                    elif "xyx" in operator:  #  in [ "sym3_xyx", "sym5_xyx"]:
                        c = a * b * a
                    else: #if operator in ["sym3_xy", "sym_xy"]:
                        c = b[a]
                elif operator == "+":
                    c = (a + b) % MODULUS
                elif operator == "-":
                    c = (a - b) % MODULUS
                elif operator == "-&+":
                    c = (-a + b) % MODULUS
                elif operator == "-&-":
                    c = (-a - b) % MODULUS
                elif operator == "+*":
                    c = (a + b) % MODULUS if a % 2 == 0  else (a * b) % MODULUS
                elif operator == "+-":
                    c = (a + b) % MODULUS if a % 2 == 0  else (a - b) % MODULUS
                elif "_mod" in operator:
                    expression = operator.split("_mod")[0]
                    function = eval(f"lambda x, y: ({expression})")
                    c = function(a, b)
                else:
                    c = eval(f"({a} {operator} {b}) % {MODULUS}")
                    
                if operator.startswith("sym") or "_mod" in operator:  # Huh: convert back to index..
                    a = i #elems_dict[str(a)]
                    b = j #elems_dict[str(b)]
                    c = elems_dict[str(c)]

                if data_type=='tuple': 
                    if len(operators)>1:
                        eq = ((t, a, b), c)  # t = task index
                    else:
                        eq = ((a, b), c)
                elif data_type=='text':
                    # eq = " ".join(map(render, [a, operator, b, EQ_TOKEN, c]))
                    # eq = " ".join(map(render, [a, "o", b, EQ_TOKEN, c]))
                    eq = " ".join(map(render, [a, b, c]))
                else:
                    raise ValueError
                
                eqs.append(eq)
        
        # Get a sparse tensor representation of dataset
        if data_type=='tuple': 
            # import pdb; pdb.set_trace()
            coo = torch.tensor([(eq[1], *eq[0]) for eq in eqs])  # eqs[:5] =  [((0, 0), 0), ((0, 1), 1), ((0, 2), 2), ((0, 3), 3), ((0, 4), 4)]
            M = torch.sparse_coo_tensor(coo.T,torch.ones(coo.shape[0]).bool(), size=(N,)*coo.shape[1]) #.to_dense()
        else:
            M, coo = None, None
        return eqs, M, None #coo
  
    def get_output(self, M,xy):
        x,y = xy[:,0], xy[:,1]
        z = torch.einsum('ijk,bi,bj->bk',M.to_dense()+0.0,x,y)
        return z
    
    def _get_vectorized_data(self, M, total_batch=2.0):
        total_batch = int(total_batch * M.shape[0]**2)
        noise_level = 0.00

        x, y = torch.randn(2, total_batch, M.shape[0]) #.split((1,1))
        x, y = x/x.pow(2).sum(dim=1,keepdim=True).sqrt(), y/y.pow(2).sum(dim=1,keepdim=True).sqrt()
        xy = torch.stack((x, y),dim=1)
        z = self.get_output(M,xy)
        if noise_level > 0:
            noise = torch.randn_like(z); noise = noise/noise.pow(2).sum(dim=1,keepdim=True).sqrt()
            z += noise * noise_level

        data = torch.stack((x, y, z),dim=1)
        return data


    # @classmethod
    def get_dsname(self, task, operand_length) -> str:
        task, noise_level = self._get_operator_and_noise_level(task)
        ds_name = VALID_OPERATORS.get(task,task)
        if operand_length is not None:
            ds_name += f"_length-{operand_length}"
        if noise_level > 0:
            ds_name += f"_noise-{noise_level}"
        return ds_name


    @staticmethod # @classmethod
    def _get_operator_and_noise_level(task):
        if "_noisy" in task:
            task, noise_level = task.split("_noisy_")
            return task, int(noise_level)
        else:
            return task, 0

    # @classmethod
    def make_data(self, task, task_rank, operands=None, 
                  data_type='tuple',) -> List[str]:
        task, noise_level = self._get_operator_and_noise_level(task)     
        factors = None 

        if "binary/" in task: 
            data, M, factors = self._make_binary_operation_data(task, data_type, operands=None)
        else:
            raise ValueError(f"unsupported task: {task}")
        
        return data, M, factors
    
def shuffle_data(data, rng, data_type='text', noise_level=0, shuffle=True):
    tensor_data = isinstance(data, torch.Tensor)
    if tensor_data:
        dtype, data = data.dtype, data.numpy()
    if shuffle:
        rng.shuffle(data)
    if tensor_data:
        data = torch.tensor(data, dtype=dtype)

    # if data_type=='text':
    #     if noise_level > 0:
    #         random_answer_eqns = rng.choice(data, size=noise_level)
    #         random_answers = [ random_eq.split(" = ")[1] for random_eq in random_answer_eqns   ]
    #         for i in range(noise_level):
    #             data[i] = data[i].split(" = ")[0] + " = " + random_answers[i]

    if data_type=='tuple':
        if noise_level > 0:
            random_answer_eqns = rng.choice(data, size=noise_level)
            random_answers = [ random_eq[-1] for random_eq in random_answer_eqns ]
            for i in range(noise_level):
                data[i][-1] = random_answers[i]
    else:
        raise ValueError
    return data
    
def get_ortho(n0, n1=None):
    n1 = n1 or n0
    return torch.nn.init.orthogonal_(torch.empty(n0,n1))

    
def get_data_from_M(M, shapes_start=0):
    if shapes_start==0:
        tuples = itertools.product(*(list(range(m)) for m in M.shape))         # tuples = itertools.product(list(range(N)), repeat=len(M.shape))
    elif shapes_start==1:  # for "completion/tensor2"
        assert len(M.shape)>=3
        L=list(range(len(M.shape)));        L = L[1:]+L[:1]
        M = M.permute(L)
        tuples = itertools.product(*(list(range(m)) for m in M.shape[:-shapes_start]))         # tuples = itertools.product(list(range(N)), repeat=len(M.shape))
        
    int_type = M.dtype == torch.int64

    data_list = []
        
    for index_tuple in tuples:
        target = M[index_tuple].to(M.dtype)
        if shapes_start==0:
            target=target.unsqueeze(-1)
        if shapes_start==1 and int_type:
            target = target.argmax().item()
        data = (index_tuple, target)
        data_list.append(data)
    return data_list
    
def convert_list2tuple(data, loss_type, data_type):
    if data_type == 'text':
        input_list = [d[:-1] for d in data]
        target_list = [d[-1] for d in data]
        input = torch.stack(input_list, dim=0)
        target = torch.stack(target_list, dim=0)
    else:    
        assert isinstance(data,list) #and len(data[0])==2
        input_list = [d[0] for d in data]
        target_list = [d[1] for d in data]
    
        input = LongTensor(input_list)
    
        if isinstance( target_list[0], int) :
            if loss_type == 'regression':
                target = Tensor(target_list)
            else:
                target = LongTensor(target_list)
        else:
            target = torch.stack(target_list, dim=0)
        
    return input, target

def calculate_batchsize(ds_size: int, batchsize_hint: int = 1, max_batch=None) -> int:
    """
    Calculates which batch size to use

    :param ds_size: the number of equations in the dataset
    :param batchsize_hint: * 0 means we use a default batch_size
                            * -1 means the entire dataset
                            * float between 0 and 1 means each batch is
                                that fraction of the DS
                            * int > 1 means that specific batch size
    :returns: the actual batch_size to use
    """
    if (batchsize_hint > 0) and (batchsize_hint <= 1):
        if max_batch is None:
            return  math.ceil(ds_size * batchsize_hint)
        else:
            return  min(math.ceil(ds_size * batchsize_hint), max_batch)
    
    elif batchsize_hint > 1:
        return int(min(batchsize_hint, ds_size))
    else:
        print('batch_size:', batchsize_hint)
        raise ValueError("batchsize_hint must be >= 0")

###########################
# create Mask
def create_mask(datamodule):    
    x = datamodule.train_dataset.tensors[0]
    M_shape = datamodule.train_dataset.M.shape
    Mask = torch.sparse_coo_tensor(x.T,torch.ones(x.shape[0]).bool(), size=M_shape[:2]).to_dense() + 0.0
    Mask = Mask.unsqueeze(dim=2).repeat(1,1,M_shape[2])
    return Mask


class ArithmeticDataModule(LightningDataModule):
    def __init__(self, *args, M = None, **kwargs):
        super().__init__()
        for k,v in kwargs.items():
            setattr(self, k, v)

        self.setup_done=False
        self.M = M
        self.num_workers = getattr(self, 'num_workers', NUM_WORKERS)

    def setup(self, stage: Optional[str] = None):          # Assign train/val datasets for use in dataloaders
        # run setup only once
        if self.setup_done:
            pass
        else:
            self.setup_helper(stage)
            self.setup_done=True
    
    def setup_helper(self, stage):
        set_random_seed(self.seed)

        dataset_full = ArithmeticDataset(   tensor_width = self.tensor_width,
                                            task = self.task,
                                            task_rank = self.task_rank,
                                            data_type = self.data_type, 
                                            seed=self.seed,
                                            M = self.M, )
        split_sizes = calc_split_len(self.train_frac, len(dataset_full))
        
        from copy import deepcopy
        train_data_batch, val_data_batch = split_sizes
        self.train_dataset = deepcopy(dataset_full);  self.train_dataset.tensors = list(tensor[:train_data_batch] for tensor in self.train_dataset.tensors)
        self.val_dataset   = deepcopy(dataset_full);  self.val_dataset.tensors   = list(tensor[-val_data_batch:] for tensor in self.val_dataset.tensors)

        print('train dataset size:', len(self.train_dataset), 'val dataset size:', len(self.val_dataset), 'ratio:',len(self.train_dataset)/(len(self.train_dataset)+len(self.val_dataset)),'%') 
        
    def train_dataloader(self):
        batch_size = calculate_batchsize(len(self.train_dataset), batchsize_hint=self.batchsize_hint, max_batch=None) #100000)
        if isinstance(self.train_dataset,ArithmeticDataset):
            if self.batchsize_hint==1:
                return IterableDataset(self.train_dataset, batch_size, shuffle=False, drop_last=False)
            else:
                return IterableDataset(self.train_dataset, batch_size, shuffle=True, drop_last=True)
        else:
            return MultiEpochsDataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        batch_size = calculate_batchsize(len(self.val_dataset), batchsize_hint=1, max_batch=None) #20000) # Full batch
        # print('val', batch_size)
        if isinstance(self.val_dataset,ArithmeticDataset):
            return IterableDataset( self.val_dataset, batch_size, shuffle=False, drop_last=False)  # no need to batch validation data
        else:
            return MultiEpochsDataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)

    def test_dataloader(self):
        return self.val_dataloader()
            
#################################
    
class IterableDataset(torch.utils.data.IterableDataset):
    """
    An iterator over batches of data in an ArithmeticDataset
    """

    def __init__(self, dataset, batch_size, shuffle: bool = True, drop_last=False) -> None:
        """
        :param dataset: the dataset to iterate over
        :param device: the torch device to send batches to
        :param batchsize_hint: * 0 means we use a default batch_size
                               * -1 means the entire dataset
                               * float between 0 and 1 means each batch is
                                 that fraction of the DS
                               * int > 1 means that specific batch size
        :param shuffle: whether or not to randomly shuffle the dataset
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.reset_iteration(shuffle=shuffle)

    def reset_iteration(self, shuffle=True):
        self.ii = 0
        if shuffle: # and self.dataset.train:
            self.idx = torch.randperm(len(self.dataset))
        else:
            self.idx = torch.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Tensor]:
        """
        Returns one batch of data.
        :raises: StopIteration when we're out of data
        :returns: batch tensor of shape (self.batch_size, tokens_per_eq)
        """

        batch_begin = self.ii * self.batch_size
        batch_end = (self.ii+1) * self.batch_size
        should_end = (batch_end > len(self.dataset)) if self.drop_last else (batch_begin > len(self.dataset) - 1)
        
        if should_end: 
            # print('reset iteration', self.ii, len(self.dataset))
            self.reset_iteration()
            raise StopIteration
        self.ii += 1
        return self.dataset[self.idx[batch_begin : batch_end]]

    def __len__(self) -> int:
        """
        :returns: the total number of batches
        """
        # return math.ceil(len(self.dataset) / self.batch_size)
        return math.floor(len(self.dataset) / self.batch_size)

#################################
# from https://discuss.pytorch.org/t/enumerate-dataloader-slow/87778/3

class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.  """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    
################


class ArithmeticTokenizer:
    """Stores the list of token text to token id mappings and converts between them"""

    def __init__(self, N):
        self.itos = self.get_tokens(N) #, total_range = S5_range)
        self.stoi: Dict[str, int] = dict([(s, i) for i, s in enumerate(self.itos)])

    def _encode(self, s: str) -> Tensor:
        return LongTensor([self.stoi[t] for t in s.split(" ")])

    def encode(self, obj: Union[str, List]) -> Tensor:
        """
        Convert a string of text into a rank-1 tensor of token ids
        or convert a list of strings of text into a rank-2 tensor of token ids

        :param obj: the string or list of strings to convert
        :returns: a tensor of the token ids
        """
        if isinstance(obj, str):
            return self._encode(obj)
        elif isinstance(obj, list):
            if isinstance(obj[0], tuple):
                assert len(obj[0])==2 # input/target
                return obj
            else:
                return torch.stack([self._encode(s) for s in obj], dim=0)
        else:
            raise NotImplementedError

    def decode(self, tensor: Tensor, with_brackets: bool = False) -> str:
        """
        Convert a tensor of token ids into a string of text

        :param tensor: a tensor of the token ids
        :param with_brackets: if true, the returned string will include <> brackets
                              around the text corresponding to each token.
        :returns: string of these tokens.
        """
        indices = tensor.long()
        if with_brackets:
            l = "<"
            r = ">"
        else:
            l = ""
            r = ""
        tokens = [l + self.itos[i] + r for i in indices]
        return " ".join(tokens)

    def __len__(self) -> int:
        """
        :returns: the number of tokens in this vocabulary
        """
        return len(self.itos)

    @classmethod
    def get_tokens(cls, total_range):
        # if 'sym' in task:
        #     num = inverse_factorial[N]
        #     tokens = ( list(map(render, itertools.permutations(range(num)))))
        # else: 
        tokens = ( 
                    # ["o", EQ_TOKEN] #, EOS_TOKEN]
                    list(map(render, list(range(total_range)))) #NUMS+NUMS_extra))
                    # + list(sorted([k.replace('binary/','').replace('unary/','') for k in VALID_OPERATORS.keys()]))                    # # + list(sorted(list(VALID_OPERATORS.keys())))
                )
        return tokens
    

def render(operand, join_str=""):
    if (
        isinstance(operand, list)
        or isinstance(operand, tuple)
        or isinstance(operand, np.ndarray)
    ):
        return join_str.join(map(render, operand))
    elif isinstance(operand, Permutation):
        return "".join(map(str, operand.array_form))
    elif isinstance(operand, Mod):
        return str(operand._value)
    else:
        return str(operand)
    
    
    


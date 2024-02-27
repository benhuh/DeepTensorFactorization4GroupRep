from pathlib import Path #, PurePath

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
# import os 
import yaml

def get_events(log_dir, events_files, hparams_files, keep_dict=None):  #, version_limit=None  # Recursive!!
    # events_files, hparams_files = [], []
    log_dir = Path(log_dir).expanduser()

    if len(list(log_dir.glob('events.*')))>0:
        # version_num = int(log_dir.parts[-1].split('version_')[-1])
        # if version_limit is not None and version_num >= version_limit:
        #     # pass
        events_list = list(log_dir.glob('events.*'))
        hparams_list = list(log_dir.glob('hparams*'))
        
        # assert len(events_list) == 1   # and assert len(hparams_list) == 1
        # events_list.sort() #key=os.path.getmtime)   # sort by the time of last modification
        h = hparams_list[0]
        hparams = yaml.safe_load(Path(h).read_text())

        keep = True
        if keep_dict is not None:
            for key,val in keep_dict.items():
                if not isinstance(val, (list,tuple)):
                    val=[val]
                keep = keep and hparams.get(key, None) in val
        if keep:
            hparams_files.append(hparams)
            for e in events_list:
                evt_acc = EventAccumulator(str(e), purge_orphaned_data=False)
                evt_acc.Reload()
                events_files.append(evt_acc)

    else:
        sub_log_dir_list = list(log_dir.glob('*'))
        sub_log_dir_list.sort() #(key=os.path.getmtime)  # sort by the time of last modification

        for sub_log_dir in sub_log_dir_list:
            # events_files_, hparams_files_ = get_events(sub_log_dir, keep_dict)   # Recursive!!
            get_events(sub_log_dir, events_files, hparams_files, keep_dict)   # Recursive!!
            # if len(events_files_)>0:
            #     print(len(events_files_))
            # events_files += events_files_
            # hparams_files += hparams_files_
            # print(sub_log_dir,len(hparams_files_))
            
    return None #events_files, hparams_files
    
def fields_to_tuple(fields):
    return [ tuple(f.__dict__.values())  for f in fields]

def get_values(evt_acc, tag):
    try:
        timestamps, steps_, values_ = zip(*fields_to_tuple(evt_acc.Scalars(tag)))
        # timestamps, steps, values = zip(*evt_acc.Scalars(tag))
        return np.array(timestamps), np.array(steps_), np.array(values_)
    except:
        print(f'{tag} does not exist')
        return [None], [None], [None]


except_list = ['epoch', 'perplexity', 'hp_metric', 'singular vec', 'singular val', 'scheduler_criterion']
non_logarithmic_list = ['accuracy', 'weight'] #, 'scheduler'] #, 'erank']
non_logarithmic_list += ['sing_val i','sing_val j','sing_val k', 'singular_val']
# non_logarithmic_list += ['singular norm A','singular norm B','singular norm C',]

# tags_dict = dict(accuracy = [ 'accuracy/train',  'accuracy/validation'],
#                  loss = [ 'loss/train',  'loss/validation'],
#                  singular = [ f'singular/[{i}]' for i in range(20)],
#                )

def get_key_val(tag):
    splited = tag.split('/')

    if splited[0] in except_list:
        key_val = (None,)
    else:
        if len(splited)>1:
            # key,val = splited
            key = splited[0]
            val = "/".join(splited[1:])
            key_val = key, (val, tag)
        else:
            key_val = (None,)            # key_val = tag, ('-', tag) 
    return key_val

skip_tag_list_ICML = [ 'loss/reg'] #, 'loss/Lagrange']

def get_tags_dict(tag_list, skip_list=None, keep_list=None, plot_separately=False):
    skip_list = skip_list or []
    skip_tag_list = skip_tag_list_ICML
    tag_list.sort()

    losses = ['loss/train',  'loss/val', 'loss/reg']
    for key in losses:
        if key in tag_list:
            tag_list.remove(key)
    tag_list = losses + tag_list

    tag_dict = {}
    for tag in tag_list:
        if tag in skip_tag_list:
            continue
        key_val = get_key_val(tag)
        if len(key_val)==2:
            key,val = key_val
            if key not in skip_list and (keep_list is None or key in keep_list):
                if not (plot_separately and val[0] in ['i','j','k']):
                    if key not in tag_dict:
                        tag_dict[key] = [val]
                    else:
                        tag_dict[key] += [val]
        else:
            pass
    return tag_dict


from dtf.training import get_print_str 
from dtf.visualization_new import get_default_save_name

def plot_scalars(log_dir, tags_dict=None, save_fig=False, save_name=None, skip_list=None, keep_list=None, ylims=None, plot_separately=False, decay_coeff=0, **kwargs):

    events_files, hparams_files = [], []
    get_events(log_dir, events_files, hparams_files)
    
    if len(events_files)==0:
        return 
    print(len(events_files))
    e = events_files[0]    # print(str(e))

    hparams = hparams_files[0]
    get_print_str(hparams)

    if tags_dict is None:
        tags_dict = get_tags_dict(e.scalars.Keys(), skip_list, keep_list, plot_separately)
    # print(e.scalars.Keys(), tags_dict.keys())

    if plot_separately:
        temp = []
        for val in tags_dict.values():
            temp += val
        num_panels = len(temp)
    else:
        num_panels = len(tags_dict.keys())
    
    # print(num_panels, tags_dict)
    # print(e.scalars.Keys())
    if len(tags_dict)<2:
        raise ValueError

    # figsize=(3.*num_panels, 3) #figsize=(8*num_panels, 5)
    figsize=(2.2*num_panels, 2.5) #figsize=(8*num_panels, 5)
    fig, axes = plt.subplots(1, num_panels, figsize=figsize, sharex=True, sharey=False)
    fig.tight_layout(w_pad=3, h_pad=3)

    axes = axes if num_panels>1 else [axes]
    ylims = ylims or [None]*num_panels
        
    
    # for ax, (key,val_tags), ylim in zip(axes, tags_dict.items(), ylims):
    j=0
    for i, ((key,val_tags), ylim) in enumerate(zip(tags_dict.items(), ylims)):
        # print(key)
        vals=[]
        for (val,tag) in val_tags:
            if plot_separately:
                ax=axes[j]
                j+=1
            else:
                ax=axes[i]

            vals.append(val)
            plot = ax.plot  if key in non_logarithmic_list else  ax.semilogy
            # plot = ax.semilogx  if key in non_logarithmic_list else  ax.loglog
            steps, values = [], []
            for e in events_files:
                timestamps, steps_, values_ = get_values(e, tag)
                steps.append(steps_); values.append(values_)
            if plot_separately:
                for step,value in zip(steps,values):
                    plot(step,value)
                set_axis(ax, tag, None, ylim)
            else:
                steps = np.concatenate(steps)
                values = np.concatenate(values)
                plot(steps,values)

            # print(tag)
            # if tag in ['imbalance/mean']:
            # if tag in ['imbalance2/mean']:
                # values = np.exp(-decay_coeff*steps*2)/3
            #     plot(steps, values, 'k:')
            # if tag == 'singular_val/i/0':
                # values = np.exp(-decay_coeff*steps)/2
                # plot(steps, values, 'k:')
        if not plot_separately:
            set_axis(ax, key, vals, ylim)

    plt.subplots_adjust(left=0, right=0.9, top=1, bottom=0)
    plt.tight_layout(w_pad=-0.1, h_pad=0)
    plt.show()

    save_name = get_default_save_name(None, add_str='training_traj', save_fig=save_fig, save_name=save_name)        # save_name = save_name or get_default_save_name(log_dir)
    if save_name is not None:
        fig.savefig(f'{save_name}.pdf')

title_dict = dict(
    loss = 'Loss',
    accuracy = 'Accuracy',
    hessian='Hessian',
    imbalance2='Imbalance',
    singular_val='Singular Value',
    orth_loss = 'Orthogonality',
    grad_norm = 'Gradient Norm',
    norm = 'Norm',
)

# from collections import defaultdict
vals_dict = dict(
    val = 'test',
    all = 'tensor',
    indiv = 'slice',
)
def set_axis(ax, key, vals, ylim):

    ax.set_title(title_dict[key])
    # ax.set_ybound(lower=1e-6, upper=None)
    # ax.set_xbound(lower=.75e1, upper=None)
    # ax.set_xbound(lower=.75e1, upper=170)
    ax.set_xbound(lower=1, upper=None)

    if key == 'singular':
        ax.set_ybound(lower=1e-3, upper=5e0)
        # ax.set_ybound(lower=1e-6, upper=None)
        # ax.set_ybound(lower=1e-2, upper=None)
    elif key.startswith('singular_val0'):
        pass
    elif key.startswith('singular_val'):
        # ax.set_ybound(lower=1e-1, upper=None)
        # ax.set_ybound(lower=3e0, upper=3e1)
        # ax.set_ybound(lower=3e-1, upper=3e0)
        # ax.set_ybound(lower=None, upper=None)
        pass
    elif key.startswith('character'):
        pass
    elif key.startswith('hessian'):
        pass
    else:
        # ax.set_ybound(lower=1e-3, upper=1.5e1)
        # ax.set_ybound(lower=1e-5, upper=1.5e1)
    # if key != 'singular':
        if vals is not None:
            legend = [vals_dict.get(elm, None) or elm for elm in vals]
            ax.legend(legend)
            # print(vals)
            # import pdb; pdb.set_trace()
    if key == 'loss':
        # ax.set_ybound(lower=3e-5, upper=2e1)
        # ax.set_ybound(lower=1e-4, upper=1e-1)
        # ax.set_ybound(lower=1e-2, upper = 3e-1)
        ax.set_ybound(lower=1e-3, upper=None)
    elif key == 'accuracy':
        ax.set_ybound(lower=0, upper=110)
    elif key.startswith('imbalance'):
        ax.set_ybound(lower=1e-6, upper=1e0)
    elif key == 'orth_loss':
        ax.set_ybound(lower=1e-4, upper=1e2)
        # ax.set_ybound(lower=1e-2, upper=1e1)

    if ylim is not None:
        ax.set_ybound(lower=ylim[0], upper=ylim[1])    

    if key.startswith('imbalance'):
        # ax.set_xlabel(r'iteration #: $t$')
        ax.set_xlabel(r'iteration $t$')
# ###################################

# import numpy as np
# from tinydb import TinyDB, Query, where
# from tinydb.storages import MemoryStorage
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# from pathlib import Path
# import json


# def get_db(base_path, db=None, load_flag=True):
#     if db is None:
#         db = TinyDB(storage=MemoryStorage)
#     if load_flag:
#         load(base_path, db)
#     return db

# def load(base_dir, db=None, print_flag=True): 
#     if print_flag:
#         print(f'Loading {base_dir}')
    
#     log_dir_list = list(Path(base_dir).expanduser().glob('*'))
#     log_dir_list.sort(key=os.path.getmtime)
#     for log_dir in log_dir_list:
#         if not log_dir.is_dir() or db.contains(Query().log_dir == str(log_dir)):
#             continue
#         try:
#             if len(list(log_dir.glob('events.*')))>0:
#                 events_files = list(log_dir.glob('events.*'))
#                 config_files = list(log_dir.glob('config*'))
#                 events_files.sort()#key=os.path.getmtime)
#                 config_files.sort()#key=os.path.getmtime)
#                 # assert(len(events_files)==1)
#                 # assert(len(config_files)==1)
#                 for events_file, config_file in zip(events_files, config_files):
#                     # print(f'Loading {events_file}')
#                     evt_acc = EventAccumulator(str(events_file), purge_orphaned_data=False)
#                     evt_acc.Reload()
                    
#                     with open(str(config_file)) as file:
#                         config = json.load(file)
#                     # config['base_dir'] = str(base_dir)
#                     # config['log_dir'] = str(log_dir)
                    
#                     config = {**config, **config['problem']}
#                     config.pop('problem')
#                     config['events'] = evt_acc

#                     db.insert(config)
#             else: # Recursive load 
#                 load(log_dir, db, print_flag=True)
#         except Exception as e: 
#             print(e)

# # def unload(base_dir):
# #     for log_dir in Path(base_dir).expanduser().glob('*'):
# #         if not log_dir.is_dir() or not db.contains(Run.log_dir == str(log_dir)):
# #             continue
# #         db.remove(where('log_dir') == str(log_dir))
# #         print('Removing', log_dir)
        
# def get_values(run, tag):
#     timestamps, steps, values = zip(*run['events'].Scalars(tag))
#     return np.array(values)

# def get_steps(run, tag):
#     timestamps, steps, values = zip(*run['events'].Scalars(tag))
#     return np.array(steps)


# ###########################################
# import os
# from torch.utils.tensorboard import SummaryWriter
# from logging import Logger #getLogger, FileHandler

# def set_logging(config, kwarg_dict):

#     tensorboard_path, partial_path = get_tensorboard_path(config, kwarg_dict)
#     print(partial_path)

#     _writer = SummaryWriter(tensorboard_path, flush_secs=30)
#     _log = Logger(name=tensorboard_path)
#     return _writer, _log, tensorboard_path

# def save_config(config, path, filename, pop_list:list=None):
#     os.makedirs(path, exist_ok=True)
#     config_path = os.path.join(path, filename)
    
#     config_dict = vars(config).copy()
#     config_dict['problem'] = vars(config_dict['problem'])
#     if pop_list is not None: #len(pop_list)>0:
#         for pop_key in pop_list:
#             config_dict.pop(pop_key)
                
#     with open(config_path, 'a') as f:
#         f.write(json.dumps(config_dict, sort_keys=True) + "\n")
            

# def get_base_path(config): 
#     root_path = os.path.dirname(os.path.realpath(__file__))
#     base_path = os.path.join(root_path, 
#                              'results', 
#                              config.problem.name,
#                              config.experiment,
#                              # dict2str(vars(config.problem)), 
#                              )
#     return base_path

# import re
# def dict2str(cfg_dict):
#     cfg_dict_ = cfg_dict.copy()
#     for key, val in cfg_dict.items():
#         if key == 'name' or val == None: # remove 'name' and any None entries
#             cfg_dict_.pop(key)
            
#     config_str = json.dumps(cfg_dict_) #, sort_keys=True)
#     config_str = re.sub(r"('|{|}| )", "", config_str)
#     config_str = re.sub(r'"', '', config_str)
#     config_str = re.sub(r",", "|", config_str)
#     return config_str


# def get_tensorboard_path(config, kwarg_dict): # **kwargs): 
#     # config.set_params(**kwarg_dict)    
#     base_path = get_base_path(config)
    
#     add_path = dict2str(kwarg_dict)
    
# #     _wide = '_wide' if config.wide else '_narrow'
# #     add_path = f'depth{config.depth}_wd{config.weight_decay}_init_scale{config.init_scale}' + _wide 

#     path = os.path.join(base_path, add_path)
#     tensorboard_path, run_path = get_run_path(path, run_str = 'run')
#     return tensorboard_path, add_path+'/'+run_path

# def get_run_path(path, run_str = 'run', suffix=''):
#     run_num = 0
#     while True:
#         run_path = run_str + str(run_num)+suffix
#         full_path = os.path.join(path, run_path)
#         if not os.path.exists(full_path):
#             break
#         run_num += 1
#     return full_path, run_path

# ###########################################

# import matplotlib.pyplot as plt

# def make_2_plot(db, loop_params, params, loss_or_svd=None):
#     key1, vals1 = loop_params.popitem()  # last=False
#     key2, vals2 = loop_params.popitem()

#     fig, axes = plt.subplots(len(vals1), len(vals2), figsize=(20, 12) , sharex=True, sharey=True)
#     fig.tight_layout(w_pad=3, h_pad=3)
    
#     for i, val1 in enumerate(vals1):
#         params[key1] = val1 

#         for j, val2 in enumerate(vals2):
#             params[key2] = val2
            
#             ax = axes[i,j]
#             runs = db.search(Query().fragment(params))
#             # print(params)
#             # runs = runs[-1:]
#             # print(run)
#             # test_loss = get_values(run, 'loss/test')[-1]
#             # train_loss = get_values(run, 'loss/train')[-1]

#             make_1_plot(runs, loss_or_svd, ax=ax)
                
#             # ax.set_title(f'{_wide}, depth{depth}, wd{wd}, loss: [{test_loss:.4f}, {train_loss:.4f}]')
#             ax.set_title(params)
#             # ax.set_xlim(left = 0, right = T) 


# def make_1_plot(db, loop_params, params, loss_or_svd=None):
#     key1, vals1 = loop_params.popitem()  # last=False

#     fig, axes = plt.subplots(len(vals1),  figsize=(20, 12) , sharex=True, sharey=True)
#     fig.tight_layout(w_pad=3, h_pad=3)
    
#     for i, val1 in enumerate(vals1):
#         params[key1] = val1 

#         ax = axes[i]
#         runs = db.search(Query().fragment(params))
#         # print(params)
#         # runs = runs[-1:]
#         # print(run)
#         # test_loss = get_values(run, 'loss/test')[-1]
#         # train_loss = get_values(run, 'loss/train')[-1]

#         make_0_plot(runs, loss_or_svd, ax=ax)

#         # ax.set_title(f'{_wide}, depth{depth}, wd{wd}, loss: [{test_loss:.4f}, {train_loss:.4f}]')
#         ax.set_title(params)
#         # ax.set_xlim(left = 0, right = T) 

            
            
# def make_0_plot(db, loss_or_svd=None, params=None, ax=None):
#     if params is None:
#         runs = db
#     else:
#         runs = db.search(Query().fragment(params))

#     if ax is None:
#         fig = plt.figure(figsize=(8, 6))
#         ax = plt.axes()

#     alpha=1
    
#     for run in runs:
    
#         # print(depth, dataset, lr, np.min(get_values(run, 'loss/test')))
#     #     xs = get_steps(run, 'singular_values/0')
#         xs = get_steps(run, 'loss/test')

#         if loss_or_svd in [None, 'loss']:
#     #                     ax.semilogy(xs, get_values(run, 'loss/surrogate'), '-', color='blue', alpha=alpha)
#             ax.semilogy(xs, get_values(run, 'loss/test'), '--', color=plt.cm.hot(0 / 10), alpha=alpha)
#             ax.semilogy(xs, get_values(run, 'loss/train'), ':', color=plt.cm.hot(3 / 10), alpha=alpha)
#             # ax.set_ylim(bottom = 0.0001, top = 3)

#         if loss_or_svd in [None, 'svd']:
#             for k in range(20):
#                 try:
#                     ax.semilogy(xs, get_values(run, f'singular_values/{k}'), color=plt.cm.summer(k / 10), alpha=alpha)
#                 except:
#                     pass
#         ax.set_ylim(bottom = 1e-4, top = 100)    
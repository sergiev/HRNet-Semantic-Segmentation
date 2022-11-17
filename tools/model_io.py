import argparse
from lib2to3.pytree import convert

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from yacs.config import CfgNode
from pathlib import Path

import _init_paths
import models
import datasets
from config import config
from config import update_config

DICT_VALID_TYPES = {tuple, list, str, int, float, bool}


def convert_to_dict(cfg_node, key_list=[]):
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in DICT_VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), DICT_VALID_TYPES), )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict

def load_model(cfg_path, weights_path):
    args = argparse.Namespace(cfg=cfg_path, 
        opts=['TEST.MODEL_FILE', weights_path])
    update_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)
    model_state_file = config.TEST.MODEL_FILE  

    print('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    return model

if __name__=='__main__':
    bound_cfg = 'experiments/arable_boundaries/baseline_paddle.yaml'
    field_cfg = 'experiments/arable_fields/fields_paddle.yaml'

    bound_weights = 'output/arable_boundaries/baseline_paddle/best.pth'
    field_weights = 'output/arable_fields/fields_paddle/best.pth'
    
    model_bound = load_model(bound_cfg, bound_weights)  # model for segmenting boundaries
    bound_cfg = convert_to_dict(config.clone())
    model_field = load_model(field_cfg, field_weights)  # model for segmenting fields
    field_cfg = convert_to_dict(config.clone())

    out_path = f"output/arable_both_models/hrnet_arable_both.pth"
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    torch.save({
                'field_state_dict': model_field.state_dict(),
                'bound_state_dict': model_bound.state_dict(),
                'field_cfg': field_cfg,
                'bound_cfg': bound_cfg
                }, out_path)
    print("successfully:", out_path)
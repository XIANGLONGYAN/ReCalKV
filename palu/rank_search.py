
import os, click
import torch
import torch.nn as nn
from loguru import logger
from .model import AVAILABLE_MODELS
from .data_utils import get_calib_data
import math
import re
import numpy as np
from tqdm import tqdm
import json
import sys
from palu.decomposition import get_whiten_scale_matrix

# default block size is 32
def rounding_search_result(config: dict, block_size=32):
    for module_name in config.keys():
        ranks = config[module_name]
        for i in range(len(ranks)):
            ranks[i] = max(1, round(ranks[i] / block_size)) * block_size
        config[module_name] = ranks
    return config

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def split_values(data, group_number):
    result = {}
    for key, value in data.items():
        new_value = [v // group_number for v in value for _ in range(group_number)]
        result[key] = new_value
    return result

def calib_fisher_info(model, calib_loader, device, use_cache=True):
    model.half()
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_calib_fisher_info.pt"

    logger.info(f"[Fisher] Search cache_file={cache_file}", fg="yellow")


    cache_file_whiten = f"cache/whiten/{model_id.replace('/','_')}_w2_scaling_matrices_fp16.pt"
    if os.path.exists(cache_file_whiten) and use_cache:
        logger.info(f"[whiten] File {cache_file_whiten} exist.", fg="green")
        logger.info(f"[whiten] Load scaling diag matrix from cache: {cache_file_whiten}", fg="yellow")
        scaling_matrics = torch.load(cache_file_whiten, map_location="cpu")

    if os.path.exists(cache_file) and use_cache:
        logger.info(f"[Fisher] File {cache_file} exist.", fg="green")
        logger.info(f"[Fisher] Load cache_file={cache_file}", fg="yellow")
        all_fisher_info = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "attn" in name:
                module.fisher_info = all_fisher_info[name].to(module.weight.device)
        return

    model.eval()

    logger.info(f"[Fisher] No cache_file={cache_file}", fg="red")
    logger.info(f"[Fisher] Create fisher info list...", fg="yellow")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "attn" in name:
            module.fisher_info = 0

    # get fisher info
    for batch in tqdm(calib_loader):
        input_ids = batch["input_ids"][:, :-1].to(model.device)
        labels = batch["input_ids"][:, 1:].to(model.device)
        out = model(input_ids=input_ids, labels=labels)
        out[0].backward()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "attn" in name:
                module.fisher_info += module.weight.grad.detach().to(torch.float32).pow(2)
        model.zero_grad()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "attn" in name:
            module.fisher_info = module.fisher_info.div(len(calib_loader)).sqrt()

    # remove and save fisher_info
    all_fisher_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "attn" in name:
            module._forward_hooks.clear()
            all_fisher_info[name] = module.fisher_info
    

    logger.info(f"[Fisher] Save the fisher info list to:  {cache_file}", fg="yellow")
    torch.save(all_fisher_info, cache_file)
    model.cpu()

@torch.no_grad()
def rank_search(model: nn.Module, tokenizer, args):
    logger.info(f"[Rank search] Do rank searching. Search method: {args.search_method}", fg="yellow")
    if args.search_method == "uniform":
        target_model_class = AVAILABLE_MODELS[model.config.model_type]["ModelForCausalLM"]
        total_rank = 0
        select_result = {}
        # get_kv_info returns (number of low-rank groups for K/V, hidden dim per group)
        info = target_model_class.get_kv_info(model, args.head_group_size)
        
        for name, module in model.named_modules():
            if "k_proj" in name or "v_proj" in name:                
                module_rank = info.num_lr_groups * info.lr_group_dims
                total_rank += module_rank
                
                select_result.update({name: [info.lr_group_dims*args.param_ratio_target] * info.num_lr_groups})

        # round to a multiple of 32 (memory alignment)
        select_result = rounding_search_result(select_result)
        rank_sum = sum([sum(v) for k, v in select_result.items()])
        logger.info(f"[Rank search] KV-Cache Compression Ratio: {100-(rank_sum / total_rank * 100): .2f}%")
        return select_result, rank_sum, total_rank    
    elif args.search_method == "fisher":
        # Prepare Fisher information
        calib_loader = get_calib_data(args.calib_dataset, tokenizer, args.model_id, args.calib_nsamples, seqlen=args.calib_seqlen)
        
        calib_fisher_info(model, calib_loader, torch.device(args.device), args.use_cache)
        
        target_model_class = AVAILABLE_MODELS[model.config.model_type]["ModelForCausalLM"]
        total_rank = 0
        fisher_sum = 0.0
        fisher_info_dict = {}
        select_result = {}
        
        info = target_model_class.get_kv_info(model, args.head_group_size)
        for name, module in model.named_modules():
            if "k_proj" in name or "v_proj" in name:
                module_rank = info.num_lr_groups * info.lr_group_dims
                total_rank += module_rank
                
                select_result.update({name: [info.lr_group_dims] * info.num_lr_groups})
                
                fisher = module.fisher_info.reshape(info.num_lr_groups, -1, module.in_features)
                if not torch.isfinite(fisher).all():
                    logger.info(fisher)
                
                fisher_list = [torch.mean(fisher[i]).item() for i in range(info.num_lr_groups)]
        
                fisher_info_dict.update({name: fisher_list})
                fisher_sum += sum(fisher_list)

        target_rank = total_rank * args.param_ratio_target
        
        indexes = []
        select_result_float = {}

        for name, fisher in fisher_info_dict.items():
            ranks = []
            for i in range(len(fisher)):
                rank_float = target_rank * fisher[i] / fisher_sum
                
                ranks.append(rank_float)
                indexes.append((name, i))
                select_result[name][i] = min(select_result[name][i], math.floor(rank_float))

            select_result_float.update({name: ranks})
                
        indexes = sorted(indexes, key=lambda x: select_result_float[x[0]][x[1]] - select_result[x[0]][x[1]])
        dif = target_rank - sum([sum(v) for k, v in select_result.items()])

        while dif > 0:
            for i in range(len(indexes)):
                if select_result[indexes[i][0]][indexes[i][1]] == info.lr_group_dims:
                    continue
                select_result[indexes[i][0]][indexes[i][1]] += 1
                dif -= 1

                if dif == 0:
                    break
        select_result = rounding_search_result(select_result)
        rank_sum = sum([sum(v) for k, v in select_result.items()])
        logger.info(f"[Rank Search] KV-Cache Compression Ratio: {100-(rank_sum / total_rank * 100): .2f}%")
        return select_result, rank_sum, total_rank
    elif args.search_method == "fisher_uniform":
        # Prepare Fisher information, wikitext2 2048
        calib_loader = get_calib_data(args.calib_dataset, tokenizer, args.model_id, args.calib_nsamples, seqlen=args.calib_seqlen)
        calib_fisher_info(model, calib_loader, torch.device(args.device), args.use_cache)
        
        # target_model_class: PaluLlamaForCausalLM
        target_model_class = AVAILABLE_MODELS[model.config.model_type]["ModelForCausalLM"]
            
        total_rank = 0
        
        fisher_sum = 0.0
        fisher_info_dict = {}
        select_result = {}
        # Passing num_key_value_heads as the group size makes each k/v projection a single group; the rank is split into per-head-group ranks later
        info = target_model_class.get_kv_info(model, model.config.num_key_value_heads)

        for name, module in model.named_modules():
            if "k_proj" in name or "v_proj" in name:
                # num_lr_groups=1, lr_group_dims=4096
                module_rank = info.num_lr_groups * info.lr_group_dims
                
                total_rank += module_rank
                
                select_result.update({name: [info.lr_group_dims] * info.num_lr_groups})
                
                # 1 * output_features * in_features
                fisher = module.fisher_info.reshape(info.num_lr_groups, -1, module.in_features)
                
                fisher_list = [torch.mean(fisher[i]).item() for i in range(info.num_lr_groups)]
                
                fisher_info_dict.update({name: fisher_list})
                fisher_sum += sum(fisher_list)

        
        # total rank budget across all K/V matrices
        target_rank = total_rank * args.param_ratio_target
        
        indexes = []
        select_result_float = {}

        for name, fisher in fisher_info_dict.items():
            ranks = []
            for i in range(len(fisher)):
                # allocate rank linearly proportional to each group's mean Fisher info
                rank_float = target_rank * fisher[i] / fisher_sum
                ranks.append(rank_float)
                indexes.append((name, i))
                select_result[name][i] = min(select_result[name][i], math.floor(rank_float))
            
            select_result_float.update({name: ranks})
        
        # indexes = sorted(indexes, key=lambda x: select_result_float[x[0]][x[1]] - select_result[x[0]][x[1]])
        
        indexes = sorted(indexes, key=lambda x: select_result_float[x[0]][x[1]] - select_result[x[0]][x[1]], reverse=True)
        

        dif = target_rank - sum([sum(v) for k, v in select_result.items()])

        while dif > 0:
            for i in range(len(indexes)):
                if select_result[indexes[i][0]][indexes[i][1]] == info.lr_group_dims:
                    continue
                select_result[indexes[i][0]][indexes[i][1]] += 1
                dif -= 1

                if dif == 0:
                    break
        
        
        # fisher_uniform treats each layer as one group; split each layer's rank evenly across its head-groups
        select_result = split_values(select_result, model.config.num_key_value_heads//args.head_group_size)

        # round every rank to a multiple of 32 for memory alignment
        select_result = rounding_search_result(select_result)
        rank_sum = sum([sum(v) for k, v in select_result.items()])
        logger.info(f"[Rank Search] KV-Cache Compression Ratio: {100-(rank_sum / total_rank * 100): .2f}%")


        return select_result, rank_sum, total_rank
    else:
        raise NotImplementedError 
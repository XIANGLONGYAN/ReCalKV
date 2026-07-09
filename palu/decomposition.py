from loguru import logger
import random
import torch.nn as nn
import torch
import os
import click
import sys
from tqdm import tqdm
from .data_utils import get_calib_data
from .model import HeadwiseLowRankModule
from .paths import dataset_dir
from datasets import load_from_disk
# from memory_profiler import profile


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_wikitext2(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=dataset_cache_dir)
    # testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=dataset_cache_dir)
    
    traindata = load_from_disk(dataset_dir("wikitext", "traindata"))
    testdata = load_from_disk(dataset_dir("wikitext", "testdata"))

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_loaders(name, nsamples=256, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if 'ptb' in name:
        # if 'new' in name:
        #     return get_ptb_new(nsamples, seed, seqlen, tokenizer)
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if 'c4' in name:
        # if 'new' in name:
        #     return get_c4_new(nsamples, seed, seqlen, tokenizer)
        return get_c4(nsamples, seed, seqlen, tokenizer)

def get_ptb(nsamples, seed, seqlen, tokenizer):
    # traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    # testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    traindata = load_from_disk(dataset_dir("ptb", "traindata"))
    testdata = load_from_disk(dataset_dir("ptb", "testdata"))

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_c4(nsamples, seed, seqlen, tokenizer):
    # traindata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    # valdata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'

    traindata = load_from_disk(dataset_dir("c4", "traindata"))
    valdata = load_from_disk(dataset_dir("c4", "valdata"))

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_parent_module(model, full_module_name):
    """
    Given the model and a full module name (e.g. "model.layers.0.self_attn.v_proj"),
    return the parent module and the final attribute name.
    """
    parts = full_module_name.split(".")
    parent = model
    for part in parts[1:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]

@torch.no_grad()
def get_whiten_scale_matrix(model, tokenizer, args, dev):
    model_id = model.config._name_or_path
    #NOTE (brian1009): Might need to check the random seed, currently we have < 0.1 perplexity difference at Llama2-7B
    calib_loader = get_calib_data(
        "wikitext2", 
        tokenizer, 
        model_id, 
        nsamples=256, 
        seqlen=2048
    )
    cache_file = f"cache/whiten/{model_id.replace('/','_')}_w2_scaling_matrices_fp16.pt"
    os.makedirs("cache/whiten", exist_ok=True)
    """
    cache format:
    [
        {
            "attn.q_proj": torch.Tensor,
            "attn.k_proj": torch.Tensor,
            "attn.v_proj": torch.Tensor,
            "attn.o_proj": torch.Tensor,
            "mlp.gate_proj": torch.Tensor,
            "mlp.up_proj": torch.Tensor,
            "mlp.down_proj": torch.Tensor
        },
        ... (stacked n times, in the order of model layers)
    ]
    """
    logger.info(f"[whiten] Calibration dataset: {args.calib_dataset}", fg="yellow")
    logger.info(f"[whiten] Search cache_file={cache_file}", fg="yellow")

    if os.path.exists(cache_file) and args.use_cache:
        logger.info(f"[whiten] File {cache_file} exist.", fg="green")
        logger.info(f"[whiten] Load scaling diag matrix from cache: {cache_file}", fg="yellow")
        scaling_matrics = torch.load(cache_file, map_location="cpu")


        layers = model.model.layers
        for i in tqdm(range(len(layers))):
            layer = layers[i]
            subset = find_layers(layer) # Collect all linear layers
            for name in subset:
                if name in scaling_matrics[i]:
                    scaling_diag_matrix = scaling_matrics[i][name]
                    subset[name].scaling_diag_matrix = scaling_diag_matrix

        return 
    
    logger.info(f"No cache_file={cache_file}", fg="red")
    logger.info(f"Create whiten scale matrix dict...", fg="yellow")

    # Create Scaling Matrix with low-resource inference
    # Adapted from https://github.com/AIoT-MLSys-Lab/SVD-LLM/blob/main/SVDLLM.py
    # Here, inference are performed in an layer-wise manner.
    use_cache = model.config.use_cache
    model.config.use_cache = False
    #FIXME: This is not a good implementation...
    if "llama" in model_id or "mistral" in model_id or "vicuna" in model_id or "longchat":
        layers = model.model.layers
    elif "opt" in model_id:
        layers = model.model.decoder.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    # inps = torch.zeros(
    #     (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "position_embeddings": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
                cache["position_embeddings"] = kwargs.get("position_embeddings", None)
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    scaling_matrices = []
    logger.info("[Decomposition] Start to calculate the scaling matrix in layer-wise manner...")
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0), position_ids=position_ids[0].unsqueeze(0), position_embeddings=position_embeddings)[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        layer_scaling_matrices = {}
        


        for name in subset:
            if not ("k_proj" in name or "v_proj" in name):
                continue
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().cuda()
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                subset[name].scaling_diag_matrix = scaling_diag_matrix
            except Exception as e:
                logger.warning("eigen scaling_diag_matrix is not positive!")
                if torch.isnan(raw_scaling_diag_matrix).any():
                    logger.warning("raw scaling_diag_matrix contains NaN!")
                elif torch.isinf(raw_scaling_diag_matrix).any():
                    logger.warning("raw scaling_diag_matrix contains Inf!")
                if not torch.equal(raw_scaling_diag_matrix, raw_scaling_diag_matrix.T):
                    logger.warning("raw scaling_diag_matrix is not a symmetric matrix!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-3) * torch.eye(raw_scaling_diag_matrix.shape[0]).cuda()
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                if torch.isnan(scaling_diag_matrix).any():
                    logger.warning("scaling_diag_matrix contains NaN!")
                elif torch.isinf(scaling_diag_matrix).any():
                    logger.warning("scaling_diag_matrix contains Inf!")
                del eigenvalues
                subset[name].scaling_diag_matrix = scaling_diag_matrix
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                logger.warning("scaling_diag_matrix is not full rank!")
                reg_inv =  1e-3 * torch.eye(scaling_diag_matrix.shape[0]).cuda() 
                scaling_diag_matrix += reg_inv
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                del reg_inv
            
            del scaling_matrix_inv
            layer_scaling_matrices[name] = scaling_diag_matrix.cpu()
            torch.cuda.empty_cache()
        scaling_matrices.append(layer_scaling_matrices)
        layers[i] = layer.cpu()
        inps = outs
        torch.cuda.empty_cache()
        
    model.config.use_cache = use_cache
    if args.use_cache:
        torch.save(scaling_matrices, cache_file)
        logger.info(f"Save the whiten scale matrix dict to:  {cache_file}")

@torch.no_grad()
def get_whiten_scale_matrix_v2(model, tokenizer, args, dev):
    model_id = model.config._name_or_path
    #NOTE (brian1009): Might need to check the random seed, currently we have < 0.1 perplexity difference at Llama2-7B
    calib_loader = get_calib_data(
        "wikitext2", 
        tokenizer, 
        model_id, 
        nsamples=256, 
        seqlen=2048
    )
    cache_file = f"cache/whiten/{model_id.replace('/','_')}_w2_scaling_matrices_fp16_v2.pt"
    os.makedirs("cache/whiten", exist_ok=True)
    """
    cache format:
    [
        {
            "attn.q_proj": torch.Tensor,
            "attn.k_proj": torch.Tensor,
            "attn.v_proj": torch.Tensor,
            "attn.o_proj": torch.Tensor,
            "mlp.gate_proj": torch.Tensor,
            "mlp.up_proj": torch.Tensor,
            "mlp.down_proj": torch.Tensor
        },
        ... (stacked n times, in the order of model layers)
    ]
    """
    logger.info(f"[whiten] Calibration dataset: {args.calib_dataset}", fg="yellow")
    logger.info(f"[whiten] Search cache_file={cache_file}", fg="yellow")

    if os.path.exists(cache_file) and args.use_cache:
        logger.info(f"[whiten] File {cache_file} exist.", fg="green")
        logger.info(f"[whiten] Load scaling diag matrix from cache: {cache_file}", fg="yellow")
        scaling_matrics = torch.load(cache_file, map_location="cpu")


        layers = model.model.layers
        for i in tqdm(range(len(layers))):
            layer = layers[i]
            subset = find_layers(layer) # Collect all linear layers
            for name in subset:
                if name in scaling_matrics[i]:
                    scaling_diag_matrix = scaling_matrics[i][name]
                    subset[name].scaling_diag_matrix = scaling_diag_matrix

        return 
    
    logger.info(f"No cache_file={cache_file}", fg="red")
    logger.info(f"Create whiten scale matrix dict...", fg="yellow")

    # Create Scaling Matrix with low-resource inference
    # Adapted from https://github.com/AIoT-MLSys-Lab/SVD-LLM/blob/main/SVDLLM.py
    # Here, inference are performed in an layer-wise manner.
    use_cache = model.config.use_cache
    model.config.use_cache = False
    #FIXME: This is not a good implementation...
    if "llama" in model_id or "mistral" in model_id or "vicuna" in model_id or "longchat":
        layers = model.model.layers
    elif "opt" in model_id:
        layers = model.model.decoder.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    # inps = torch.zeros(
    #     (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "position_embeddings": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
                cache["position_embeddings"] = kwargs.get("position_embeddings", None)
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    scaling_matrices = []
    logger.info("[Decomposition] Start to calculate the scaling matrix in layer-wise manner...")
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks, position_ids=position_ids[0].unsqueeze(0), position_embeddings=position_embeddings)[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        layer_scaling_matrices = {}


        for name in subset:
            if not ("k_proj" in name or "v_proj" in name):
                continue
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().cuda()

            # new code

            U, S, Vt = torch.linalg.svd(raw_scaling_diag_matrix, full_matrices=False)
            sqrtSigma = torch.sqrt(torch.diag(S))
            scaling_diag_matrix = torch.matmul(U, sqrtSigma)

            subset[name].scaling_diag_matrix = scaling_diag_matrix
            layer_scaling_matrices[name] = scaling_diag_matrix.cpu()
            torch.cuda.empty_cache()
        scaling_matrices.append(layer_scaling_matrices)
        layers[i] = layer.cpu()
        inps = outs
        torch.cuda.empty_cache()
        
    model.config.use_cache = use_cache
    if args.use_cache:
        torch.save(scaling_matrices, cache_file)
        logger.info(f"Save the whiten scale matrix dict to:  {cache_file}")

@torch.no_grad()
def compress_model_whiten(model, tokenizer, args, dev, selection_result):
    logger.info("Compressing model with whiten decomposition...")
    # NOTE(brian1009): Prepare whiten scaling matrix
    get_whiten_scale_matrix(model, tokenizer, args, dev)
    # Compress the model
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    logger.info(f"Start decompose the layer with selected ranks... #target layers: {len(selection_result.keys())}")
    for layername, selected_head_rank in tqdm(selection_result.items()):
        logger.debug(f"Decompose {layername} with ranks: {selected_head_rank}")
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]

        #     raw_linear,
        #     selected_head_rank
        # if selected_head_rank[0] == 512:
        #     continue
        head_wise_svd_linear = HeadwiseLowRankModule.from_linear_whiten_pre(
            raw_linear,
            selected_head_rank
        )
        # LlamaAttention Module
        
        # k_proj or v_proj
        setattr(info["father"], info["name"],  head_wise_svd_linear)

@torch.no_grad()
def compress_model_ours(model_name, model, dataloader, tokenizer, args, dev, selection_result):
    logger.info("Compressing model with whiten decomposition...")
    # get_whiten_scale_matrix(model, tokenizer, args, dev)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), args.model_seq_len, model.config.hidden_size), dtype=dtype, device=dev
    )   
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "position_embeddings": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            cache['position_embeddings'] = kwargs.get("position_embeddings", None)
            raise ValueError 
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()    
    if 'opt' in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    position_embeddings = cache['position_embeddings']
   
    if "opt" not in model_name:
        position_ids = cache['position_ids'] 
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        kv = {}
        for name in subset:
            if "k_proj" in name or "v_proj" in name:
                kv[name] = BatchUpdater()
        def add_batch(name):
            def tmp(_, inp, out):
                kv[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in kv:
            handles.append(subset[name].register_forward_hook(add_batch(name))) 
            

        for j in range(0, inps.shape[0], 8):
            attn = attention_masks[j:j+8].to(dev) if attention_masks is not None else None
            pos_ids = position_ids[j:j+8].to(dev) if position_ids is not None else None

            if "opt" not in model_name:
                
                outs[j:j+8] = layer(inps[j:j+8], attention_mask=attn, position_ids=pos_ids, position_embeddings=position_embeddings)[0]
            else:
                outs[j:j+8] = layer(inps[j:j+8], attention_mask=attn)[0]
     
        for h in handles:
            h.remove()  
            
        for original_key in kv:
            new_name = f"model.layers.{i}.{original_key}"
            if  "v_proj" in new_name:
                total_rank = sum(selection_result[new_name])
                selected_head_rank = [total_rank]
                print(new_name)
                print(selected_head_rank)
                head_wise_svd_linear = HeadwiseLowRankModule.from_linear_adasvd(
                    subset[original_key],
                    selected_head_rank,
                    args.num_iter,
                    kv[original_key].inps,
                    kv[original_key].outs,
                )
            else:
                print(new_name)
                print(selection_result[new_name])

                #     selection_result[new_name],

                # HSR: reorder + grouped SVD on Keys. head_num is read from the
                # model config so GQA models (num_key_value_heads < num_attention_heads)
                # are handled correctly.
                head_wise_svd_linear = HeadwiseLowRankModule.from_linear_whiten_reorder(
                    subset[original_key],
                    selection_result[new_name],
                    head_num = model.config.num_key_value_heads,
                )
            
            parent_module = getattr(layer, "self_attn")
            attr_name = original_key.split('.')[-1]
            setattr(parent_module, attr_name, head_wise_svd_linear)
            
            
        layer = layer.to(dev)
        
        # for j in range(inps.shape[0]):
        #     if "opt" not in model_name:
        #     else:
        
        for j in range(inps.shape[0]):
            inp = inps[j].unsqueeze(0)
            attn = attention_masks[j].unsqueeze(0).to(dev) if attention_masks is not None else None
            
            # pos_ids = position_ids[j].unsqueeze(0).to(dev) if position_ids is not None else None
            
            if position_ids is not None:
                if position_ids.size(0) == 1:
                    pos_ids = position_ids[0].unsqueeze(0).to(dev)
                else:
                    pos_ids = position_ids[j].unsqueeze(0).to(dev)
            else:
                pos_ids = None

            if "opt" not in model_name:
                outs[j] = layer(inp, attention_mask=attn, position_ids=pos_ids, position_embeddings=position_embeddings)[0]
            else:
                outs[j] = layer(inp, attention_mask=attn)[0]

        layers[i] = layer.cpu()
        del kv
        torch.cuda.empty_cache()
        inps = outs
        # outs = None
        # del outs
    model.config.use_cache = use_cache

@torch.no_grad()
def compress_model_ours_reorder(model_name, model, dataloader, tokenizer, args, dev, selection_result):
    logger.info("Compressing model with whiten decomposition...")
    # get_whiten_scale_matrix(model, tokenizer, args, dev)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), args.model_seq_len, model.config.hidden_size), dtype=dtype, device=dev
    )   
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError 
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()    
    if 'opt' in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
   
    if "opt" not in model_name:
        position_ids = cache['position_ids'] 
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        kv = {}
        for name in subset:
            if "k_proj" in name or "v_proj" in name:
                kv[name] = BatchUpdater()
        def add_batch(name):
            def tmp(_, inp, out):
                kv[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in kv:
            handles.append(subset[name].register_forward_hook(add_batch(name))) 
            

        for j in range(0,inps.shape[0], 8):
            if "opt" not in model_name:
                outs[j:j+8] = layer(inps[j:j+8], attention_mask=attention_masks[j:j+8].to(dev), position_ids=position_ids[j:j+8].to(dev))[0]
            else:
                outs[j:j+8] = layer(inps[j:j+8], attention_mask=attention_masks[j:j+8].to(dev))[0]
     
        for h in handles:
            h.remove()  
            
        for original_key in kv:
            new_name = f"model.layers.{i}.{original_key}"
            if  "v_proj" in new_name:
                total_rank = sum(selection_result[new_name])
                selected_head_rank = [total_rank]
                print(new_name)
                print(selected_head_rank)
                head_wise_svd_linear = HeadwiseLowRankModule.from_linear_whiten_pre(
                    subset[original_key],
                    selected_head_rank,
                )
            else:
                print(new_name)
                print(selection_result[new_name])

                #     selection_result[new_name],

                # HSR: reorder + grouped SVD on Keys. head_num is read from the
                # model config so GQA models (num_key_value_heads < num_attention_heads)
                # are handled correctly.
                head_wise_svd_linear = HeadwiseLowRankModule.from_linear_whiten_reorder(
                    subset[original_key],
                    selection_result[new_name],
                    head_num = model.config.num_key_value_heads,
                )
            
            parent_module = getattr(layer, "self_attn")
            attr_name = original_key.split('.')[-1]
            setattr(parent_module, attr_name, head_wise_svd_linear)
            
            
                
        layer = layer.to(dev)
        for j in range(inps.shape[0]):
            if "opt" not in model_name:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
        layers[i] = layer.cpu()
        del kv
        torch.cuda.empty_cache()
        inps = outs
        # outs = None
        # del outs
    model.config.use_cache = use_cache

@torch.no_grad()
def compress_model_ours_calib(model_name, model, dataloader, tokenizer, args, dev, selection_result):
    logger.info("Compressing model with whiten decomposition...")
    # get_whiten_scale_matrix(model, tokenizer, args, dev)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), args.model_seq_len, model.config.hidden_size), dtype=dtype, device=dev
    )   
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError 
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()    
    if 'opt' in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
   
    if "opt" not in model_name:
        position_ids = cache['position_ids'] 
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        kv = {}
        for name in subset:
            if "k_proj" in name or "v_proj" in name:
                kv[name] = BatchUpdater()
        def add_batch(name):
            def tmp(_, inp, out):
                kv[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in kv:
            handles.append(subset[name].register_forward_hook(add_batch(name))) 
            

        for j in range(0,inps.shape[0], 8):
            if "opt" not in model_name:
                outs[j:j+8] = layer(inps[j:j+8], attention_mask=attention_masks[j:j+8].to(dev), position_ids=position_ids[j:j+8].to(dev))[0]
            else:
                outs[j:j+8] = layer(inps[j:j+8], attention_mask=attention_masks[j:j+8].to(dev))[0]
     
        for h in handles:
            h.remove()  
            
        for original_key in kv:
            new_name = f"model.layers.{i}.{original_key}"
            if  "v_proj" in new_name:
                total_rank = sum(selection_result[new_name])
                selected_head_rank = [total_rank]
                print(new_name)
                print(selected_head_rank)
                head_wise_svd_linear = HeadwiseLowRankModule.from_linear_adasvd(
                    subset[original_key],
                    selected_head_rank,
                    args.num_iter,
                    kv[original_key].inps,
                    kv[original_key].outs,
                )
            else:
                print(new_name)
                print(selection_result[new_name])

                #     selection_result[new_name],

                # reorder heads by CKA similarity (HSR)
                head_wise_svd_linear = HeadwiseLowRankModule.from_linear_whiten_pre(
                    subset[original_key],
                    selection_result[new_name],
                )
            
            parent_module = getattr(layer, "self_attn")
            attr_name = original_key.split('.')[-1]
            setattr(parent_module, attr_name, head_wise_svd_linear)
            
            
                
        layer = layer.to(dev)
        for j in range(inps.shape[0]):
            if "opt" not in model_name:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
        layers[i] = layer.cpu()
        del kv
        torch.cuda.empty_cache()
        inps = outs
        # outs = None
        # del outs
    model.config.use_cache = use_cache


@torch.no_grad()
def compress_model_ours_baseline(model_name, model, dataloader, tokenizer, args, dev, selection_result):
    logger.info("Compressing model with whiten decomposition...")
    # get_whiten_scale_matrix(model, tokenizer, args, dev)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), args.model_seq_len, model.config.hidden_size), dtype=dtype, device=dev
    )   
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError 
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()    
    if 'opt' in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
   
    if "opt" not in model_name:
        position_ids = cache['position_ids'] 
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        kv = {}
        for name in subset:
            if "k_proj" in name or "v_proj" in name:
                kv[name] = BatchUpdater()
        def add_batch(name):
            def tmp(_, inp, out):
                kv[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in kv:
            handles.append(subset[name].register_forward_hook(add_batch(name))) 
            

        for j in range(0,inps.shape[0], 8):
            if "opt" not in model_name:
                outs[j:j+8] = layer(inps[j:j+8], attention_mask=attention_masks[j:j+8].to(dev), position_ids=position_ids[j:j+8].to(dev))[0]
            else:
                outs[j:j+8] = layer(inps[j:j+8], attention_mask=attention_masks[j:j+8].to(dev))[0]
     
        for h in handles:
            h.remove()  
            
        for original_key in kv:
            new_name = f"model.layers.{i}.{original_key}"
            if  "v_proj" in new_name:
                total_rank = sum(selection_result[new_name])
                selected_head_rank = [total_rank]
                print(new_name)
                print(selected_head_rank)
                head_wise_svd_linear = HeadwiseLowRankModule.from_linear_whiten_pre(
                    subset[original_key],
                    selected_head_rank,
                )
            else:
                print(new_name)
                print(selection_result[new_name])

                #     selection_result[new_name],

                # reorder heads by CKA similarity (HSR)
                head_wise_svd_linear = HeadwiseLowRankModule.from_linear_whiten_pre(
                    subset[original_key],
                    selection_result[new_name],
                )
            
            parent_module = getattr(layer, "self_attn")
            attr_name = original_key.split('.')[-1]
            setattr(parent_module, attr_name, head_wise_svd_linear)
            
            
                
        layer = layer.to(dev)
        for j in range(inps.shape[0]):
            if "opt" not in model_name:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
        layers[i] = layer.cpu()
        del kv
        torch.cuda.empty_cache()
        inps = outs
        # outs = None
        # del outs
    model.config.use_cache = use_cache

class BatchUpdater:
    def __init__(self):
        self.inps = []
        self.outs = []
    
    def add_batch_update_u(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        self.inps.append(inp.sum(dim=0))
        self.outs.append(out.sum(dim=0))
        del inp, out
        torch.cuda.empty_cache()
     
# Wrapper for different decompose methods
def compress_model(model, tokenizer, args, dev, selection_result, permutation_result=None, head_num=-1):
    if args.decompose_method == "whiten":
        # Palu baseline: whitening + grouped SVD (no HSR, no OVC).
        compress_model_whiten(model, tokenizer, args, dev, selection_result)
    elif args.decompose_method == 'ours':
        # ReCalKV: HSR (Keys) + OVC (Values).
        get_whiten_scale_matrix_v2(model, tokenizer, args, dev)
        dataloader, _ = get_loaders(args.updating_dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        compress_model_ours(args.model_id, model, dataloader, tokenizer, args, dev, selection_result)
    elif args.decompose_method == 'ours_reorder':
        get_whiten_scale_matrix_v2(model, tokenizer, args, dev)
        dataloader, _ = get_loaders(args.calib_dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        compress_model_ours_reorder(args.model_id, model, dataloader, tokenizer, args, dev, selection_result)
    elif args.decompose_method == 'ours_calib':
        get_whiten_scale_matrix_v2(model, tokenizer, args, dev)
        dataloader, _ = get_loaders(args.calib_dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        compress_model_ours_calib(args.model_id, model, dataloader, tokenizer, args, dev, selection_result)

    elif args.decompose_method == 'ours_baseline':
        get_whiten_scale_matrix_v2(model, tokenizer, args, dev)
        dataloader, _ = get_loaders(args.calib_dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        compress_model_ours_baseline(args.model_id, model, dataloader, tokenizer, args, dev, selection_result)

    else:
        raise ValueError(f"Decomposition method {args.decompose_method} is not supported.")
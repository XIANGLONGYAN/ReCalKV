from .model.modules import HeadwiseLowRankModule
import torch.nn as nn

def configure_permutation(
        model: nn.Module, 
        config: dict,
    ):
    perm_dict = getattr(config, "permutation_info", {})
    for name, module in model.named_modules():
        if isinstance(module, HeadwiseLowRankModule):
            perm_info = perm_dict.get(name, None)
            if perm_info is not None:
                if isinstance(perm_info, dict):
                    perm_info = {int(k): v for k, v in perm_info.items()}
                module.permutation_info = perm_info
                module.permutation = True
                module.head_num = max(perm_info.keys()) + 1
import math
import torch as th
import torch.nn as nn

from typing import List, Tuple, Optional


"""
Credits for `HuggingFace transformers` library.
"""
class EinLinear(nn.Module):
    def __init__(self, n_models, in_features, out_features, bias):
        super().__init__()
        self.n_models = n_models
        self.out_features = out_features
        self.in_features = in_features
        self.weight = nn.Parameter(th.Tensor(n_models, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(th.Tensor(n_models, out_features))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        for i in range(self.n_models):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, input):
        """
        Args:
            input (`th.FloatTensor` of shape `(B, n_models, input_dim)`):
                The input to the layer.
        """
        # [ batch_size x n_models x output_dim ]
        output = th.einsum("eoi,bei->beo", self.weight, input)
        if self.bias is not None:
            raise RuntimeError()
        return output

    def extra_repr(self) -> str:
        return (f"n_models={self.n_models}, "
                f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


def configure_optimizer(model: nn.Module, 
                        learning_rate: float, 
                        weight_decay: float,
                        betas: Tuple[float, float],
                        blacklist_names: List[str] = [],
                        submodel: Optional[nn.Module] = None) -> th.optim.AdamW:
    """
    Separate the model parameters into decay and no_decay groups through the blacklist_names and other heuristics.
    Then return an AdamW optimizer with the decaying and non-decaying parameters.

    Args:
        model (nn.Module): The model to configure the optimizer for.
        learning_rate (float): The learning rate.
        weight_decay (float): The weight decay.
        blacklist_names (List[str]): The names of the modules to exclude from weight decay.

    Returns:
        (th.optim.AdamW): The optimizer.
    """
    decay, no_decay = set(), set()
    whitelist_modules = (nn.Linear, nn.Conv2d, EinLinear)
    blacklist_modules = (nn.LayerNorm, nn.Embedding)

    # classify the parameters into decay and no_decay
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters():
            full_param_name = f"{module_name}.{param_name}" if module_name else param_name
            
            if any([full_param_name.startswith(black_m_name) for black_m_name in blacklist_names]):
                # if parameters are in the blacklist names
                no_decay.add(full_param_name)
            elif 'bias' in param_name:
                # we exclude bias from weight decay
                no_decay.add(full_param_name)
            elif param_name.endswith("weight"):
                if isinstance(module, whitelist_modules):
                    decay.add(full_param_name)
                elif isinstance(module, blacklist_modules):
                    no_decay.add(full_param_name)

    # Special cases for the position embedding and the latent transformer in the VQ-VAE
    if hasattr(model, "pos_emb"):
        no_decay.add("pos_emb")

    if submodel is not None and hasattr(submodel, "pos_emb"):
        assert hasattr(model, "model"), "submodel should have a `model` attribute"
        no_decay.add("model.pos_emb")
        if submodel.bottleneck == "attention":
            no_decay.add("model.latent_pooling.query")  # Random query projection for AsymAttentio = AsymAttention.query
            no_decay.add("model.expand.query")          # AsymAttention.query
            no_decay.add("model.latent_pooling.attention.in_proj_weight")   # input projection weights for MultiheadAttention
            no_decay.add("model.expand.attention.in_proj_weight")           # MultiheadAttention.in_proj_weight

    param_dict = {param_name: param for param_name, param in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Parameters {str(inter_params)} should not appear in both decay and no_decay"
    assert len(param_dict.keys() - union_params) == 0, f"Parameters {str(param_dict.keys() - union_params)} were not separated into decay or no_decay"

    # Specify the AdamW weight decay for each parameter group
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}
    ]

    optimizer = th.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer


def init_weights(module: nn.Module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)

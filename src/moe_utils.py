import torch
from safetensors.torch import save_file, load_file
import os
import json
from typing import Dict, List, Tuple
from pathlib import Path
import torch.nn as nn


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _iter_custom_lora_param_names(model):
    for n, p in model.named_parameters():
        if ".custom_lora." in n:
            yield n, p

def save_injected_adapters(
    model,
    save_dir: str,
    include_frozen_ab: bool = False,
    extra_config: dict | None = None,
):
    """
    include_frozen_ab=False:
        saves ONLY trainable adapter params (router/mixture/etc), very small.
        Requires re-injection (with frozen A/B) before loading for inference.

    include_frozen_ab=True:
        saves ALL parameters under .custom_lora. (including frozen A/B).
        Larger, but more self-contained (still requires re-injection to create modules).
    """
    os.makedirs(save_dir, exist_ok=True)

    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}

    # Save params
    tensors = {}
    if include_frozen_ab:
        # everything under custom_lora (params + buffers)
        for k, v in model.state_dict().items():
            if ".custom_lora." in k:
                tensors[k] = v.detach().cpu()
    else:
        # trainable-only under custom_lora
        for n, p in _iter_custom_lora_param_names(model):
            if p.requires_grad:
                tensors[n] = p.detach().cpu()
        # also include any buffers under custom_lora (rare, but safe)
        for n, b in model.named_buffers():
            if ".custom_lora." in n:
                tensors[n] = b.detach().cpu()

    save_file(tensors, os.path.join(save_dir, "adapter_model.safetensors"))

    cfg = {
        "format": "custom_lora_only",
        "include_frozen_ab": include_frozen_ab,
        "trainable_param_count": sum(p.numel() for n, p in _iter_custom_lora_param_names(model) if p.requires_grad),
    }
    if extra_config:
        cfg.update(extra_config)

    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # print(f"[ok] Saved {len(tensors)} tensors to {save_dir}/adapter_model.safetensors")
    
def load_injected_adapters(model, load_dir: str, strict: bool = False):
    path = os.path.join(load_dir, "adapter_model.safetensors")
    sd = load_file(path)  # CPU tensors
    incompatible = model.load_state_dict(sd, strict=strict)

    # Avoid printing gigantic missing key lists (base model keys).
    missing_custom = [k for k in incompatible.missing_keys if ".custom_lora." in k]
    unexpected_custom = [k for k in incompatible.unexpected_keys if ".custom_lora." in k]
    if missing_custom or unexpected_custom:
        print("[warn] Incompatible custom adapter keys:")
        if missing_custom:
            print("  missing (custom):", missing_custom[:20], "..." if len(missing_custom) > 20 else "")
        if unexpected_custom:
            print("  unexpected (custom):", unexpected_custom[:20], "..." if len(unexpected_custom) > 20 else "")
    else:
        print("[ok] Loaded adapter tensors cleanly.")

    return incompatible

def _collect_passage_paths(trained_data_adapters_dir: str) -> List[Path]:
    root = Path(trained_data_adapters_dir)
    assert root.exists(), f"Adapters dir not found: {root}"
    passages = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("passage_")],
                      key=lambda p: int(p.name.split("_")[-1]))
    assert len(passages) > 0, f"No passage_* folders found under {root}"
    return passages

def _load_all_adapter_states(passages: List[Path]) -> List[Dict[str, torch.Tensor]]:
    states = []
    for p in passages:
        st_path = p / "adapter_model.safetensors"
        assert st_path.exists(), f"Missing {st_path}"
        states.append(load_file(str(st_path)))
    return states

def _find_weights_for_module(
    module_name: str,
    adapter_states: List[Dict[str, torch.Tensor]],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    module_name example: 'model.layers.0.mlp.down_proj'
    Match keys ending with '.{module_name}.lora_A.weight' and '.lora_B.weight'
    """
    suffix_A = f".{module_name}.lora_A.weight"
    suffix_B = f".{module_name}.lora_B.weight"

    A_list, B_list = [], []
    for st in adapter_states:
        kA = next((k for k in st.keys() if k.endswith(suffix_A)), None)
        kB = next((k for k in st.keys() if k.endswith(suffix_B)), None)
        if kA is None or kB is None:
            # fallback: drop leading 'model.' if present
            alt_suffix_A = "." + module_name.split("model.", 1)[-1] + ".lora_A.weight"
            alt_suffix_B = "." + module_name.split("model.", 1)[-1] + ".lora_B.weight"
            kA = next((k for k in st.keys() if k.endswith(alt_suffix_A)), kA)
            kB = next((k for k in st.keys() if k.endswith(alt_suffix_B)), kB)
        if kA is None or kB is None:
            return [], []
        A_list.append(st[kA])
        B_list.append(st[kB])
    return A_list, B_list

def _replace_module(model: nn.Module, dotted_name: str, new_mod: nn.Module):
    parent_name, attr = dotted_name.rsplit('.', 1)
    parent = dict(model.named_modules())[parent_name]
    setattr(parent, attr.split(':')[0], new_mod)

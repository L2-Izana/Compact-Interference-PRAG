import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from utils import model_generate

FORCE_DEVICE = torch.device("cuda:0")

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out
    
class MixtureLoRA(nn.Module):
    def __init__(self, in_dim, out_dim, trained_lora_A_weights, trained_lora_B_weights, r=2, alpha=32):
        super().__init__()
        self.num_heads = len(trained_lora_A_weights)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # load frozen A/B
        self.A = nn.Linear(in_dim, r * self.num_heads, bias=False) 
        self.B = nn.Linear(r * self.num_heads, out_dim, bias=False) 
        
        A_stacked = torch.cat(trained_lora_A_weights, dim=0)      # [r*num_heads, in_dim]
        B_stacked = torch.cat(trained_lora_B_weights, dim=1)    # [out_dim, r*num_heads]
        with torch.no_grad():
            self.A.weight.copy_(A_stacked)
            self.B.weight.copy_(B_stacked)

        # trainable mixture: maps (num_heads*r) → (num_heads*r)
        self.mixture = nn.Linear(self.num_heads * r, self.num_heads * r)

        # freeze A/B
        for p in self.A.parameters(): p.requires_grad = False
        for p in self.B.parameters(): p.requires_grad = False
        # train mixture only
        for p in self.mixture.parameters(): p.requires_grad = True

    def forward(self, x):
        return self.scaling * self.B(self.mixture(self.A(x)))

# --------------------------------------------------------------------------
# Wrapper Linear (base frozen; inference-mode; add MoE delta)
# --------------------------------------------------------------------------------
class LinearWithCustomLoRA(nn.Module):
    def __init__(self, base_linear: nn.Linear, custom_lora_adapter):
        super().__init__()
        self.base = base_linear
        self.custom_lora = custom_lora_adapter
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        # crucial: avoid storing base activations (prevents 7→40 GB spike)
        with torch.no_grad():
            base_out = self.base(x)
        return base_out + self.custom_lora(x)

# --------------------------------------------------------------------------------
# Injection utilities
# --------------------------------------------------------------------------------
TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]

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

def inject_hydra_lora_force_cuda0(model: nn.Module, trained_data_adapters_dir: str, r: int = 2, alpha: int = 32, target_modules: List[str]=None, verbose: bool = True, architecture="moe_mixture"):
    """
    Injects MoE-LoRA into all target Linear modules and places them on cuda:0.
    Assumes we will move the *entire model* to cuda:0 as well (see main).
    """
    if target_modules is None:
        target_modules = TARGET_MODULES

    passages = _collect_passage_paths(trained_data_adapters_dir)
    adapter_states = _load_all_adapter_states(passages)
    if verbose:
        print(f"Found {len(adapter_states)} passages: {[p.name for p in passages]}")

    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False

    injected = 0
    modules = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for name, module in modules:
        if not any(t in name for t in target_modules):
            continue

        A_list, B_list = _find_weights_for_module(name, adapter_states)
        if len(A_list) == 0:
            if verbose:
                print(f"[skip] No matching LoRA weights for {name}")
            continue

        in_dim, out_dim = module.in_features, module.out_features
        for i, (A, B) in enumerate(zip(A_list, B_list)):
            assert A.shape[1] == in_dim, f"{name} passage#{i} A shape {A.shape} incompatible with in_dim={in_dim}"
            assert B.shape[0] == out_dim, f"{name} passage#{i} B shape {B.shape} incompatible with out_dim={out_dim}"
            assert A.shape[0] == B.shape[1], f"{name} passage#{i} r mismatch: {A.shape[0]} vs {B.shape[1]}"

        if architecture == "moe":
            moe = MoELoRA(
                in_dim=in_dim,
                out_dim=out_dim,
                trained_lora_A_weights=A_list,
                trained_lora_B_weights=B_list,
                r=(A_list[0].shape[0] if r is None else r),
                alpha=alpha,
            )
        elif architecture == "mixture":
            moe = MixtureLoRA(
                in_dim=in_dim,
                out_dim=out_dim,
                trained_lora_A_weights=A_list,
                trained_lora_B_weights=B_list,
                r=(A_list[0].shape[0] if r is None else r),
                alpha=alpha,
            )
        elif architecture == "moe_subspace":
            moe = MoESubspaceLoRA(
                in_dim=in_dim,
                out_dim=out_dim,
                trained_lora_A_weights=A_list,
                trained_lora_B_weights=B_list,
                r=(A_list[0].shape[0] if r is None else r),
                alpha=alpha,
            )
        elif architecture == "moe_mixture":
            moe = MoEMixtureLoRA(
                in_dim=in_dim,
                out_dim=out_dim,
                trained_lora_A_weights=A_list,
                trained_lora_B_weights=B_list,
                r=(A_list[0].shape[0] if r is None else r),
                alpha=alpha,
            )
        elif architecture == "moe_mixture_subspace":
            moe = MoEMixtureSubspaceLoRA(
                in_dim=in_dim,
                out_dim=out_dim,
                trained_lora_A_weights=A_list,
                trained_lora_B_weights=B_list,
                r=(A_list[0].shape[0] if r is None else r),
                alpha=alpha,
            )
        elif architecture == "advanced_moe_mixture":
            moe = AdvancedMoEMixtureLoRA(
                in_dim=in_dim,
                out_dim=out_dim,
                trained_lora_A_weights=A_list,
                trained_lora_B_weights=B_list,
                r=(A_list[0].shape[0] if r is None else r),
                alpha=alpha,
            )
        # Force MoE placement to cuda:0 (ignore shard)
        moe.to(device=FORCE_DEVICE, dtype=module.weight.dtype)

        wrapped = LinearWithCustomLoRA(module, moe)
        _replace_module(model, name, wrapped)
        injected += 1
        if verbose:
            print(f"[ok] Injected MoEMixtureLoRA into {name} (heads={len(A_list)}, r={moe.r}, alpha={alpha}, device={FORCE_DEVICE})")

    if verbose:
        print(f"Done. Injected {injected} modules.")
    print(f"[INFP] Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# --------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------
def train(
    args,
    question,
    augments,
    model,
    tokenizer,
    config
):
    import torch, os, json
    from encode import TrainingData, TrainingDataCollator
    from prompt_template import get_prompt
    from utils import model_generate  # assuming you have these
    from tqdm import tqdm

    def get_closed_book_QA(aug_model, augments, tokenizer):
        prompt_ids = []
        for aug in augments:
            qas = aug[f"{aug_model}_qa"]
            for qa in qas:
                prompt_ids.append(get_prompt(tokenizer, qa["question"], None, qa["answer"]))
        return prompt_ids

    # Prepare training data
    prompt_ids = get_closed_book_QA("qwen2.5-1.5b-instruct", augments, tokenizer)
    train_data = TrainingData(prompt_ids, tokenizer)
    collator = TrainingDataCollator(tokenizer, FORCE_DEVICE)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        collate_fn=collator,
        shuffle=False,
    )

    # Model setup
    model.config.use_cache = False
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(pbar):
            batch = move_batch_to_device(batch, torch.device("cuda:0"))
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (step + 1)})

        # 🔍 Generate after each epoch
        model.eval()
        print(f"\nModel output after epoch {epoch+1}:")
        try:
            answer = model_generate(question, model, tokenizer, config)
        except Exception as e:
            print(f"[Warning] model_generate failed: {e}")
            continue
        print(f"Q: {question}\nA: {answer}\n{'-'*60}")

        # Optional: save intermediate checkpoints
        # epoch_dir = os.path.join(save_path, f"epoch_{epoch+1}")
        # os.makedirs(epoch_dir, exist_ok=True)
        # model.save_pretrained(epoch_dir)
    model.save_pretrained("/scratch/doluk/Compact-Interference-PRAG/temp_moe_adapter")  # <-- set your path
    return model

# --------------------------------------------------------------------------------
# Example main
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--architecture", type=str, choices=["moe", "mixture", "moe_subspace", "moe_mixture", "moe_mixture_subspace", "advanced_moe_mixture"], default="moe_subspace")
    ap.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = ap.parse_args()
    # 1) Load your HF model (keep accelerate device map intact if used there)
    from utils import get_model  # your project helper
    model, tokenizer, config = get_model(model_name="qwen2.5-1.5b-instruct")
    
    # 2) Inject adapters
    trained_adapter_dir = "/scratch/doluk/Compact-Interference-PRAG/offline/qwen2.5-1.5b-instruct/rank=2_alpha=32/popqa/lr=0.0003_epoch=2_direct/aug_model=qwen2.5-1.5b-instruct/total/data_0"  # <-- set your path
    inject_hydra_lora_force_cuda0(model, trained_data_adapters_dir=trained_adapter_dir, alpha=32, target_modules=["gate_proj", "up_proj", "down_proj"], architecture=args.architecture)
    model.to(device=FORCE_DEVICE)
    # 3) Load aug data
    data_file = "/scratch/doluk/Compact-Interference-PRAG/data_aug/popqa/qwen2.5-1.5b-instruct/total.json"  # <-- set your path
    with open(data_file, "r") as f:
        data = json.load(f)
    augments = data[0]["augment"]

    # 4) Train (router-only)
    torch.cuda.empty_cache()
    model = train(args, "What is George Rankin's occupation?", augments, model, tokenizer, config)
    # print(model_generate("What is George Rankin's occupation?", model, tokenizer, config))
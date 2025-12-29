import argparse
import json
from typing import List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from moe_utils import load_injected_adapters, move_batch_to_device, save_injected_adapters, _collect_passage_paths, _find_weights_for_module, _load_all_adapter_states, _replace_module
from utils import model_generate, setup_logging
import logging


DEVICE = torch.device("cuda:0")
TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]
TESTING_SAVE_PATH = "/scratch/doluk/Compact-Interference-PRAG/temp_moe_adapter"

logger = logging.getLogger(__name__)

class MoELoRA(nn.Module):
    def __init__(self, in_dim, out_dim, trained_lora_A_weights, trained_lora_B_weights, r=None, alpha=32):
        """
        trained_lora_A_weights/B_weights: list[Tensor] with shapes:
          A_i: (r, in_dim)   (down-proj)
          B_i: (out_dim, r)  (up-proj)
        """
        super().__init__()
        assert len(trained_lora_A_weights) == len(trained_lora_B_weights)
        self.num_heads = len(trained_lora_A_weights)

        inferred_r = trained_lora_A_weights[0].shape[0]
        if r is None:
            r = inferred_r
        else:
            logging.debug(f"trained_lora_A_weights shape: {trained_lora_A_weights[0].shape}")
            logging.debug(f"trained_lora_B_weights shape: {trained_lora_B_weights[0].shape}")
            assert r == inferred_r, f"Provided r={r} != inferred r={inferred_r}"

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # A_i: in_dim -> r ; B_i: r -> out_dim
        self.As = nn.ModuleList([nn.Linear(in_dim, r, bias=False) for _ in range(self.num_heads)])
        self.Bs = nn.ModuleList([nn.Linear(r, out_dim, bias=False) for _ in range(self.num_heads)])

        with torch.no_grad():
               for i in range(self.num_heads):
                # A: (r, in_dim) matches Linear(in_dim->r).weight
                self.As[i].weight.copy_(trained_lora_A_weights[i])
                # B: (out_dim, r) matches Linear(r->out_dim).weight
                self.Bs[i].weight.copy_(trained_lora_B_weights[i])

        # Router: trainable
        self.router = nn.Linear(in_dim, self.num_heads)

        # Freeze A/B; only router trains
        for p in self.As.parameters():
            p.requires_grad = False
        for p in self.Bs.parameters():
            p.requires_grad = False
        for p in self.router.parameters():
            p.requires_grad = True

    def forward(self, x):
        router_logits = self.router(x)  # [..., num_heads]
        router_scores = self.num_heads * F.softmax(router_logits, dim=-1)
        expert_outs = [self.Bs[i](self.As[i](x)) for i in range(self.num_heads)]
        expert_outs = torch.stack(expert_outs, dim=-1)  # [..., out_dim, num_heads]
        return self.scaling * torch.sum(expert_outs * router_scores.unsqueeze(-2), dim=-1)
    
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

class MoESubspaceLoRA(nn.Module):
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

        # trainable moe-subspace: maps input → (num_heads*r), instead of num_heads like in MoE
        self.moe_subspace = nn.Linear(in_dim, self.num_heads * r)

        # freeze A/B
        for p in self.A.parameters(): p.requires_grad = False
        for p in self.B.parameters(): p.requires_grad = False
        # train mixture only
        for p in self.moe_subspace.parameters(): p.requires_grad = True

    def forward(self, x):   # x: [B, S, in_dim]
        mixture_out = self.num_heads * F.softmax(self.moe_subspace(x), dim=-1)  # [B, S, num_heads*r] # scale by num_heads to keep magnitude
        logger.debug(f"mixture_out:", mixture_out)
        z = self.A(x)  # [B, S, num_heads*r]
        row_wise_product = z * mixture_out  # [B, S, num_heads*r]
        return self.scaling * self.B(row_wise_product)  # [B, S, out_dim]

class MoEMixtureLoRA(nn.Module):
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
        self.moe = nn.Linear(in_dim, self.num_heads)
        
        # freeze A/B
        for p in self.A.parameters(): p.requires_grad = False
        for p in self.B.parameters(): p.requires_grad = False
        # train mixture only
        for p in self.moe.parameters(): p.requires_grad = True
        for p in self.mixture.parameters(): p.requires_grad = True
        
    def forward(self, x):   # x: [B, S, in_dim]
        routings = self.num_heads * F.softmax(self.moe(x), dim=-1)  # [B, S, num_heads] # scale by num_heads to keep magnitude
        z = self.mixture(self.A(x))  # [B, S, num_heads*r]
        # This is MoE (not Subpsace Moe), so need to repeat routings for r dims
        row_wise_routings = routings.unsqueeze(-1).repeat(1, 1, 1, self.r).view(*z.shape)  # [B, S, num_heads*r]
        row_wise_product = z * row_wise_routings  # [B, S, num_heads*r]
        return self.scaling * self.B(row_wise_product)  # [B, S, out_dim]

class MoEMixtureSubspaceLoRA(nn.Module):
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
        self.moe_subspace = nn.Linear(in_dim, self.num_heads * r)
        
        # freeze A/B
        for p in self.A.parameters(): p.requires_grad = False
        for p in self.B.parameters(): p.requires_grad = False
        # train mixture only
        for p in self.moe_subspace.parameters(): p.requires_grad = True
        for p in self.mixture.parameters(): p.requires_grad = True
        
    def forward(self, x):   # x: [B, S, in_dim]
        routings = self.num_heads * F.softmax(self.moe_subspace(x), dim=-1)  # [B, S, num_heads*r] # scale by num_heads to keep magnitude
        z = self.mixture(self.A(x))  # [B, S, num_heads*r]
        row_wise_product = z * routings  # [B, S, num_heads*r]
        return self.scaling * self.B(row_wise_product)  # [B, S, out_dim]

class AdvancedMoEMixtureLoRA(nn.Module):
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
        self.mixture_moe = nn.Linear(in_dim, (self.num_heads * r)**2, bias=True)

        # freeze A/B
        for p in self.A.parameters(): p.requires_grad = False
        for p in self.B.parameters(): p.requires_grad = False
        # train mixture only
        for p in self.mixture_moe.parameters(): p.requires_grad = True

    def forward(self, x):   # x: [B, S, in_dim]
        z = self.A(x)  # [B, S, num_heads*r]
        M = self.num_heads * self.mixture_moe(x)  # [B, S, (num_heads*r)*(num_heads*r)]
        M = M.view(*z.shape, self.num_heads * self.r)  # [B, S, num_heads*r, num_heads*r]
        z_mixed = torch.einsum('bsij,bsj->bsi', M, z)  # [B, S, num_heads*r]
        # mixture_out = self.num_heads * F.softmax(self.mixture_moe(x), dim=-1).view(-1, self.num_heads * self.r, self.num_heads * self.r)  # [B, S, num_heads*r, num_heads*r] # scale by num_heads to keep magnitude
        # y = z @ z_mixed  # [B, S, num_heads*r]
        return self.num_heads * self.scaling * self.B(z_mixed)  # [B, S, out_dim]

# class MixtureSubspaceLoRA(nn.Module):
#     def __init__(self, in_dim, out_dim, trained_lora_A_weights, trained_lora_B_weights, r=2, alpha=32):
#         super().__init__()
#         self.num_heads = len(trained_lora_A_weights)
#         self.r = r
#         self.alpha = alpha
#         self.scaling = alpha / r

#         # load frozen A/B
#         self.A = nn.Linear(in_dim, r, bias=False) 
#         self.B = nn.Linear(r * self.num_heads, out_dim, bias=False)        

#         # A_avg = torch.mean(trained_lora_A_weights, dim=0)      # [r, in_dim]
#         B_stacked = torch.cat(trained_lora_B_weights, dim=1)    # [out_dim, r*num_heads]
#         with torch.no_grad():
#             self.A.weight.copy_(A_avg)
#             self.B.weight.copy_(B_stacked)

#         # trainable mixture: maps (num_heads*r) → (num_heads*r)
#         self.mixture_moe = nn.Linear(in_dim, self.num_heads * r)

#         # freeze A/B
#         for p in self.A.parameters(): p.requires_grad = True
#         for p in self.B.parameters(): p.requires_grad = False
#         # train mixture only
#         for p in self.mixture_moe.parameters(): p.requires_grad = True

#     def forward(self, x):   # x: [B, S, in_dim]
#         mixture_out = F.softmax(self.mixture_moe(x), dim=-1) # [B, S, num_heads * r] 
#         z = self.A(x)  # [B, S, r]
#         mixture_out * self.B  # [B, S, out_dim]
#         return self.num_heads * self.scaling * sum(outs)  # [B, S, out_dim]

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


def inject_hydra_lora(model: nn.Module, trained_data_adapters_dir: str, r: None, alpha: int = 32, target_modules: List[str]=TARGET_MODULES, architecture="moe_mixture"):
    """
    Injects MoE-LoRA into all target Linear modules and places them on cuda:0.
    Assumes we will move the *entire model* to cuda:0 as well (see main).
    """
    if target_modules is None:
        target_modules = TARGET_MODULES

    passages = _collect_passage_paths(trained_data_adapters_dir)
    adapter_states = _load_all_adapter_states(passages)
    logger.debug(f"Found {len(adapter_states)} passages: {[p.name for p in passages]}")

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
            logger.debug(f"[skip] No matching LoRA weights for {name}")
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
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        moe.to(device=DEVICE, dtype=module.weight.dtype)

        wrapped = LinearWithCustomLoRA(module, moe)
        _replace_module(model, name, wrapped)
        injected += 1
        logger.debug(f"[ok] Injected MoEMixtureLoRA into {name} (heads={len(A_list)}, r={moe.r}, alpha={alpha}, device={DEVICE})")

    logger.info(f"Done. Injected {injected} modules.")
    logger.info(f"[INFP] Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

def train(args, question, augments, model, tokenizer, config, save_path):
    from encode import TrainingData, TrainingDataCollator
    from prompt_template import get_prompt
    from utils import model_generate  
    from tqdm import tqdm

    def get_closed_book_QA(aug_model, augments, tokenizer):
        prompt_ids = []
        for aug in augments:
            qas = aug[f"{aug_model}_qa"]
            for qa in qas:
                prompt_ids.append(get_prompt(tokenizer, qa["question"], None, qa["answer"]))
        return prompt_ids

    # Prepare training data
    prompt_ids = get_closed_book_QA(args.augment_model, augments, tokenizer)
    train_data = TrainingData(prompt_ids, tokenizer)
    collator = TrainingDataCollator(tokenizer, DEVICE)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=collator,
        shuffle=False,
    )

    # Model setup
    model.config.use_cache = False
    if args.freeze_A:
        logging.debug(f"Total number of trainable parameters before freezing: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")
        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.requires_grad = False
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    logging.debug(f"Total number of trainable parameters: {sum([p.numel() for p in model_parameters])}")
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)

    # Training loop
    for epoch in range(args.num_post_train_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_post_train_epochs}")

        for step, batch in enumerate(pbar):
            batch = move_batch_to_device(batch, DEVICE)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (step + 1)})

        # Generate after each epoch
        model.eval()
        logger.debug(f"\nModel output after epoch {epoch+1}:")
        try:
            answer = model_generate(question, model, tokenizer, config)
        except Exception as e:
            logger.warning(f"[Warning] model_generate failed: {e}")
            continue
        logger.debug(f"Q: {question}\nA: {answer}\n{'-'*60}")
        epoch_save_path = os.path.join(save_path, f"{args.lora_architecture}", f"lr={args.learning_rate}_epoch={epoch+1}")
        logger.debug(f"Training done on epoch {epoch}, saving to {epoch_save_path}")    
        save_injected_adapters(
            model,
            epoch_save_path,
            include_frozen_ab=False,  
            extra_config={
                "architecture": args.lora_architecture,
                "alpha": args.lora_alpha,
                "r": args.lora_rank,
                "target_modules": TARGET_MODULES,
            },
        )
    return model


# --------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_architecture", type=str, choices=["moe", "mixture", "moe_subspace", "moe_mixture", "moe_mixture_subspace", "advanced_moe_mixture"], default="moe")
    ap.add_argument("--model_name", type=str, default="qwen2.5-1.5b-instruct")
    ap.add_argument("--lora_rank", type=int, default=2)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--num_post_train_epochs", type=int, default=10, help="Number of training epochs")
    ap.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--freeze_A", action="store_true")
    args = ap.parse_args()
    args.augment_model =  args.model_name
    
    # 1) Load your HF model (keep accelerate device map intact if used there)
    from utils import get_model  # your project helper
    model, tokenizer, config = get_model(model_name="qwen2.5-1.5b-instruct")
    
    # 2) Inject adapters
    trained_adapter_dir = f"/scratch/doluk/Compact-Interference-PRAG/offline/{args.model_name}/rank={args.lora_rank}_alpha=32/popqa/lr=0.0003_epoch=2_direct/aug_model=qwen2.5-1.5b-instruct/total/data_0"  # <-- set your path
    inject_hydra_lora(model, trained_data_adapters_dir=trained_adapter_dir, r=args.lora_rank, alpha=32, target_modules=TARGET_MODULES, architecture=args.lora_architecture)
    setup_logging(debug=args.debug)
    model.to(device=DEVICE)
    # 3) Load aug data
    data_file = "/scratch/doluk/Compact-Interference-PRAG/data_aug/popqa/qwen2.5-1.5b-instruct/total.json"  # <-- set your path
    with open(data_file, "r") as f:
        data = json.load(f)
    augments = data[0]["augment"]

    # 4) Train (router-only)
    torch.cuda.empty_cache()
    model = train(args, "What is George Rankin's occupation?", augments, model, tokenizer, config, TESTING_SAVE_PATH)
    
    print("Training complete, test with saved adapters")
    model, tokenizer, config = get_model(model_name="qwen2.5-1.5b-instruct")

    # 2) Re-inject adapters (loads frozen A/B from trained_adapter_dir)
    inject_hydra_lora(
        model,
        trained_data_adapters_dir=trained_adapter_dir,
        r=args.lora_rank,
        alpha=32,
        target_modules=TARGET_MODULES,
        architecture=args.lora_architecture,
    )
    model.to(device=DEVICE)
    model.eval()

    # 3) Load the trained router/mixture weights
    load_injected_adapters(model, os.path.join(TESTING_SAVE_PATH, args.lora_architecture,  f"lr={args.learning_rate}_epoch={args.num_post_train_epochs}"), strict=False)

    # 4) Run inference
    with torch.no_grad():
        print(model_generate("What is George Rankin's occupation?", model, tokenizer, config))
        
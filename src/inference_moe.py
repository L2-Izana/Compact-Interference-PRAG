import os
import gc
import json
import argparse
import torch
from tqdm import tqdm
import warnings
from transformers.utils import logging as hf_logging

from moe_architecture import inject_hydra_lora, train
from moe_utils import load_injected_adapters
import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, predict, load_data, read_complete, setup_logging
from encode_moe import need_post_train
hf_logging.set_verbosity_error()
import logging


warnings.filterwarnings(
    "ignore",
    message="`do_sample` is set to `False`.*",
)
TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]
DEVICE = torch.device("cuda:0")

def main(args):
    data_list = load_data(args.dataset, args.data_type, args.augment_model)

    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)
    
    cot_name = "cot" if args.with_cot else "direct"
    output_root_dir = os.path.join(
        ROOT_DIR, 
        "output" if not args.use_warmup else "warmup/output",
        "freezeA" if args.freeze_A else "",
        "warmup_cot" if args.with_cot_warmup else "",
        f"warmup_epoch{args.warmup_epoch}" if args.warmup_epoch > 1 else "",
        f"{args.lora_architecture}",
        args.model_name, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
        f"post_train_epochs={args.num_post_train_epochs}",
        f"lr={args.learning_rate}_{cot_name}"
    )
    prag_output_dir = os.path.join(output_root_dir, "prag")
    combine_output_dir = os.path.join(output_root_dir, "combine")
    
    for filename, fulldata in data_list:
        filename = filename.split(".")[0]
        logger.info(f"### Solving {filename} ###")
        prag_output_dir = os.path.join(prag_output_dir, filename)
        combine_output_dir = os.path.join(combine_output_dir, filename)
        logger.info(f"Output 2 inference methods prag and combine into directory {output_root_dir}")
        os.makedirs(output_root_dir, exist_ok=True)
        os.makedirs(prag_output_dir, exist_ok=True)
        os.makedirs(combine_output_dir, exist_ok=True)
        
        with open(os.path.join(prag_output_dir, "config.json"), "w") as fout:
            json.dump(vars(args), fout, indent=4)
        with open(os.path.join(combine_output_dir, "config.json"), "w") as fout:
            json.dump(vars(args), fout, indent=4)

        prag_predict_file = os.path.join(prag_output_dir, "predict.json")
        combine_predict_file = os.path.join(combine_output_dir, "predict.json")
        prag_ret, start_with = read_complete(prag_predict_file)
        combine_ret, start_with = read_complete(prag_predict_file)
        logging.info(f"Already processed {len(prag_ret)}, starting with test_id={start_with}")
        fulldata = fulldata[start_with:] if args.sample == -1 else fulldata[start_with:args.sample]
        save_every = max(len(fulldata)//30, 1)
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            test_id = test_id + start_with
            assert test_id == len(prag_ret), f"test_id {test_id} != len(ret) {len(prag_ret)}"
            assert test_id == len(combine_ret), f"test_id {test_id} != len(ret) {len(combine_ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]

            def get_pred(model, generation_config, psgs):
                text = predict(model, tokenizer, generation_config, 
                                        question, with_cot=args.with_cot, 
                                        passages=psgs)
                pred = {
                    "test_id": test_id, 
                    "question": question, 
                    "answer": answer,   
                    "text": text,
                }
                pred.update(evaluate(text, answer, args.with_cot))
                return pred

            trained_adapter_dir = os.path.join(
                ROOT_DIR,
                "PRAG" if args.use_warmup else "", 
                "offline" if not args.use_warmup else "warmup/offline",
                "warmup_cot" if args.with_cot_warmup else "",
                f"warmup_epoch{args.warmup_epoch}" if args.warmup_epoch > 1 else "",
                args.model_name,
                f"rank={args.lora_rank}_alpha={args.lora_alpha}",
                args.dataset,
                f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
                f"aug_model={args.augment_model}",
                filename,
                f"data_{test_id}",
            )
            logging.debug(f"Trained adapter dir: {trained_adapter_dir}")
            assert os.path.exists(trained_adapter_dir), f"Trained adapter dir not found: {trained_adapter_dir}"
            assert not need_post_train(args, trained_adapter_dir=trained_adapter_dir), f"Unfinished post-trained, recheck this {trained_adapter_dir}"
            logging.info(f"Start evaluation for test_id {test_id}")
            prag_data_pred = {}
            combine_data_pred = {}
            for epoch in range(args.num_post_train_epochs):  # Full inference on 1,..,post_epochs
                logging.debug(f"For epoch {epoch+1}")
                model, tokenizer, config = get_model(model_name=args.model_name)
                model.to(device=DEVICE)
                inject_hydra_lora(
                    model,
                    trained_data_adapters_dir=trained_adapter_dir,
                    r=args.lora_rank,
                    alpha=args.lora_alpha,
                    target_modules=TARGET_MODULES,
                    architecture=args.lora_architecture,
                )
                load_injected_adapters(model, os.path.join(trained_adapter_dir, args.lora_architecture,  f"lr={args.learning_rate}_epoch={epoch+1}"), strict=False)
                model.eval()
                prag_data_pred[epoch] = get_pred(model, config, psgs=None)
                combine_data_pred[epoch] = get_pred(model, config, passages)
                gc.collect()
                torch.cuda.empty_cache()
            prag_ret.append(prag_data_pred)
            combine_ret.append(combine_data_pred)
            assert len(prag_ret) == len(combine_ret), f"Inpersistent length of prag_ret: {len(prag_ret)} and combine_ret: {len(combine_ret)}"
            if (len(prag_ret) % save_every) == 0:
                with open(prag_predict_file, "w") as fout:
                    json.dump(prag_ret, fout, indent=4)
                with open(combine_predict_file, "w") as fout:
                    json.dump(combine_ret, fout, indent=4)
                metrics = ["em", "f1", "prec", "recall"]
                prag_ret_str = ""
                combine_ret_str = ""
                for epoch in range(args.num_post_train_epochs):
                    prag_ret_epoch = [d[epoch] for d in prag_ret]
                    combine_ret_epoch = [d[epoch] for d in combine_ret]
                    prag_ret_str += f"Epoch {epoch}\n"
                    combine_ret_str += f"Epoch {epoch}\n"
                    for met in metrics:
                        prag_acc = sum(float(d[met]) for d in prag_ret_epoch) / len(prag_ret_epoch)
                        combine_acc = sum(float(d[met]) for d in combine_ret_epoch) / len(combine_ret_epoch)
                        prag_ret_str += f"\t{met}\t{ round(prag_acc, 4)}\n"
                        combine_ret_str += f"\t{met}\t{ round(combine_acc, 4)}\n"
                prag_ret_str += "\n" + json.dumps(vars(args), indent=4)
                combine_ret_str += "\n" + json.dumps(vars(args), indent=4)
                with open(os.path.join(prag_output_dir, "result.txt"), "w") as fout:
                    fout.write(prag_ret_str)
                with open(os.path.join(combine_output_dir, "result.txt"), "w") as fout:
                    fout.write(combine_ret_str)
                torch.cuda.empty_cache()
                
        with open(prag_predict_file, "w") as fout:
            json.dump(prag_ret, fout, indent=4)
        with open(combine_predict_file, "w") as fout:
            json.dump(combine_ret, fout, indent=4)
        metrics = ["em", "f1", "prec", "recall"]
        prag_ret_str = ""
        combine_ret_str = ""
        for epoch in range(args.num_post_train_epochs):
            prag_ret_epoch = [d[epoch] for d in prag_ret]
            combine_ret_epoch = [d[epoch] for d in combine_ret]
            prag_ret_str += f"Epoch {epoch}\n"
            combine_ret_str += f"Epoch {epoch}\n"
            for met in metrics:
                prag_acc = sum(float(d[met]) for d in prag_ret_epoch) / len(prag_ret_epoch)
                combine_acc = sum(float(d[met]) for d in combine_ret_epoch) / len(combine_ret_epoch)
                prag_ret_str += f"\t{met}\t{ round(prag_acc, 4)}\n"
                combine_ret_str += f"\t{met}\t{ round(combine_acc, 4)}\n"
        prag_ret_str += "\n" + json.dumps(vars(args), indent=4)
        combine_ret_str += "\n" + json.dumps(vars(args), indent=4)
        with open(os.path.join(prag_output_dir, "result.txt"), "w") as fout:
            fout.write(prag_ret_str)
        with open(os.path.join(combine_output_dir, "result.txt"), "w") as fout:
            fout.write(combine_ret_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ---- existing args ----
    parser.add_argument("--model_name", type=str, default="qwen2.5-1.5b-instruct")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="popqa")
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--augment_model", type=str, default=None)

    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--num_post_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    parser.add_argument("--lora_architecture", type=str,
                        choices=[
                            "moe", "mixture", "moe_subspace",
                            "moe_mixture", "moe_mixture_subspace",
                            "advanced_moe_mixture"
                        ],
                        default="moe")

    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=32)

    parser.add_argument("--use_warmup", action="store_true")
    parser.add_argument("--freeze_A", action="store_true")
    parser.add_argument("--warmup_epoch", type=int, default=1)
    parser.add_argument("--with_cot_warmup", action="store_true")

    # ---- logging args ----
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log_file", type=str, default="logs/main.log")

    args = parser.parse_args()

    setup_logging(debug=args.debug, log_file=args.log_file)

    logger = logging.getLogger(__name__)
    logger.info("Arguments: %s", vars(args))

    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"

    if args.augment_model is None:
        args.augment_model = args.model_name

    if (args.freeze_A or args.with_cot_warmup) and not args.use_warmup:
        raise ValueError("Warmup settings require --use_warmup")

    main(args)

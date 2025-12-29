import os
import gc
import json
import argparse
import torch
from tqdm import tqdm
import warnings
from transformers.utils import logging as hf_logging

from moe_architecture import inject_hydra_lora, train
import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, predict, load_data, read_complete, setup_logging
hf_logging.set_verbosity_error()
import logging


warnings.filterwarnings(
    "ignore",
    message="`do_sample` is set to `False`.*",
)
TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]

def main(args):
    data_list = load_data(args.dataset, args.data_type, args.augment_model)

    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)
    
    cot_name = "cot" if args.with_cot else "direct"
    # output_root_dir = os.path.join(
    #     ROOT_DIR, 
    #     "output" if not args.use_warmup else "warmup/output",
    #     "freezeA" if args.freeze_A else "",
    #     "warmup_cot" if args.with_cot_warmup else "",
    #     f"warmup_epoch{args.warmup_epoch}" if args.warmup_epoch > 1 else "",
    #     f"{args.lora_architecture}",
    #     args.model_name, 
    #     f"rank={args.lora_rank}_alpha={args.lora_alpha}",
    #     args.dataset,
    #     f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
    #     f"aug_model={args.augment_model}",
    #     "post_train",
    #     f"lr={args.learning_rate}_epoch={args.num_post_train_epochs}_{cot_name}",
    #     args.inference_method, 
    # )
    for filename, fulldata in data_list:
        filename = filename.split(".")[0]
        logger.info(f"### Solving {filename} ###")
        # output_dir = os.path.join(output_root_dir, filename)
        # logger.info(f"Output directory: {output_dir}")
        # os.makedirs(output_dir, exist_ok=True)
        # with open(os.path.join(output_dir, "config.json"), "w") as fout:
        #     json.dump(vars(args), fout, indent=4)

        # predict_file = os.path.join(output_dir, "predict.json")
        # ret, start_with = read_complete(predict_file)

        fulldata = fulldata[start_with:] if args.sample == -1 else fulldata[start_with:args.sample]
        save_every = max(len(fulldata)//30, 1)
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            model, tokenizer, generation_config = get_model(
                args.model_name,
                max_new_tokens = args.max_new_tokens,
            )
            augments = data["augment"]

            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]

            def get_pred(model, psgs):
                merge_text = predict(model, tokenizer, generation_config, 
                                        question, with_cot=args.with_cot, 
                                        passages=psgs)
                pred = {
                    "test_id": test_id, 
                    "question": question, 
                    "answer": answer,   
                    "text": merge_text,
                }
                pred.update(evaluate(merge_text, answer, args.with_cot))
                pred["single_passages_results"] = []
                
                assert len(model.active_adapters) == 1, f"There should be only 1 active adapter, but receive {model.active_adapters}"
                assert len(model.peft_config.keys()) == 4, f"There should be 4 total adapters, but receive {model.peft_config.keys()}"
                for i in range(3):
                    model.set_adapter(str(i))
                    assert model.active_adapters == [str(i)], f"Conflict active adapters {model.active_adapters} vs {i}"
                    text = predict(model, tokenizer, generation_config, 
                                        question, with_cot=args.with_cot, 
                                        passages=psgs)
                    pred["single_passages_results"].append({
                        "passage_id": i,
                        "text": text,
                        **evaluate(text, answer, args.with_cot)
                    })
                return pred

            if args.inference_method == "icl":
                ret.append(get_pred(model, psgs=passages))
            else:
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
                  
                inject_hydra_lora(model, trained_data_adapters_dir=trained_adapter_dir, r=args.lora_rank, alpha=args.lora_alpha, target_modules=TARGET_MODULES, architecture=args.lora_architecture)
                torch.cuda.empty_cache()
                logging.info(f"Start training for test_id {test_id}")
                
                train(args, question, augments, model, tokenizer, generation_config, save_path=trained_adapter_dir)

        #         for epoch in range(args.num_post_train_epochs): # Full inference on 1,..,post_epochs
                    
                
        #         ret.append(get_pred(model, psgs=None))
        #         torch.cuda.empty_cache()
        #         gc.collect()
        #     if (len(ret) % save_every) == 0:
        #         with open(predict_file, "w") as fout:
        #             json.dump(ret, fout, indent=4)
        #         metrics = ["em", "f1", "prec", "recall"]
        #         ret_str = ""
        #         for met in metrics:
        #             acc = sum(float(d[met]) for d in ret) / len(ret)
        #             acc = round(acc, 4)
        #             ret_str += f"{met}\t{acc}\n"
        #         ret_str += "\n" + json.dumps(vars(args), indent=4)
        #         with open(os.path.join(output_dir, "result.txt"), "w") as fout:
        #             fout.write(ret_str)


        # with open(predict_file, "w") as fout:
        #     json.dump(ret, fout, indent=4)

        # ##### Evaluating #####
        # metrics = ["em", "f1", "prec", "recall"]
        # ret_str = ""
        # for met in metrics:
        #     acc = sum(float(d[met]) for d in ret) / len(ret)
        #     acc = round(acc, 4)
        #     ret_str += f"{met}\t{acc}\n"
        # ret_str += "\n" + json.dumps(vars(args), indent=4)
        # with open(os.path.join(output_dir, "result.txt"), "w") as fout:
        #     fout.write(ret_str)

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

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--num_post_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    # parser.add_argument("--inference_method", type=str, required=True, choices=["icl", "prag", "combine"])

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

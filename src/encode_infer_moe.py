import os
import gc
import json
import argparse
import torch
from tqdm import tqdm

from encode_moe import inject_hydra_lora_force_cuda0, train
import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, predict, load_data, read_complete

def main(args):
    data_list = load_data(args.dataset, args.data_type, args.augment_model)

    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)
    
    cot_name = "cot" if args.with_cot else "direct"
    load_adapter_path = os.path.join(
        ROOT_DIR, 
        "offline", 
        args.model_name, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
    )
    output_root_dir = os.path.join(
        ROOT_DIR, 
        f"output_{args.lora_architecture}_checkpoint",
        args.model_name, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
        args.inference_method, 
    )
    for filename, fulldata in data_list:
        filename = filename.split(".")[0]
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(output_root_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as fout:
            json.dump(vars(args), fout, indent=4)

        predict_file = os.path.join(output_dir, "predict.json")
        ret, start_with = read_complete(predict_file)

        fulldata = fulldata[start_with:] if args.sample == -1 else fulldata[start_with:args.sample]
        data_file = "/scratch/doluk/Compact-Interference-PRAG/data_aug/popqa/qwen2.5-1.5b-instruct/total.json" 
        with open(data_file, "r") as f:
            aug_data = json.load(f)
        save_every = max(len(fulldata)//10, 1)
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            model, tokenizer, generation_config = get_model(
                args.model_name,
                max_new_tokens = args.max_new_tokens,
            )
            augments = aug_data[test_id]["augment"]

            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]

            def get_pred(model, psgs):
                text = predict(model, tokenizer, generation_config, 
                                        question, with_cot=args.with_cot, 
                                        passages=psgs)
                prefix = "The answer is "
                if text.startswith(prefix):
                    text = text[len(prefix):]
                pred = {
                    "test_id": test_id, 
                    "question": question, 
                    "answer": answer, 
                    "text": text,
                }
                pred.update(evaluate(text, answer, args.with_cot))
                return pred

            if args.inference_method == "icl":
                ret.append(get_pred(model, psgs=passages))
            else:
                trained_adapter_dir = f"/scratch/doluk/Compact-Interference-PRAG/offline/qwen2.5-1.5b-instruct/rank=2_alpha=32/popqa/lr=0.0003_epoch=2_direct/aug_model=qwen2.5-1.5b-instruct/total/data_{test_id}"  
                inject_hydra_lora_force_cuda0(model, trained_data_adapters_dir=trained_adapter_dir, alpha=32, target_modules=["gate_proj", "up_proj", "down_proj"], architecture=args.lora_architecture)
                torch.cuda.empty_cache()
                train(question, augments, model, tokenizer, generation_config)

                ret.append(get_pred(model, psgs=None))
                torch.cuda.empty_cache()
                gc.collect()
            if (len(ret) % save_every) == 0:
                with open(predict_file, "w") as fout:
                    json.dump(ret, fout, indent=4)
                metrics = ["em", "f1", "prec", "recall"]
                ret_str = ""
                for met in metrics:
                    acc = sum(float(d[met]) for d in ret) / len(ret)
                    acc = round(acc, 4)
                    ret_str += f"{met}\t{acc}\n"
                ret_str += "\n" + json.dumps(vars(args), indent=4)
                with open(os.path.join(output_dir, "result.txt"), "w") as fout:
                    fout.write(ret_str)


        with open(predict_file, "w") as fout:
            json.dump(ret, fout, indent=4)

        ##### Evaluating #####
        metrics = ["em", "f1", "prec", "recall"]
        ret_str = ""
        for met in metrics:
            acc = sum(float(d[met]) for d in ret) / len(ret)
            acc = round(acc, 4)
            ret_str += f"{met}\t{acc}\n"
        ret_str += "\n" + json.dumps(vars(args), indent=4)
        with open(os.path.join(output_dir, "result.txt"), "w") as fout:
            fout.write(ret_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen2.5-1.5b-instruct")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="popqa")
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1) # -1 means all
    parser.add_argument("--augment_model", type=str, default=None)  
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--inference_method", type=str, required=True, choices=["icl", "prag", "combine"])
    parser.add_argument("--lora_architecture", type=str, default="moe_lora", choices=["moe_lora", "mixture_lora", "moe_mixture"])
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)
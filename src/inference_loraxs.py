import os
import warnings
from lora_xs.utils.initialization_utils import fill_trained_data_adapters, fill_trained_passage_adapter
from root_dir_path import ROOT_DIR
from utils import get_model, load_data, read_complete, model_generate, evaluate, predict
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import gc
import json
import argparse
import torch
from tqdm import tqdm
from peft import PeftModel

import prompt_template
from root_dir_path import ROOT_DIR
from transformers.utils import logging

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.set_verbosity_error()
def main(args):
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    # model, tokenizer, generation_config = get_model(
    #     args.model_name,
    #     max_new_tokens = args.max_new_tokens,
    # )
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)
    
    cot_name = "cot" if args.with_cot else "direct"
    load_adapter_path = os.path.join(
        ROOT_DIR, 
        "offline_loraxs", 
        args.model_name, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
    )
    output_root_dir = os.path.join(
        ROOT_DIR, 
        "output_loraxs",
        args.model_name, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
        args.inference_method, 
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=['down_proj', 'gate_proj', 'up_proj'],
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
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
        save_every = max(len(fulldata)//10, 1)
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            model, tokenizer, generation_config = get_model(
                args.model_name,
                max_new_tokens = args.max_new_tokens,
            )
            model = get_peft_model(model, peft_config)
            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]

            def get_pred(model, psgs):
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

            if args.inference_method == "icl":
                ret.append(get_pred(model, psgs=passages))
            else:
                data_adapter_dir = os.path.join(load_adapter_path, filename, f"data_{test_id}")
                fill_trained_data_adapters(model, peft_config,
                                           os.path.join(ROOT_DIR, "offline_loraxs", args.model_name, f"rank={args.lora_rank}", "svd_adapters.pt"),
                                           data_adapter_dir)
                ret.append(get_pred(model, psgs=None if args.inference_method == "prag" else passages))
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


# def main(args):
#     model, tokenizer, generation_config = get_model("qwen2.5-1.5b-instruct")
#     question = "What is George Rankin's occupation?"
#     print(f"Base model output: {model_generate(question, model, tokenizer, generation_config)}")
#     passage_adapter = r"/scratch/doluk/Compact-Interference-PRAG/offline_loraxs/qwen2.5-1.5b-instruct/rank=16_alpha=256/popqa/lr=0.0003_epoch=5_direct/aug_model=qwen2.5-1.5b-instruct/total/data_0/passage_0"
#     data_adapter = r"/scratch/doluk/Compact-Interference-PRAG/offline_loraxs/qwen2.5-1.5b-instruct/rank=16_alpha=256/popqa/lr=0.0003_epoch=5_direct/aug_model=qwen2.5-1.5b-instruct/total/data_0"
    
#     assert os.path.exists(passage_adapter), f"Passage adapter path {passage_adapter} does not exist, check again"
#     peft_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         target_modules=['down_proj', 'gate_proj', 'up_proj'],
#         inference_mode=False,
#         r=args.lora_rank,
#         lora_alpha=args.lora_alpha,
#         lora_dropout=0,
#     )
#     peft_model = get_peft_model(model, peft_config)
#     fill_trained_passage_adapter(
#         peft_model,
#         peft_config,
#         os.path.join(ROOT_DIR, "offline_loraxs", "qwen2.5-1.5b-instruct", "rank=16", "svd_adapters.pt"),
#         os.path.join(passage_adapter, "adapter_model.safetensors")
#     )
#     print(f"Single-passage Model output: {model_generate(question, peft_model, tokenizer, generation_config)}")
#     model, tokenizer, generation_config = get_model("qwen2.5-1.5b-instruct")
#     peft_model = get_peft_model(model, peft_config)
#     fill_trained_data_adapters(
#         peft_model,
#         peft_config,
#         os.path.join(ROOT_DIR, "offline_loraxs", "qwen2.5-1.5b-instruct", "rank=16", "svd_adapters.pt"),
#         data_adapter
#     )
#     peft_model.eval()
#     peft_model.to("cuda")
#     print(f"Multiple-passage Model output: {model_generate(question, peft_model, tokenizer, generation_config)}")
#     # print(peft_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1) # -1 means all
    parser.add_argument("--augment_model", type=str, default=None)  
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--inference_method", type=str, required=True, choices=["icl", "prag", "combine"])
    # LoRA
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)
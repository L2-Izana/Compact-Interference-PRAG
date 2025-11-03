import os
from lora_xs.utils.initialization_utils import find_and_fill_trained_weights
from root_dir_path import ROOT_DIR
from utils import get_model, load_data, read_complete, model_generate
from peft import LoraConfig, TaskType, get_peft_model


def main(args):
    model, tokenizer, generation_config = get_model("qwen2.5-1.5b-instruct")
    question = "What is George Rankin's occupation?"
    print(model_generate(question, model, tokenizer, generation_config))
    passage_adapter = r"/scratch/doluk/Compact-Interference-PRAG/offline_loraxs/qwen2.5-1.5b-instruct/rank=16_alpha=256/popqa/lr=0.0003_epoch=5_direct/aug_model=qwen2.5-1.5b-instruct/total/data_0/passage_0"
    assert os.path.exists(passage_adapter), f"Passage adapter path {passage_adapter} does not exist, check again"
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=['down_proj', 'gate_proj', 'up_proj'],
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
    )
    peft_model = get_peft_model(model, peft_config)
    find_and_fill_trained_weights(
        peft_model,
        peft_config,
        os.path.join(ROOT_DIR, "offline_loraxs", "qwen2.5-1.5b-instruct", "rank=16", "svd_adapters.pt"),
        os.path.join(passage_adapter, "adapter_model.safetensors")
    )
    peft_model.eval()
    peft_model.to("cuda")
    # print(peft_model)
    print(model_generate(question, peft_model, tokenizer, generation_config))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=128)
    args = parser.parse_args()
    main(args)
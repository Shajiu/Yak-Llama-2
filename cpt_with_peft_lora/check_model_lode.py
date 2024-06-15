from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import os
from transformers import LlamaTokenizer
from typing import TYPE_CHECKING, Any, Dict, Tuple

model_name_or_path="../cymcode/merge_llama2_with_chinese_lora_13B/huggingface/"

llama_tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
print(len(llama_tokenizer))


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    return {
        "trust_remote_code": True,
        "cache_dir": "./",
        "revision": "1.2",
        "token": "11",
    }

init_kwargs = _get_init_kwargs("ModelArguments")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **init_kwargs)
tokenizer_vocab_size = len(tokenizer)
print(tokenizer_vocab_size)
config = AutoConfig.from_pretrained(model_name_or_path, **init_kwargs)
print(config.vocab_size)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config, **init_kwargs)
print(model)

# lora_model_path="./output_dir/pt_lora_model"
# lora_state_dict = torch.load(os.path.join(lora_model_path,'adapter_model.bin'),map_location='cpu')
# print(len(lora_state_dict))
# for key in lora_state_dict.items():
#     print(key)
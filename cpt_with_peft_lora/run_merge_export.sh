# 转为huggingface格式
python3.9 merge_lora_low_mem.py \
  --base_model per_models/Llama-2-13b-hf \
  --lora_model output_dir_7B/checkpoint-219800 \
  --output_type huggingface \
  --output_dir ./merge_llama2_with_chinese_lora_13B/huggingface/ 


# 转为pth格式
python3.9 merge_lora_low_mem.py \
  --base_model per_models/Llama-2-13b-hf \
  --lora_model output_dir_7B/checkpoint-219800 \
  --output_type pth \
  --output_dir ./merge_llama2_with_chinese_lora_13B/pth/ 

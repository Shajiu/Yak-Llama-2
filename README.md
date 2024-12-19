
## Yak-Llama 2: 面向藏文高效扩展的大语言模型
![Yak](https://github.com/Shajiu/Yak-Llama-2/blob/main/cpt_with_peft_lora/logo.png#pic_center=600x200)

**一、引言：**

-  本项目开源了**Yak-Llama 2: 面向藏文高效扩展的大语言模型**，以进一步促进大模型在藏文NLP社区的开放研究。这些模型**在原版LLaMA的基础上扩充了藏文词表**并使用了藏文数据进行继续预训练，进一步提升了藏文基础语义理解能力。同时，藏文Chat模型进一步使用了藏文指令数据进行精调，显著提升了模型对指令的理解和执行能力。

**二、摘要：**

-  本研究针对大型语言模型在处理藏语任务方面的局限性，提出了一套综合解决方案。
-  首先，我们构建了一个15G 的大规模高质量藏语无监督预训练数据集和一个包含100 万条数据的监督式多任务微调数据集，有效缓解了数据资源稀缺的问题。
-  进一步，通过分析藏文的特殊构词和语法特性，我们选定了基于Unigram 的词元化策略，并开发了一个高效的藏文分词器。
-  此外，通过对Llama 2-7B/13B 模型进行藏文词表扩展和采用藏文比英中按3:4 比例的混合继续预训练，显著增强了模型对藏语的理解和生成能力。
-  为优化模型训练与部署的效率，并提升其对藏语语义和指令的遵循能力，本研究在有监督的多任务指令数据集上应用了LoRA+ 微调方法，进一步提升了性能。
-  为验证所提方法的有效性，我们建立了面向藏文的十种评测基准，并通过自动与人工评测进行了综合验证，结果显示该方法在英文能力保持良好的前提下，在藏文理解与生成方面取得了显著进步。本研究的创新之处在于系统性地解决LLMs 处理藏语数据的效率和性能问题，特别是通过独创的词元化策略和混合预训练方法大幅提升了处理效率和泛化能力。
-  所有研究成果已开源，为藏语及其他少数民族语言的LLMs 应用和研究提供了宝贵资源和坚实基础，促进了LLMs 在处理特定语种上的研究和部署，同时为人工智能领域内语言多样性的发展贡献了重要力量。

**三、贡献：**

- 🚀 对Llama 2 进行藏文词表扩充，提高编解码效率。与原始Llama 2 相对，藏文序列长度减少约42.64%，模型在藏文域的最大文本输入长度提升至原来的两倍以上；
- 🚀 本文初步全面探索针对藏文的词元化策略，最终选取Unigram 为藏文的最佳词元化策略，解决原基座模型的藏语词元化性能不佳问题，增强模型对藏文内容的理解和生成能力；
- 🚀 结合藏文和英中文数据进行3:4 的比例混合CPT，并对藏文进行指令微调，同时开源了基于Llama 2-7B/13B 规模的Base 和Chat 模型权重，支持更广泛的应用和研究。

**四、训练策略：**

本文采取两阶段方式:
-  第一阶段，固定模型Transformer 部分的参数，仅训练Embedding，在尽量不干扰原模型的情况下适配新增的藏文词向量；
-  第二阶段：为模型添加LoRA+ 权重，训练Embedding 的同时也更新LoRA+ 参数。两阶段的训练方式虽然效率较低，然而有效缓解了由于藏文数据与Llama 2 模型预训练时使用的数据分布存在差距而在CPT 过程中出现分布偏移的问题。

**五、使用方法：**

#### 1. 部署环境
```sh
pip install -r requirements.txt
```

#### 2. 继续预训练数据处理为如下格式
```sh
# 存储在./data/data.txt中，一行为一条样本
གྲོང་ཚོར་བཅའ་སྡོད་བྱ་ཡུལ་དུ་ཁྲང་ཧྥུའུ་ལེའང་གིས་གྲོང་ཚོར་བཅའ་སྡོད་ལས་བྱེད་པ་དང་བསམ་བློ་བརྗེ་རེས་བྱས་པ་དང་། གཞུང་ལས་ཁང་(ཅུད)གི་ཏང་ཙུའུ་ཡིས་བཅའ་སྡོད་ལས་བྱེད་པར་ཐུགས་ཁུར་ཟབ་བཞེས་གནང་ལུགས་བརྒྱུད་བསྒྲགས་བྱས་པ་མ་ཟད། གྲོང་ཚོར་བཅའ་སྡོད་ལས་བྱེད་པའི་འཚོ་བ་དང་བདེ་ཐང་གི་གནས་ཚུལ་ལ་རྒྱུས་ལོན་བྱས་ཁར་། གྲོང་ཚོར་བཅའ་སྡོད་ལས་དོན་རུ་ཁག་གི་ལས་དོན་སྙན་ཞུ་ནན་ཏན་ངང་གསན་པ་དང་ཁོང་ཚོར་ཐོབ་པའི་གྲུབ་འབྲས་ལ་ཁས་ལེན་གང་ལེགས་གནང་བ་རེད།
```

#### 3. CPT训练
```sh
sh run_pt.sh
```
详细内容如下
```sh

lr=2e-5
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=per_models/Llama-2-13b-hf                 # 本地已下载的Llama-2-13b-hf基座模型路径
tibetan_tokenizer_path=per_models/merged_tokenizer_hf      # 通过扩充的藏文词汇表路径
dataset_dir=data                                           # 训练数据的文件夹
data_cache=temp_data_cache_dir       
per_device_train_batch_size=12
gradient_accumulation_steps=8
block_size=256
output_dir=output_dir_13

deepspeed_config_file=ds_z2_config.json                     # deepspeed配置文件                  

torchrun --nnodes 1 --nproc_per_node 1 --master_port=25678 train.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tibetan_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.005 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --seed $RANDOM \
    --bf16 \
    --num_train_epochs 100000 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 100 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --torch_dtype float16 \
    --load_in_kbits 16 \
    --save_safetensors False \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
```

#### 4. 合并权重
```sh
sh run_merge_export.sh
```
详细内容如下
```sh

# 转为huggingface格式
python3.9 merge_lora_low_mem.py \
  --base_model per_models/Llama-2-13b-hf \
  --lora_model output_dir_7B/checkpoint-219800 \
  --output_type huggingface \
  --output_dir ./merge_llama2_with_chinese_lora_13B/huggingface/ 

-----------------------------------------------------------------OR-----------------------------------------------------------------

# 转为pth格式
python3.9 merge_lora_low_mem.py \
  --base_model per_models/Llama-2-13b-hf \
  --lora_model output_dir_7B/checkpoint-219800 \
  --output_type pth \
  --output_dir ./merge_llama2_with_chinese_lora_13B/pth/ 
```

#### 5. 启动服务
```sh
python serve_demo.py
```
#### 6. 请求服务的方式
```sh
POST   http://10.x.x.x:8718/api/chat

{"query":"xxxxxxxxx"}
```

#### 7. 离线交互式对话/文件预测
- 使用transformers库推理

```sh
python inference_hf.py \
    --base_model path_to_llama3_chinese_instruct_hf_dir \
    --with_prompt \
    --interactive \
```
- 使用vLLM进行推理加速

```sh
python inference_hf.py \
    --base_model path_to_llama3_chinese_instruct_hf_dir \
    --with_prompt \
    --interactive \
    --use_vllm 
```

#### 8. 基于llama.cpp的量化部署
- 安装 llama.cpp
```sh
https://github.com/ggerganov/llama.cpp
```
- 编译
 
```sh
make LLAMA_CUBLAS=1
```

- 将上述.pth/bin模型权重转换为ggml的FP16格式
```
python convert_hf_to_gguf.py ../path_to_llama3_chinese_instruct_hf_dir/
```

- 对FP16模型进行4-bit量化
```
./llama-quantize.exe ../path_to_llama3_chinese_instruct_hf_dir/ggml-model-f16.gguf ../path_to_llama3_chinese_instruct_hf_dir/ggml-model-q4_0.bin q4_0​
```

- 交互式测试
```
 ./llama-cli.exe  --conversation -m ../path_to_llama3_chinese_instruct_hf_dir/ggml-model-q4_0.bin --color -f prompts/alpaca.txt -c 2048 --temp 0.2 -n 256
```

- 服务版本
```sh
./llama-server.exe -m ../path_to_llama3_chinese_instruct_hf_dir/ggml-model-q4_0.gguf -c 2048
```
**五、模型下载：**

| 模型名称 | 类型 | 训练数据 |重构模型|大小|下载地址|
| :-----| ----: | :----: |:----: |:----: |:----: |
| Yak_Llama2_7B | 基座 | 8B |原版LLaMA-7B|12.9 GB|[Yak_Llama2_7B](https://huggingface.co/shajiu/Yak_Llama2_7B)|
| Yak_Llama2_13B    | 基座 | 13B  |原版LLaMA-7B|24.7 GB |[Yak_Llama2_13B](https://huggingface.co/shajiu/Yak_Llama2_13B)|


**六、数据集下载

| 数据名称 | 数据类型 | 数据规模 |下载地址|
| :-----| ----: | :----: |:----: |
| TibetanGeneral corpus | 继续预训练 | 15GB| [TibetanGeneral corpus](https://huggingface.co/datasets/shajiu/TibetanGeneral_corpus)|
| TibetanSft corpus     | 指令微调   | 1GB | [TibetanSft corpus](https://huggingface.co/datasets/shajiu/TibetanSft_corpus)|


出于数据安全考虑和潜在的危害性影响，我们选择不公开完整的安全测试数据集。仓库中可下载的公开测试集仅包括部分数据。但是，为了进行安全研究，研究人员可以通过邮件至`zhumx@ncut.edu.cn`进行申请。经过批准后，我们​​将向申请人提供完整的数据集。
**声明:** 数据集包含有害违规内容示例，均不代表本团队立场。



**七、免责声明：**

本项目相关资源仅供学术研究之用，严禁用于商业用途。 使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目不对其准确性作出保证。对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。本项目由个人及协作者业余时间发起并维护，因此无法保证能及时回复解决相应问题。


**八、致谢：**

本项目主要基于以下开源项目二次开发，在此对相关项目和研究开发人员表示感谢。
- [Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)


**九、引用：**

若使用本项目的数据、代码或模型，请引用本项目。
```text
@misc{Tibetan_Mental_Health_Chat,
  author = {shajiu},
  title = {Yak-Llama 2: 面向藏文高效扩展的大语言模型},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{git@github.com:Shajiu/Yak-Llama-2.git}},
}
```

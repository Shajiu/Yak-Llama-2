# -*- coding: UTF-8 -*-

"""
功能为：基于peft库的lora做cpt训练使用

Authors:  shajiu
@File:  train.py
@Software:  PyCharm
"""
import logging
import numpy as np
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping
from pathlib import Path
import datasets
import torch
from datasets import load_dataset, concatenate_datasets

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
    BitsAndBytesConfig
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from sklearn.metrics import accuracy_score
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from config import ModelArguments, DataTrainingArguments, MyTrainingArguments

logger = logging.getLogger(__name__)


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "pt_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8/INT4 parameters to fp32
    for param in model.parameters():
        if ((param.dtype == torch.float16) or (param.dtype == torch.bfloat16)) and loaded_in_kbit:
            param.data = param.data.to(torch.float32)

    for name, module in model.named_modules():
        if 'norm' in name:
            module = module.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model

def accuracy(predictions, references, normalize=True, sample_weight=None):
    return {
        "accuracy": float(
            accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
        )
    }

def compute_metrics(eval_preds):
    # 拆解标签/预测
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    # 去掉第一维度上的值，然后将labels展平
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    # 最后一个维度寻求最大的值返回
    return logits.argmax(dim=-1)

def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        # 判断类型是否为long/float
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.

    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

    except ValueError:  # quick fix by simply take the first example
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果命令行参数只有一个，并且是一个 JSON 文件的路径，则解析该文件以获取参数
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # 否则，解析命令行参数 
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_pert_lora", model_args, data_args)
    
    # 设置日志记录的格式和级别
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
                        handlers=[logging.StreamHandler(sys.stdout)], )

    if training_args.should_log:
        # 如果需要记录日志，则设置日志级别为 info
        transformers.utils.logging.set_verbosity_info()
    
    # 获取日志级别，并设置 logger 的日志级别
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # transformers.tokenization_utils.logging.set_verbosity_warning()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        # 保证当前路径下是否存在文件(防止替换)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 在初始化模型之前设置种子。
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.config_name:
        # 加载单独的配置文件
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        # 加载模型中的配置文件
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        # 直接新建一个配置文件
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        # 加载词汇表中的分词器
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.tokenizer_name_or_path:
        # 加载额外配置的词汇表分词器
        tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # 设置特殊标记
    tokenizer.add_eos_token = True

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples["text"])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    if data_args.block_size is None:
        # 未设置值时处理
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            # 此处给予最大值
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            # 当设置的值大于最大值时
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        # 选取最小的一个值
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        lm_datasets = []
        # 获取数据路径
        path = Path(data_args.dataset_dir)
        # 获取扩展名为txt的文件
        files = [file.name for file in path.glob("*.txt")]
        # 如果仅仅用于调试，则选择一个第一个文件即可
        if training_args.debug_mode is True:
            files = [files[0]]
        for idx, file in enumerate(files):
            # 拼接文件路径
            data_file = os.path.join(path, file)
            # 仅获取文件名称(不包括扩展名)
            filename = ''.join(file.split(".")[:-1])
            # 拼接缓存路径
            cache_path = os.path.join(data_args.data_cache_dir, filename + f"_{block_size}")
            # 创建文件夹
            os.makedirs(cache_path, exist_ok=True)
            try:
                # 数据不完全加载在磁盘中进行
                processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                logger.info(f'training datasets-{filename} has been loaded from disk')
            except Exception:
                # 拼接两个路径
                cache_dir = os.path.join(data_args.data_cache_dir, filename + f"_text_{block_size}")
                os.makedirs(cache_dir, exist_ok=True)
                raw_dataset = load_dataset("text", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)
                logger.info(f"{file} has been loaded")
                tokenized_dataset = raw_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns="text",
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(cache_dir, 'tokenized.arrow') for k in raw_dataset},
                    desc="Running tokenizer on dataset",
                )
                grouped_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(cache_dir, 'grouped.arrow') for k in tokenized_dataset},
                    desc=f"Grouping texts in chunks of {block_size}",
                )
                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)
            if idx == 0:
                lm_datasets = processed_dataset['train']
            else:
                assert lm_datasets.features.type == processed_dataset["train"].features.type
                # 合并数据集
                lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])
        
        # 合并后的数据划分训练集和测试集
        lm_datasets = lm_datasets.train_test_split(test_size=data_args.validation_split_percentage)
    #如果需要进行训练
    if training_args.do_train:
        # 获取训练集部分
        train_dataset = lm_datasets['train']
        if data_args.max_train_samples is not None:
            # 如果设置了数据使用最大规模。计算最大训练样本数据
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            # 选择最大训练样本数的子集
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("Training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
    # 针对开发集
    if training_args.do_eval:
        # 获取测试集
        eval_dataset = lm_datasets["test"]
        # 如果测试集不是为空
        if data_args.max_eval_samples is not None:
            # 如果设置了数据使用最大规模。计算最大开发样本数据
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            # 选组最大开发集规模
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("Evaluation example:")
        # 打印评估集的第一条
        logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))
        
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if training_args.load_in_kbits in [4, 8]:
        load_in_4bit = training_args.load_in_kbits == 4
        load_in_8bit = training_args.load_in_kbits == 8
        if training_args.modules_to_save is not None:
            load_in_8bit_skip_modules = training_args.modules_to_save.split(',')
        else:
            load_in_8bit_skip_modules = None
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_args.load_in_kbits == 4,
            load_in_8bit=training_args.load_in_kbits == 8,
            llm_int8_threshold=6.0,
            load_in_8bit_skip_modules=load_in_8bit_skip_modules,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type
        )
    else:
        load_in_4bit = False
        load_in_8bit = False
        quantization_config = None
    if quantization_config is not None:
        logger.info(f"quantization_config:{quantization_config.to_dict()}")
    # 预训练模型地址
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        # 加载模型
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            use_flash_attention_2=training_args.use_flash_attention_2
        )
    else:
        # 从头开始训练模型
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params")
    if training_args.load_in_kbits in [4, 8]:

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    model.config.use_cache = False
    # 获取基座模型中携带的词汇表的大小
    model_vocab_size = model.get_output_embeddings().weight.size(0)
    # 获取词汇表的大小
    tokenizer_vocab_size = len(tokenizer)
    logger.info(f"Model vocab size: {model_vocab_size}")
    logger.info(f"Tokenizer vocab size: {tokenizer_vocab_size}")
    # 判断外部加入的词汇表大小是否合理。56724为扩展后的词汇表大小。
    if tokenizer_vocab_size != 56724:
        raise ValueError(
            f"The vocab size of tokenizer is {tokenizer_vocab_size}, not 55296. Please use Chinese-LLaMA-2 tokenizer.")
    # 如果是我们自己定义的tokenizer，需要将模型的嵌入层和lm_head层的词表数目进行重新设置
    if model_vocab_size != tokenizer_vocab_size:
        logger.info(f"Resize model vocab size to {tokenizer_vocab_size}")
        model.resize_token_embeddings(len(tokenizer))
    # 如果不是全部参数微调
    if not training_args.full_finetuning:
        # 微调方式的路径已经设置的话，中间过程的加载和训练。
        if training_args.peft_path is not None:
            logger.info("Peft from pre-trained model")
            model = PeftModel.from_pretrained(model, training_args.peft_path, device_map=device_map)
        else:
            logger.info("Init new peft model")
            # 获取微调方式
            target_modules = training_args.trainable.split(',')
            # 获取启动方式
            modules_to_save = training_args.modules_to_save
            # 先训练词嵌入+再微调
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            # 获取lora方法中的秩
            lora_rank = training_args.lora_rank
            # 获取lora方法中的dropout参数
            lora_dropout = training_args.lora_dropout
            # 获取lora方法中的缩放系数
            lora_alpha = training_args.lora_alpha
            # 
            logger.info(f"target_modules: {target_modules}")
            logger.info(f"lora_rank: {lora_rank}")
            # 加载Lora的配置文件
            # 因果语言模型
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,           # 指定了任务类型为TaskType.CAUSAL_LM(因果语言模型),
                target_modules=target_modules,          # 指定 PEFT 模型的目标模块为 "query_key_value"。
                inference_mode=False,                   # 推理模式为False
                r=lora_rank, lora_alpha=lora_alpha,     # 指定 PEFT 模型的 r 值为 64，指定 PEFT 模型的 lora_alpha 值为128
                lora_dropout=lora_dropout,              # lora的丢失率
                modules_to_save=modules_to_save)        # 两阶段训：先词嵌入、然后微调
            model = get_peft_model(model, peft_config)
        # 函数用于打印模型可训练参数
        model.print_trainable_parameters()
        logger.info(f"model.modules_to_save: {model.modules_to_save}")
    # 不是非全量参数微调&梯度检查点&非两阶段微调
    if not training_args.full_finetuning and training_args.gradient_checkpointing and \
            (not model.modules_to_save or 'embed_tokens' not in model.modules_to_save):
        # enable requires_grad to avoid exception during backward pass when using gradient_checkpoint without tuning embed.
        # 
        if hasattr(model.base_model, "enable_input_require_grads"):
            model.base_model.enable_input_require_grads()
        # 检查对象是否具有指定的属性  
        elif hasattr(model.base_model, "get_input_embeddings"):
            def make_inputs_require_grad(_module, _input, _output):
                _output.requires_grad_(True)
            # 每当嵌入层执行前向传播时，make_inputs_require_grad函数将被调用，这通常用于确保可以针对输入计算梯度，这对于某些高级模型训练技术（例如对抗训练）是必需的。
            model.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )
    # 不是全微调，对于普通选手，一般不会使用全微调，太耗资源了
    # 一般是使用peft的方法来进行微调
    # peft的lora是一种横向微调的手段，是否可以探索下深度的lora呢？
    # 多个模型的lora，拼接起来，称为新的更深度的模型。
    trainer.add_callback(SavePeftModelCallback)
    # 训练阶段
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # 存储地址非空的情况下
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # 
        metrics = train_result.metrics
        # 
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        # 选取最小合适的训练数据集
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            # 计算自然数的指数
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    main()

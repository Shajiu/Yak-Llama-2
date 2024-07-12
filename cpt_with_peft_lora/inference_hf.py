# -- coding: utf-8 --
# @time :
# @author : shajiu
# @email : 18810979033@163.com
# @file : .py
# @software: pycharm

"""
功能:主要用于加载模型并做推理
"""
import argparse
import json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig

"""
支持推理预测部分的代码
"""
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
system_format = '<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
user_format = '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
assistant_format = '{content}<|eot_id|>'

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True,
                    help="存放HF格式的Llama-3-Chinese-Instruct模型权重和配置文件的目录。也可使用🤗Model Hub模型调用名称")
parser.add_argument('--tokenizer_path', default=None, type=str, help="存放对应tokenizer的目录。若不提供此参数，则其默认值与--base_model相同")
parser.add_argument('--data_file', default=None, type=str, help="非交互方式启动下，按行读取file_name中的的内容进行预测")
parser.add_argument('--with_prompt', action='store_true',
                    help="是否将输入与prompt模版进行合并。如果加载Llama-3-Chinese-instruct模型，请务必启用此选项！")
parser.add_argument('--interactive', action='store_true', help="以交互方式启动，以便进行多次单轮问答（此处不是llama.cpp中的上下文对话）")
parser.add_argument('--predictions_file', default='./predictions.json', type=str,
                    help="非交互式方式下，将预测的结果以json格式写入file_name")
parser.add_argument('--gpus', default="0", type=str, help="指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如0,1,2")
parser.add_argument('--only_cpu', action='store_true', help='仅使用CPU进行推理')
parser.add_argument('--load_in_8bit', action='store_true', help="使用8bit方式加载模型，降低显存占用")
parser.add_argument('--load_in_4bit', action='store_true', help="使用4bit方式加载模型，降低显存占用，推荐使用--load_in_4bit")
parser.add_argument("--use_vllm", action='store_true', help="使用vLLM作为LLM后端进行推理")
parser.add_argument('--use_flash_attention_2', action='store_true', help="使用Flash-Attention 2加速推理，如果不指定该参数，代码默认SDPA加速。")
args = parser.parse_args()

# 可以使用vLLM作为LLM后端进行推理
if args.use_vllm:
    # 使用8bit或4bit方式加载模型，降低显存占用
    if args.load_in_8bit or args.load_in_4bit:
        raise ValueError("vLLM currently does not support quantization, please use fp16 (default) or unuse --use_vllm.")
    # 仅仅使用CPU
    if args.only_cpu:
        raise ValueError(
            "vLLM requires GPUs with compute capability not less than 7.0. If you want to run only on CPU, please unuse --use_vllm.")
# 未使用vLLM的前提下通过 使用8bit或4bit方式加载模型，降低显存占用
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")

# 直接使用CPU加载
if args.only_cpu is True:
    args.gpus = ""  # 设置GPU编号为空
    # 此时，无法通过使用8bit或4bit方式加载模型
    if args.load_in_8bit or args.load_in_4bit:
        raise ValueError("Quantization is unavailable on CPU.")

# 设置CUDA可见的GPU设备编号
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# 若使用vLLM作为LLM后端进行推理，则需要导入对应的包
if args.use_vllm:
    from vllm import LLM, SamplingParams

if args.use_vllm:
    # 构建一个生成的参数
    generation_config = dict(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        max_tokens=100,
        presence_penalty=1.0,
    )
else:
    # 不使用vLLM作为LLM后端进行推理时的设置
    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        max_new_tokens=400
    )

# 测试用例
sample_data = ["为什么要减少污染，保护环境？"]


def generate_prompt(instruction):
    # 拼接指令模板
    return system_format.format(content=DEFAULT_SYSTEM_PROMPT) + user_format.format(content=instruction)


if __name__ == '__main__':
    load_type = torch.float16

    # 如果可用，将模型移动到MPS设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        # GPU可用
        if torch.cuda.is_available():
            device = torch.device(0)
        else:
            # 只能CPU推理
            device = torch.device('cpu')
    print(f"Using device: {device}")

    if args.tokenizer_path is None:
        # 未指定tokenizer路径则默认使用模型路径
        args.tokenizer_path = args.base_model
    # 加载次元化路径
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 获取对应的索引=[2,0]
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    # 可以使用vLLM作为LLM后端进行推理
    if args.use_vllm:
        # 加载模型、tokenizer地址、GPU个数、torch.float16
        model = LLM(model=args.base_model,
                    tokenizer=args.tokenizer_path,
                    tensor_parallel_size=len(args.gpus.split(',')),
                    dtype=load_type
                    )
        # 在使用vLLM作为LLM后端进行推理时的设置
        generation_config["stop_token_ids"] = terminators
        generation_config["stop"] = ["<|eot_id|>", "<|end_of_text|>"]

    else:
        # 非使用vLLM作为LLM后端进行推理时的设置
        if args.load_in_4bit or args.load_in_8bit:
            # 使用 NF4 量化加载 4 位模型
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                bnb_4bit_compute_dtype=load_type,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
            device_map='auto',
            quantization_config=quantization_config if (args.load_in_4bit or args.load_in_8bit) else None,
            # 使用Flash-Attention 2加速推理
            attn_implementation="flash_attention_2" if args.use_flash_attention_2 else "sdpa"
        )
        # 判断是否为CPU上加载
        if device == torch.device('cpu'):
            # 将模型的所有浮点参数和缓冲的类型转换为float数据类型
            model.float()
        model.eval()

    # 判断是否加载了测试数据集
    if args.data_file is None:
        examples = sample_data
    else:
        # 读取测试数据集(一行一个问题))
        with open(args.data_file, 'r') as f:
            examples = [line.strip() for line in f.readlines()]
        # 显示前10条测试集(query)
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)
    # 判断在PyTorch中是否禁用梯度计算。
    with torch.no_grad():
        # 以交互方式启动
        if args.interactive:
            print("Start inference with instruction mode.")

            print('=' * 85)
            print("+ 该模式下仅支持单轮问答，无多轮对话能力。\n"
                  "+ 如要进行多轮对话，请使用llama.cpp")
            print('-' * 85)
            print("+ This mode only supports single-turn QA.\n"
                  "+ If you want to experience multi-turn dialogue, please use llama.cpp")
            print('=' * 85)

            while True:
                # 获取输入
                raw_input_text = input("输入您的问题:")
                # 判断输入长度是否为0
                if len(raw_input_text.strip()) == 0:
                    break
                # 是否将输入与prompt模版进行合并
                if args.with_prompt:
                    input_text = generate_prompt(instruction=raw_input_text)
                else:
                    # 直接写入进行预测
                    input_text = raw_input_text

                if args.use_vllm:
                    # 如果使用vLLM作为LLM后端进行推理时的设置【输入为list的文本，】
                    output = model.generate([input_text], SamplingParams(**generation_config), use_tqdm=False)
                    # 输出具体的结果文本部分
                    response = output[0].outputs[0].text
                else:
                    # 如果不是使用vLLM作为LLM后端进行推理时的设置的话。
                    inputs = tokenizer(input_text, return_tensors="pt")  # add_special_tokens=False ?
                    generation_output = model.generate(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs['attention_mask'].to(device),
                        eos_token_id=terminators,
                        pad_token_id=tokenizer.eos_token_id,
                        generation_config=generation_config
                    )
                    s = generation_output[0]
                    output = tokenizer.decode(s, skip_special_tokens=True)
                    # 如果使用了指令拼接这块需从结果中去除前缀部分
                    if args.with_prompt:
                        # 只返回后半部分的内容
                        response = output.split("assistant\n\n")[-1].strip()
                    else:
                        # 否则全部返回
                        response = output
                # 返回的结果
                print("Response: ", response)
                print("\n")
        else:
            # 如果不是交互式方式（文件级别预测）
            print("Start inference.")
            results = []
            # 判断是否为VllM加载推理
            if args.use_vllm:
                # 拼接使用了指令
                if args.with_prompt is True:
                    inputs = [generate_prompt(example) for example in examples]
                else:
                    inputs = examples
                outputs = model.generate(inputs, SamplingParams(**generation_config))

                for index, (example, output) in enumerate(zip(examples, outputs)):
                    response = output.outputs[0].text
                    print(f"======={index}=======")
                    print(f"Input: {example}\n")
                    print(f"Output: {response}\n")
                    results.append({"Input": example, "Output": response})
            else:
                # 不使用VLLM
                for index, example in enumerate(examples):
                    # 拼接指令
                    if args.with_prompt:
                        input_text = generate_prompt(instruction=example)
                    else:
                        input_text = example
                    inputs = tokenizer(input_text, return_tensors="pt")  # add_special_tokens=False ?
                    generation_output = model.generate(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs['attention_mask'].to(device),
                        eos_token_id=terminators,
                        pad_token_id=tokenizer.eos_token_id,
                        generation_config=generation_config
                    )
                    # 拿出结果
                    s = generation_output[0]
                    output = tokenizer.decode(s, skip_special_tokens=True)
                    # 拼接了指令/需要去除指令前缀
                    if args.with_prompt:
                        response = output.split("assistant\n\n")[1].strip()
                    else:
                        response = output
                    print(f"======={index}=======")
                    print(f"Input: {example}\n")
                    print(f"Output: {response}\n")

                    results.append({"Input": input_text, "Output": response})
            # 结果写入
            dirname = os.path.dirname(args.predictions_file)
            # 创建文件
            os.makedirs(dirname, exist_ok=True)
            # 写文件
            with open(args.predictions_file, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            # 如果开启了vllm则保存此配置项目
            if args.use_vllm:
                with open(dirname + '/generation_config.json', 'w') as f:
                    json.dump(generation_config, f, ensure_ascii=False, indent=2)
            else:
                # 存储配置文件
                generation_config.save_pretrained('./')

# -*- coding: UTF-8 -*-
#
"""
功能为：主要用于调用llama2-7B对话模型

@File:  llama2-7b-server.py
@Software:  PyCharm
"""
import json
import logging
logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from flask import Flask
from flask import Response
from flask import request
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)



def load_model(model_name):
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_response(model, tokenizer, text):
    # 对输入的文本进行编码
    inputs = tokenizer.encode(text, return_tensors='pt')

    # 使用模型生成响应
    output = model.generate(inputs, max_length=50, num_return_sequences=1)

    # 对生成的输出进行解码，获取生成的文本
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output



@app.route('/api/chat', methods=['POST'])
def qtpdnn_v0():
    """Description"""
    inputs = request.get_json()
    response = generate_response(model, tokenizer, inputs.get("query"))
    print("输出",response)
    output=inputs
    output.update({"output":response})
    return Response(json.dumps(output, ensure_ascii=False), mimetype='application/json')


if __name__ == "__main__":
    # 模型名称
    model_name = 'merge_llama2_with_chinese_lora_13B/huggingface'
    # 加载模型
    tokenizer, model = load_model(model_name)
    app.run(host='0.0.0.0', port=8718, debug=False, threaded=False, processes=1)

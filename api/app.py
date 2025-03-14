from flask import Flask, request, jsonify
from markllm.watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load your LLM model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'facebook/opt-1.3b'
transformers_config = TransformersConfig(
    model=AutoModelForCausalLM.from_pretrained(model_name).to(device),
    tokenizer=AutoTokenizer.from_pretrained(model_name),
    vocab_size=50272,
    device=device,
    max_new_tokens=200,
    do_sample=True,
    no_repeat_ngram_size=4
)

# Load watermark algorithm
watermark = AutoWatermark.load('KGW', 
                              algorithm_config='config/KGW.json',
                              transformers_config=transformers_config)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt')
    algorithm = data.get('algorithm')

    # Generate watermarked text
    watermarked_text = watermark.generate_watermarked_text(prompt)

    # Generate non-watermarked text
    non_watermarked_text = watermark.generate_unwatermarked_text(prompt)

    return jsonify({
        'watermarked_text': watermarked_text,
        'non_watermarked_text': non_watermarked_text
    })

@app.route('/detect', methods=['POST'])
def detect_watermark():
    data = request.json
    text = data.get('text')
    result = watermark.detect_watermark(text)
    return jsonify(result)

@app.route('/analyze', methods=['POST'])
def analyze_text_quality():
    data = request.json
    text = data.get('text')
    # Implement your text quality analysis logic here
    return jsonify({
        'PPL': 19.352304458618164,
        'LogDiversity': 8.37216741936574
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

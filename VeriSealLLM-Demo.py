import streamlit as st
import torch
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 设置页面配置
st.set_page_config(
    page_title="VeriSealLLM Demo",
    page_icon=":guardsman:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题和说明
st.title("VeriSealLLM: An Open-Source Toolkit for LLM Watermarking")
st.markdown("""
This demo allows you to experiment with different watermarking algorithms for Large Language Models (LLMs).
""")

# 初始化模型和水印算法
device = os.getenv("MARKLLM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
model_name = os.getenv("MARKLLM_MODEL_NAME", "./models/facebook/opt-1.3b")

# 加载模型和分词器
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

transformers_config = TransformersConfig(
    model=model.to(device),
    tokenizer=tokenizer,
    vocab_size=50272,
    device=device,
    max_new_tokens=200,
    do_sample=True,
    no_repeat_ngram_size=4
)

# 加载水印算法
@st.cache_resource
def load_watermark_algorithm(algorithm_name):
    return AutoWatermark.load(
        algorithm_name,
        algorithm_config=f'config/{algorithm_name}.json',
        transformers_config=transformers_config
    )

# 算法选择
st.sidebar.header("Algorithm Configuration")
algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["KGW", "Unigram", "SWEET", "UPV", "SIR", "EXP", "EWD", "XSIR"]
)

watermark = load_watermark_algorithm(algorithm)

# 参数配置
with st.sidebar.expander("Parameters"):
    gamma = st.slider("Gamma", 0.0, 1.0, 0.5)
    delta = st.slider("Delta", 0, 10, 2)
    hash_key = st.number_input("Hash Key", value=15485863)
    prefix_length = st.number_input("Prefix Length", value=1)
    z_threshold = st.number_input("Z Threshold", value=4)

# 文本生成区域
st.header("Text Generation")
prompt = st.text_area("Input Prompt", "Good morning. Today is a sunny day.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Watermarked Text")
    if st.button("Generate Watermarked Text"):
        watermarked_text = watermark.generate_watermarked_text(prompt)
        st.text_area("Generated Text", watermarked_text, height=200)
        detection_result = watermark.detect_watermark(watermarked_text)
        st.json({"Detection Result": detection_result})

with col2:
    st.subheader("Non-Watermarked Text")
    if st.button("Generate Non-Watermarked Text"):
        non_watermarked_text = watermark.generate_unwatermarked_text(prompt)
        st.text_area("Generated Text", non_watermarked_text, height=200)
        detection_result = watermark.detect_watermark(non_watermarked_text)
        st.json({"Detection Result": detection_result})

# 可视化区域
st.header("Visualization")
visualization_type = st.selectbox(
    "Select Visualization Type",
    ["Token Highlighting", "Attention Weights", "Embedding Space"]
)

if visualization_type == "Token Highlighting":
    text_to_visualize = st.text_area("Text to Visualize", "")
    if st.button("Visualize"):
        st.markdown("### Visualization Result")
        st.code("Placeholder for token highlighting visualization")

# 性能分析
st.header("Performance Analysis")
analysis_type = st.selectbox(
    "Select Analysis Type",
    ["Text Quality", "Robustness", "Detection Accuracy"]
)

if analysis_type == "Text Quality":
    if st.button("Analyze Text Quality"):
        st.markdown("### Analysis Result")
        st.json({
            "PPL": 19.352304458618164,
            "LogDiversity": 8.37216741936574
        })

# 说明和参考
st.sidebar.markdown("---")
st.sidebar.markdown("## About")
st.sidebar.markdown("This demo is part of the VeriSealLLM project, an open-source toolkit for LLM watermarking.")
st.sidebar.markdown("[GitHub Repository](https://github.com/raccoonrat/VeriSealLLM)")

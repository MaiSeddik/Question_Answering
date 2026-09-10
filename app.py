import os
import time
import torch
import numpy as np
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ==========================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="RoBERTa Question Answering",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern visual design
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.8rem;
    }
    .answer-card {
        background-color: #F0FDF4;
        border-left: 5px solid #22C55E;
        padding: 1.25rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .answer-title {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #15803D;
        letter-spacing: 0.05em;
    }
    .answer-text {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.3rem;
    }
    .metric-badge {
        display: inline-block;
        background-color: #E2E8F0;
        color: #334155;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .highlight {
        background-color: #FEF08A;
        padding: 0.15rem 0.3rem;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL LOADER WITH CACHING & FALLBACK
# ==========================================
LOCAL_MODEL_PATH = "./results"
DEFAULT_HF_MODEL = "deepset/roberta-base-squad2"  # Change to "MaiSeddik/roberta-squad-qa" after uploading your model to HF Hub


@st.cache_resource(show_spinner=False)
def load_qa_model(model_name_or_path: str, fallback_hub_model: str):
    """
    Loads tokenizer and model with caching.
    Tries local path first; falls back to Hugging Face Hub if missing or invalid.
    """
    source_used = "Local Path"
    active_path = model_name_or_path

    # Check if local model directory exists and has model files
    local_exists = (
        os.path.exists(model_name_or_path) and 
        any(fname.endswith(('.bin', '.safetensors', '.onnx')) for fname in os.listdir(model_name_or_path) if os.path.isfile(os.path.join(model_name_or_path, fname)))
    ) if os.path.exists(model_name_or_path) else False

    if not local_exists:
        active_path = fallback_hub_model
        source_used = "Hugging Face Hub Fallback"

    try:
        tokenizer = AutoTokenizer.from_pretrained(active_path)
        model = AutoModelForQuestionAnswering.from_pretrained(active_path)
        model.eval()  # Set model to evaluation mode
        return tokenizer, model, active_path, source_used, None
    except Exception as e:
        # If local attempt threw an error, try Hugging Face Hub fallback
        if active_path != fallback_hub_model:
            try:
                tokenizer = AutoTokenizer.from_pretrained(fallback_hub_model)
                model = AutoModelForQuestionAnswering.from_pretrained(fallback_hub_model)
                model.eval()
                return tokenizer, model, fallback_hub_model, "Hugging Face Hub Fallback", None
            except Exception as fallback_error:
                return None, None, fallback_hub_model, "Failed", str(fallback_error)
        return None, None, active_path, "Failed", str(e)


# ==========================================
# INFERENCE PIPELINE
# ==========================================
def extract_answer(context: str, question: str, tokenizer, model, max_answer_len: int = 30):
    """
    Performs QA inference using PyTorch torch.no_grad().
    Returns answer text, start position, end position, confidence score, and raw probabilities.
    """
    # Tokenize input context and question
    inputs = tokenizer(
        question,
        context,
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offset_mapping = inputs.pop("offset_mapping")[0].numpy()
    input_ids = inputs["input_ids"][0]

    # PyTorch inference without gradient calculation
    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # Compute softmax probabilities over sequence length
    start_probs = torch.softmax(start_logits, dim=-1).cpu().numpy()
    end_probs = torch.softmax(end_logits, dim=-1).cpu().numpy()

    # Identify context sequence tokens (ignore question and special tokens)
    sequence_ids = inputs.sequence_ids(0)
    context_indices = [i for i, seq_id in enumerate(sequence_ids) if seq_id == 1]

    if not context_indices:
        return "", 0, 0, 0.0

    # Search for optimal start and end position within context tokens
    best_score = -1.0
    best_start = context_indices[0]
    best_end = context_indices[0]

    for start_idx in context_indices:
        for end_idx in context_indices:
            if end_idx < start_idx:
                continue
            if end_idx - start_idx + 1 > max_answer_len:
                continue

            score = start_probs[start_idx] * end_probs[end_idx]
            if score > best_score:
                best_score = score
                best_start = start_idx
                best_end = end_idx

    # Extract character offsets from context
    start_char = offset_mapping[best_start][0]
    end_char = offset_mapping[best_end][1]

    answer_text = context[start_char:end_char].strip()
    confidence = float(best_score)

    return answer_text, start_char, end_char, confidence


# ==========================================
# PRESET EXAMPLES
# ==========================================
PRESET_EXAMPLES = {
    "Select an example...": {
        "context": "",
        "question": ""
    },
    "Artificial Intelligence & RoBERTa": {
        "context": "RoBERTa is a transformers model trained on a large corpus of English data in a self-supervised fashion. It was built on BERT's language masking strategy and modifies key hyperparameters, removing BERT's next-sentence pretraining objective and training with much larger mini-batches and learning rates. This allows RoBERTa to match or exceed BERT performance on SQuAD and GLUE benchmarks.",
        "question": "What objective did RoBERTa remove from BERT?"
    },
    "Space Exploration": {
        "context": "The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy. As the largest optical telescope in space, its high resolution and sensitivity allow it to view objects too old, distant, or faint for the Hubble Space Telescope. It was launched on 25 December 2021 on an Ariane 5 rocket from Kourou, French Guiana.",
        "question": "When was the James Webb Space Telescope launched?"
    },
    "Photosynthesis": {
        "context": "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities. This chemical energy is stored in carbohydrate molecules, such as sugars and starches, which are synthesized from carbon dioxide and water.",
        "question": "What molecules store the chemical energy produced during photosynthesis?"
    }
}


# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================
def main():
    # Sidebar Setup
    st.sidebar.title("⚙️ Model Configuration")
    
    # Model Status Card
    st.sidebar.markdown("### 📦 Model Source")
    
    with st.spinner("Loading RoBERTa Question Answering model..."):
        tokenizer, model, active_model_id, load_source, error_msg = load_qa_model(
            LOCAL_MODEL_PATH, DEFAULT_HF_MODEL
        )

    if model is None:
        st.sidebar.error(f"❌ Failed to load model: {error_msg}")
        st.error("Model could not be initialized. Please check your model path or internet connection.")
        return

    if "Local" in load_source:
        st.sidebar.success(f"✅ Loaded from local directory:\n`{active_model_id}`")
    else:
        st.sidebar.info(f"🌐 Loaded from Hugging Face Hub:\n`{active_model_id}`")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Model Information")
    st.sidebar.markdown(f"""
    - **Architecture:** `RoBERTa-base`
    - **Task:** Extractive Question Answering
    - **Fine-tuned Dataset:** SQuAD v1.1 / v2.0
    - **Inference Engine:** PyTorch (`torch.no_grad()`)
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Inference Settings")
    max_answer_len = st.sidebar.slider("Max Answer Length (Tokens)", min_value=5, max_value=100, value=30, step=5)
    min_confidence_thresh = st.sidebar.slider("Min Confidence Threshold (%)", min_value=0, max_value=100, value=10, step=5) / 100.0

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Quick Presets")
    selected_preset = st.sidebar.selectbox("Choose a sample passage:", list(PRESET_EXAMPLES.keys()))

    # Main UI Header
    st.markdown('<div class="main-header">🤖 RoBERTa Question Answering System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Extract precise answers from any passage using a fine-tuned RoBERTa transformer model.</div>',
        unsafe_allow_html=True
    )

    # Preset handle
    preset_context = PRESET_EXAMPLES[selected_preset]["context"]
    preset_question = PRESET_EXAMPLES[selected_preset]["question"]

    # Context & Question Inputs
    col_input, col_info = st.columns([7, 3])

    with col_input:
        context_input = st.text_area(
            "Context / Passage",
            value=preset_context if preset_context else "",
            height=200,
            placeholder="Paste or type the reference text context here...",
            help="The passage containing the information needed to answer the question."
        )

        question_input = st.text_input(
            "Question",
            value=preset_question if preset_question else "",
            placeholder="What question would you like to ask about the passage above?",
            help="Enter a specific question related to the provided context."
        )

    with col_info:
        st.markdown("#### 💡 How to use")
        st.markdown("""
        1. **Provide Context:** Paste any news article, document snippet, or text.
        2. **Ask Question:** Type a factual question based on the text.
        3. **Get Answer:** Click the button below to extract the exact answer span.
        """)
        st.info("Tip: You can select a sample preset from the sidebar to try it instantly!")

    # Action Button
    get_answer_btn = st.button("🔍 Get Answer", type="primary", use_container_width=True)

    # Trigger Inference
    if get_answer_btn:
        # Validation checks
        if not context_input.strip():
            st.warning("⚠️ Please provide a Context / Passage before asking a question.")
            return

        if not question_input.strip():
            st.warning("⚠️ Please enter a Question to extract an answer.")
            return

        with st.spinner("Analyzing passage and computing answer span..."):
            start_time = time.time()
            answer, start_char, end_char, confidence = extract_answer(
                context=context_input,
                question=question_input,
                tokenizer=tokenizer,
                model=model,
                max_answer_len=max_answer_len
            )
            elapsed_time = (time.time() - start_time) * 1000  # milliseconds

        st.markdown("---")
        st.markdown("### 🎯 Result")

        # Check for low confidence or empty answer
        if not answer or confidence < min_confidence_thresh:
            st.error("❓ **No confident answer could be found in the given context.**")
            st.caption(
                f"Top candidate confidence ({confidence * 100:.1f}%) was below the threshold ({min_confidence_thresh * 100:.0f}%). "
                "Try rephrasing your question or providing more detailed context."
            )
        else:
            # Answer Display Card
            col_ans, col_metric = st.columns([3, 1])

            with col_ans:
                st.markdown(f"""
                <div class="answer-card">
                    <div class="answer-title">Extracted Answer</div>
                    <div class="answer-text">"{answer}"</div>
                </div>
                """, unsafe_allow_html=True)

            with col_metric:
                st.metric(
                    label="Confidence Score",
                    value=f"{confidence * 100:.1f}%",
                    delta=f"{elapsed_time:.0f} ms inference"
                )
                st.progress(min(max(confidence, 0.0), 1.0))

            # Highlighted Passage Context Display
            st.markdown("#### 📖 Context with Highlighted Answer")
            before_text = context_input[:start_char]
            highlighted_text = context_input[start_char:end_char]
            after_text = context_input[end_char:]

            highlighted_html = (
                f"<span>{before_text}</span>"
                f"<mark class='highlight'>{highlighted_text}</mark>"
                f"<span>{after_text}</span>"
            )
            st.markdown(f"<div style='background-color:#F8FAFC; padding:1rem; border-radius:8px; line-height:1.6;'>{highlighted_html}</div>", unsafe_allow_html=True)

            # Extra technical details in expander
            with st.expander("🔬 View Detailed Span Metadata"):
                meta_df = pd.DataFrame([
                    {"Attribute": "Answer Text", "Value": answer},
                    {"Attribute": "Character Start Index", "Value": start_char},
                    {"Attribute": "Character End Index", "Value": end_char},
                    {"Attribute": "Raw Probability Product", "Value": f"{confidence:.6f}"},
                    {"Attribute": "Inference Time", "Value": f"{elapsed_time:.2f} ms"},
                    {"Attribute": "Model Source", "Value": load_source}
                ])
                st.table(meta_df)


if __name__ == "__main__":
    main()

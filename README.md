# 🤖 RoBERTa Question Answering Streamlit Deployment

A production-ready Streamlit application for Extractive Question Answering based on a `roberta-base` fine-tuned model on the SQuAD dataset using PyTorch and Hugging Face Transformers.

---

## 🌟 Key Features

- **Extractive QA Pipeline:** Given a text passage (context) and a question, the model extracts the precise substring answer.
- **Efficient Resource Management:** Model and tokenizer are loaded once and cached across user sessions using `@st.cache_resource`.
- **Automatic Fallback Mechanism:** 
  - Checks locally for fine-tuned weights in `./results`.
  - Automatically falls back to Hugging Face Hub (`deepset/roberta-base-squad2`) if local model files are absent.
- **Fast PyTorch Inference:** Executes forward pass under `torch.no_grad()` to optimize memory consumption and response time.
- **Confidence Metric & Span Highlighting:** Displays answer probability, inference time in milliseconds, and highlights the answer directly within the passage.
- **Interactive Preset Examples:** Comes pre-loaded with sample contexts and questions for quick demonstration.

---

## 📁 Repository Structure

```text
QA/
├── app.py              # Main Streamlit application UI & inference pipeline
├── requirements.txt    # Python dependencies for deployment
├── README.md           # Documentation & deployment instructions
├── QA.ipynb            # Jupyter Notebook used for training/fine-tuning
└── results/            # (Optional) Directory containing fine-tuned model weights
```

---

## 🚀 Quickstart - Running Locally

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Set Up Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Local Model Weights (Optional)
If you fine-tuned your model locally or in Colab:
- Create a directory named `results/` in the project root.
- Place your `config.json`, `model.safetensors` (or `pytorch_model.bin`), `tokenizer.json`, and `vocab.json` files inside `./results/`.
- If `./results/` is missing, the application will automatically fall back to downloading pre-trained weights from Hugging Face Hub.

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ Deployment Guidelines

### Option 1: Streamlit Community Cloud (Recommended)

1. **Push Code to GitHub:**
   Commit `app.py`, `requirements.txt`, and `README.md` to your GitHub repository.

2. **Deploy via Streamlit Cloud:**
   - Log in to [Streamlit Community Cloud](https://streamlit.io/cloud).
   - Click **"New app"**.
   - Select your GitHub repository, branch (`main`), and set Main file path to `app.py`.
   - Click **"Deploy!"**.

> 💡 **Note on Large Model Files:** Do NOT commit large PyTorch model binary files (>100MB) directly to GitHub unless using Git LFS. The app will smoothly default to downloading weights from Hugging Face Hub on Streamlit Cloud if local weights are not present.

---

### Option 2: Hugging Face Spaces (Streamlit SDK)

1. **Create Space:**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces).
   - Click **"Create new Space"**.
   - Select **Streamlit** as the Space SDK.

2. **Upload Code:**
   - Clone the Space repository locally or use the Web UI.
   - Upload `app.py`, `requirements.txt`, and `README.md`.
   - If hosting custom fine-tuned weights on Hugging Face Hub, update `DEFAULT_HF_MODEL` in `app.py` to point to your HF model path (e.g. `your-username/roberta-squad-qa`).

3. **Commit & Build:**
   - Hugging Face Spaces will automatically install `requirements.txt` and launch `app.py`.

---

## 🛠️ Configuration & Adjustments

In `app.py`, you can modify these key configuration parameters:

- `LOCAL_MODEL_PATH = "./results"`: Directory where local model files are located.
- `DEFAULT_HF_MODEL = "deepset/roberta-base-squad2"`: Fallback Hugging Face model repository.
- `max_answer_len`: Maximum token length allowed for extracted answer spans (configurable via UI slider).

---

## 📄 License
This project is open-source under the MIT License.

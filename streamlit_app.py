import io
import os
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from peft import PeftConfig, PeftModel
except ImportError:
    PeftConfig = None
    PeftModel = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Legal Document Summarizer",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent
LORA_ADAPTER_PATH = REPO_ROOT / "final_legal_pegasus_lora"


# ---------------------------------------------------------------------------
# Model loading (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the Pegasus base model from HuggingFace Hub and apply the LoRA adapter."""
    if PeftConfig is None or PeftModel is None:
        st.error("The `peft` library is not installed. Please add it to requirements.txt.")
        return None, None

    adapter_path = str(LORA_ADAPTER_PATH)
    if not LORA_ADAPTER_PATH.exists():
        st.error(
            f"LoRA adapter not found at `{adapter_path}`. "
            "Make sure `final_legal_pegasus_lora/` is present in the repository root."
        )
        return None, None

    try:
        config = PeftConfig.from_pretrained(adapter_path)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float32,  # CPU-safe
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)

        if torch.cuda.is_available():
            model = model.cuda()

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        return tokenizer, model
    except MemoryError:
        st.error(
            "Out of memory while loading the model. "
            "Streamlit Cloud free tier has ~1 GB RAM, which may be insufficient for this model (~2.3 GB). "
            "Consider upgrading to a paid tier or deploying on HuggingFace Spaces (free GPU available)."
        )
        return None, None
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# PDF extraction helpers
# ---------------------------------------------------------------------------
def _extract_with_pdfplumber(file_bytes: bytes) -> str:
    if pdfplumber is None:
        return ""
    parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                try:
                    words = page.extract_words() or []
                    text = " ".join(w.get("text", "") for w in words)
                except Exception:
                    text = ""
            parts.append(text or "")
    return "\n".join(parts).strip()


def _extract_with_pypdf(file_bytes: bytes) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        return ""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = _extract_with_pdfplumber(file_bytes)
    if text:
        return text
    return _extract_with_pypdf(file_bytes)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("⚖️ Legal Document Summarizer")
st.markdown(
    "Summarize legal documents using a fine-tuned **Pegasus** model with a LoRA adapter "
    "trained on legal texts."
)

# Sidebar settings
st.sidebar.header("⚙️ Settings")
max_input_length = st.sidebar.slider("Max Input Length (tokens)", 512, 2048, 1024, step=128)
max_output_length = st.sidebar.slider("Max Output Length (tokens)", 128, 1024, 512, step=64)
num_beams = st.sidebar.slider("Number of Beams", 1, 8, 4)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Note:** The first run downloads the base model (~2.3 GB) from HuggingFace Hub. "
    "Subsequent runs use the cached version."
)

# Load model with a visible spinner
with st.spinner("Loading model — this may take a few minutes on first run…"):
    tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.stop()

# Input section
st.header("📄 Input")
input_method = st.radio("Choose input method:", ("Enter text", "Upload PDF"), horizontal=True)

input_text = ""

if input_method == "Enter text":
    input_text = st.text_area("Enter legal text to summarize:", height=250)

else:  # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if "pdf_text" not in st.session_state or st.session_state.get("pdf_name") != uploaded_file.name:
            with st.spinner("Extracting text from PDF…"):
                extracted = extract_text_from_pdf(file_bytes)
            st.session_state.pdf_text = extracted
            st.session_state.pdf_name = uploaded_file.name

            if extracted.strip():
                st.success("PDF text extracted successfully!")
            else:
                st.warning(
                    "No text could be extracted from the PDF. "
                    "This may be a scanned/image-based PDF. Try copy-pasting the text manually."
                )

        input_text = st.text_area(
            "Extracted text (editable):",
            value=st.session_state.get("pdf_text", ""),
            height=250,
            key="pdf_text_area",
        )
    else:
        # Clear stale state when file is removed
        for key in ("pdf_text", "pdf_name"):
            st.session_state.pop(key, None)

# Generate button
if st.button("🔍 Generate Summary", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text or upload a PDF before generating a summary.")
    else:
        with st.spinner("Generating summary…"):
            try:
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=max_input_length,
                    truncation=True,
                )
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_length=max_output_length,
                        num_beams=num_beams,
                        length_penalty=1.0,
                        early_stopping=True,
                    )

                summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

                st.header("📝 Summary")
                st.write(summary)
            except MemoryError:
                st.error(
                    "Out of memory during generation. "
                    "Try reducing the Max Input Length or Number of Beams in the sidebar."
                )
            except Exception as exc:
                st.error(f"An error occurred during generation: {exc}")

# Footer
st.markdown("---")
st.markdown(
    "Built with [Streamlit](https://streamlit.io) · "
    "Model: [nsi319/legal-pegasus](https://huggingface.co/nsi319/legal-pegasus) + LoRA fine-tuning · "
    "Source: [GitHub](https://github.com/HackLoading/Legal-Document-Summarizer)"
)

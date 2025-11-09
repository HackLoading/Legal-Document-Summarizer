import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import os
import io
import pdfplumber
try:
    from pypdf import PdfReader
except Exception:  # pypdf may not be installed; handle gracefully
    PdfReader = None

# Get the parent directory (project root)
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(page_title="Legal Text Summarizer (Combined)", page_icon="⚖️", layout="wide")

# Title
st.title("⚖️ Legal Text Summarization (LED vs Pegasus)")
st.markdown("Summarize legal documents using both fine-tuned LED and Pegasus models.")

# Sidebar for settings
st.sidebar.header("Settings")
max_input_length = st.sidebar.slider("Max Input Length", 512, 2048, 1024)
max_output_length = st.sidebar.slider("Max Output Length", 128, 1024, 512)
num_beams = st.sidebar.slider("Number of Beams", 1, 8, 4)


# ---------- PDF extraction helpers ----------
def _extract_with_pdfplumber(file_bytes, diagnostics):
    text_parts = []
    per_page = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            source = "pdfplumber.extract_text"
            if not page_text:
                # Fallback: join words if text is None
                try:
                    words = page.extract_words() or []
                    if words:
                        page_text = " ".join([w.get("text", "") for w in words])
                        source = "pdfplumber.extract_words"
                except Exception:
                    pass
            page_text = page_text or ""
            text_parts.append(page_text)
            if diagnostics is not None:
                per_page.append({
                    "page": i + 1,
                    "chars": len(page_text),
                    "source": source,
                })
    return "\n".join(text_parts).strip(), per_page


def _extract_with_pypdf(file_bytes):
    if PdfReader is None:
        return "", []
    per_page = []
    text_parts = []
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            text_parts.append(t)
            per_page.append({
                "page": i + 1,
                "chars": len(t),
                "source": "pypdf.extract_text",
            })
    except Exception:
        return "", []
    return "\n".join(text_parts).strip(), per_page


def extract_text_from_pdf(file_bytes, enable_diagnostics=False):
    diag = {"attempts": []} if enable_diagnostics else None

    # Attempt 1: pdfplumber
    text1, per_page1 = _extract_with_pdfplumber(file_bytes, diag)
    if diag is not None:
        diag["attempts"].append({
            "method": "pdfplumber",
            "total_chars": len(text1),
            "per_page": per_page1,
        })
    if text1:
        return text1, diag

    # Attempt 2: pypdf
    text2, per_page2 = _extract_with_pypdf(file_bytes)
    if diag is not None:
        diag["attempts"].append({
            "method": "pypdf",
            "total_chars": len(text2),
            "per_page": per_page2,
        })
    if text2:
        return text2, diag

    return "", diag

# Load models and tokenizers
@st.cache_resource
def load_led_model():
    model_path = os.path.join(parent_dir, 'best-20251025T161540Z-1-001')
    tokenizer_path = model_path
    
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, use_safetensors=True)
    else:
        st.error("LED model not found.")
        return None, None
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return tokenizer, model

@st.cache_resource
def load_pegasus_model():
    model_path = os.path.join(parent_dir, 'final_legal_pegasus_lora')
    tokenizer_path = model_path
    
    if os.path.exists(model_path):
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, use_safetensors=True)
        model = PeftModel.from_pretrained(base_model, model_path, use_safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        st.error("Pegasus model not found.")
        return None, None
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return tokenizer, model

led_tokenizer, led_model = load_led_model()
pegasus_tokenizer, pegasus_model = load_pegasus_model()

if led_tokenizer is None or led_model is None or pegasus_tokenizer is None or pegasus_model is None:
    st.stop()

# Main content
st.header("Input Text")
input_method = st.radio("Choose input method:", ("Enter text", "Upload PDF"))

input_text = ""
if input_method == "Enter text":
    input_text = st.text_area("Enter legal text to summarize:", height=200)
elif input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    show_diag = st.checkbox("Show extraction diagnostics", value=False)
    if uploaded_file is not None:
        # Keep original bytes for re-extraction and multiple backends
        file_bytes = uploaded_file.getvalue()

        # Re-extract button to force re-run if user toggles diagnostics
        if st.button("Re-extract text") or "pdf_text" not in st.session_state:
            extracted_text, diag = extract_text_from_pdf(file_bytes, enable_diagnostics=show_diag)
            st.session_state.pdf_text = extracted_text
            if show_diag and diag:
                st.session_state.pdf_diag = diag
            else:
                st.session_state.pdf_diag = None

        # Feedback messages
        txt = st.session_state.get("pdf_text", "")
        if txt.strip():
            st.success("PDF text extracted successfully!")
        else:
            msg = "No text could be extracted from the PDF. If this is a text-based PDF, try the diagnostics to see per-page info."
            if PdfReader is None:
                msg += " Also install 'pypdf' for an additional extraction attempt."
            st.warning(msg)

        # Show diagnostics if requested
        if show_diag and st.session_state.get("pdf_diag"):
            with st.expander("Extraction diagnostics"):
                for attempt in st.session_state.pdf_diag.get("attempts", []):
                    st.write(f"Method: {attempt['method']} — Total chars: {attempt['total_chars']}")
                    for p in attempt.get("per_page", [])[:50]:  # cap for very large PDFs
                        st.write(f"Page {p['page']}: chars={p['chars']}, source={p['source']}")

        input_text = st.text_area("Extracted text (editable):", value=st.session_state.get("pdf_text", ""), height=200, key="pdf_text")
    else:
        if "pdf_text" in st.session_state:
            del st.session_state.pdf_text
        if "pdf_diag" in st.session_state:
            del st.session_state.pdf_diag

if st.button("Generate Summary"):
    if input_text.strip():
        with st.spinner("Generating summaries..."):
            # LED Summary
            led_model.generation_config.max_length = max_output_length
            led_model.generation_config.num_beams = num_beams
            led_model.generation_config.length_penalty = 1.0
            led_model.generation_config.early_stopping = True
            
            inputs_led = led_tokenizer(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)
            global_attention_mask = torch.zeros_like(inputs_led['attention_mask'])
            global_attention_mask[:, 0] = 1
            
            if torch.cuda.is_available():
                inputs_led = {k: v.cuda() for k, v in inputs_led.items()}
                global_attention_mask = global_attention_mask.cuda()
            
            with torch.no_grad():
                outputs_led = led_model.generate(**inputs_led, global_attention_mask=global_attention_mask)
            
            summary_led = led_tokenizer.decode(outputs_led[0], skip_special_tokens=True)
            
            # Pegasus Summary
            pegasus_model.generation_config.max_length = max_output_length
            pegasus_model.generation_config.num_beams = num_beams
            pegasus_model.generation_config.length_penalty = 1.0
            pegasus_model.generation_config.early_stopping = True
            
            inputs_pegasus = pegasus_tokenizer(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)
            
            if torch.cuda.is_available():
                inputs_pegasus = {k: v.cuda() for k, v in inputs_pegasus.items()}
            
            with torch.no_grad():
                outputs_pegasus = pegasus_model.generate(**inputs_pegasus)
            
            summary_pegasus = pegasus_tokenizer.decode(outputs_pegasus[0], skip_special_tokens=True)
        
        # Display summaries side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pegasus Model Summary")
            st.write(summary_pegasus)
        
        with col2:
            st.subheader("LED Model Summary")
            st.write(summary_led)
    else:
        st.warning("Please enter some text to summarize.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Hugging Face Transformers.")

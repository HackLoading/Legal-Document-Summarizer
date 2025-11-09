import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import pdfplumber

# Page config
st.set_page_config(page_title="Legal Text Summarizer", page_icon="⚖️", layout="wide")

# Title
st.title("⚖️ Legal Text Summarization")
st.markdown("Summarize legal documents using a fine-tuned LED model.")

# Sidebar for settings
st.sidebar.header("Settings")
max_input_length = st.sidebar.slider("Max Input Length", 512, 2048, 1024)
max_output_length = st.sidebar.slider("Max Output Length", 128, 1024, 512)
num_beams = st.sidebar.slider("Number of Beams", 1, 8, 4)

# Load model and tokenizer
@st.cache_resource
def load_model():
    # Hardcoded model directory path
    parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # model_path = os.path.join(parent_dir, r'C:\Users\Atharva Badgujar\Downloads\arglegalsumm-master\arglegalsumm-master\final_legal_pegasus_lora')
    model_path = os.path.join(parent_dir, 'best-20251025T161540Z-1-001')
    tokenizer_path = model_path  # Assume tokenizer is in the same dir
    
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        st.error("Model not found at hardcoded path. Please ensure the model is unzipped in the correct directory.")
        return None, None
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return tokenizer, model

tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.stop()

# Main content
st.header("Input Text")
input_method = st.radio("Choose input method:", ("Enter text", "Upload PDF"))

input_text = ""
if input_method == "Enter text":
    input_text = st.text_area("Enter legal text to summarize:", height=200)
elif input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        if "pdf_text" not in st.session_state:
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                st.session_state.pdf_text = text
            st.success("PDF text extracted successfully!")
        input_text = st.text_area("Extracted text (editable):", value=st.session_state.pdf_text, height=200, key="pdf_text")
    else:
        if "pdf_text" in st.session_state:
            del st.session_state.pdf_text  # Clear if no file

if st.button("Generate Summary"):
    if input_text.strip():
        with st.spinner("Generating summary..."):
            # Set generation config based on current settings
            model.generation_config.max_length = max_output_length
            model.generation_config.num_beams = num_beams
            model.generation_config.length_penalty = 1.0
            model.generation_config.early_stopping = True
            
            inputs = tokenizer(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)
            
            # LED global attention
            global_attention_mask = torch.zeros_like(inputs['attention_mask'])
            global_attention_mask[:, 0] = 1
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                global_attention_mask = global_attention_mask.cuda()
            
            with torch.no_grad():
                outputs = model.generate(**inputs, global_attention_mask=global_attention_mask)
            
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        st.header("Summary")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Hugging Face Transformers.")

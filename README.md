# Legal-Document-Summarizer

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/HackLoading/Legal-Document-Summarizer/main/streamlit_app.py)

Summarize legal documents using a fine-tuned [Pegasus](https://huggingface.co/nsi319/legal-pegasus) model with a LoRA adapter, powered by [PEFT](https://github.com/huggingface/peft).

## Features

- **Text input** — paste any legal text directly
- **PDF upload** — extracts text via `pdfplumber` / `pypdf`
- **Adjustable settings** — max input/output length and beam search via the sidebar
- **CPU-friendly** — runs on Streamlit Cloud free tier (no GPU required)

## Online Deployment (Streamlit Community Cloud)

The app is deployed at:

```
https://share.streamlit.io/HackLoading/Legal-Document-Summarizer/main/streamlit_app.py
```

### How it works

1. Streamlit Cloud clones this repository.
2. It installs Python packages from `requirements.txt` and system packages from `packages.txt`.
3. On first load the `nsi319/legal-pegasus` base model (~2.3 GB) is downloaded from HuggingFace Hub and cached via `@st.cache_resource`.
4. The LoRA adapter weights in `final_legal_pegasus_lora/` (~18 MB, already in the repo) are applied on top.

> **Memory note:** The free tier of Streamlit Cloud has ~1 GB of RAM, which may be tight for this ~2.3 GB model. If you encounter out-of-memory errors, consider upgrading to a paid Streamlit Cloud tier or deploying on [HuggingFace Spaces](https://huggingface.co/spaces) (free GPU available).

### Deploy your own copy

1. Fork this repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your fork, set the branch to `main`, and the main file path to `streamlit_app.py`.
4. Click **Deploy** — that's it!

## Local Development

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Repository Structure

```
.
├── streamlit_app.py               # Streamlit Cloud entry point
├── requirements.txt               # Python dependencies
├── packages.txt                   # System-level Debian packages
├── .streamlit/config.toml         # Streamlit configuration
├── final_legal_pegasus_lora/      # LoRA adapter weights (~18 MB)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
└── webapp/                        # Original local webapp files
    ├── app_streamlit.py
    ├── app_streamlit_pegasus.py
    └── app_streamlit_combined.py
```
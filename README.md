# Meeting Minutes Assistant

AI-powered meeting minutes generator using Streamlit and LangChain.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API Key:
Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your-api-key-here"
```

3. Run locally:
```bash
streamlit run streamlit_app.py
```

## Streamlit Cloud Deployment

1. Push code to GitHub (IMPORTANT: add `.streamlit/secrets.toml` to .gitignore)

2. On Streamlit Cloud Dashboard:
   - Connect your GitHub repository
   - Add secret in Settings → Secrets:
     ```toml
     OPENAI_API_KEY = "your-api-key-here"
     ```
   - Deploy!

## Project Structure

```
/MoM
├── .streamlit/
│   └── secrets.toml    # Local secrets (don't commit!)
├── streamlit_app.py    # Main Streamlit application
├── app.py             # Core application logic
├── requirements.txt   # Dependencies
└── README.md         # This file
```

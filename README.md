# The Dean's Dilemma

Interactive Streamlit companion to the *Dean's Dilemma* deck: a data-driven admission policy
analysis of 5,000 synthetic MBA students (entrance test, academic drivers, cut-offs, gender equity,
early-warning prediction).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Streamlit Community Cloud: new app → this repo → branch `main` → main file `app.py`.

## Layout
- `app.py` – Streamlit UI (one page per slide section)
- `analysis.py` – data prep, statistics and models
- `data/Synthetic_Dean_Dilemma_65_35.xlsx` – dataset
- `docs/Dean_Dilemma_Final.pptx` – the original deck

import pandas as pd
import streamlit as st
from pathlib import Path

from src.predict import load_model, classify_clauses, similar_clauses, summarize
from src.text_extraction import extract_text, split_into_clauses

st.set_page_config(page_title="Document Risk & Compliance Classifier", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


bundle = get_model()

st.title("AI Document Risk & Compliance Classifier")
st.caption("Classifies contract clauses as Low / Medium / High risk and flags uncertain ones for manual review.")

threshold = st.sidebar.slider("Confidence threshold", 0.5, 0.95, float(bundle["threshold"]), 0.05)
files = st.sidebar.file_uploader("Upload documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)
pasted = st.sidebar.text_area("...or paste text (separate clauses with a blank line)", height=200)

docs = {}
for f in files or []:
    docs[f.name] = extract_text(f, f.name)
if pasted.strip():
    docs["pasted_text"] = pasted

tab1, tab2, tab3 = st.tabs(["Overview", "Document detail", "Model performance"])

with tab3:
    st.subheader("Test-set results (threshold 0.8)")
    st.markdown(
        "- All clauses: **92.9%** accuracy\n"
        "- Auto-labeled (confidence ≥ 0.8): **97.0%** accuracy on **85.8%** of clauses\n"
        "- Sent to manual review: **14.2%**"
    )
    cm = Path("reports/confusion_matrix.png")
    if cm.exists():
        st.image(str(cm), width=500)

if not docs:
    with tab1:
        st.info("Upload a document or paste text in the sidebar to begin.")
    st.stop()

all_results, rows = {}, []
for name, text in docs.items():
    clauses = split_into_clauses(text)
    if not clauses:
        continue
    res = classify_clauses(bundle, clauses, threshold)
    all_results[name] = res
    s = summarize(res)
    rows.append({"document": name, "document_risk": s["document_risk"], "clauses": s["clauses"],
                 "high": s["high"], "medium": s["medium"], "low": s["low"],
                 "needs_review": s["review"], "flagged": s["review"] > 0})

if not rows:
    with tab1:
        st.warning("No readable clauses were found. Scanned PDFs (images) can't be read yet.")
    st.stop()

summary = pd.DataFrame(rows)

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(summary))
    c2.metric("Flagged for manual review", int(summary["flagged"].sum()))
    c3.metric("High-risk documents", int((summary["document_risk"] == "High").sum()))
    only_flagged = st.checkbox("Show flagged documents only")
    st.dataframe(summary[summary["flagged"]] if only_flagged else summary, use_container_width=True)

with tab2:
    doc = st.selectbox("Document", list(all_results))
    df = pd.DataFrame(all_results[doc])
    levels = st.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    review_only = st.checkbox("Only clauses needing manual review")
    view = df[df["risk"].isin(levels)]
    if review_only:
        view = view[view["needs_review"]]
    st.dataframe(view, use_container_width=True,
                 column_config={"confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0)})

    st.subheader("Explain a clause")
    if len(view):
        i = st.selectbox("Pick a clause", view.index,
                         format_func=lambda k: f"{k}: {view.loc[k, 'clause'][:80]}...")
        row = df.loc[i]
        st.write(f"**Predicted:** {row['risk']} ({row['confidence']:.0%}) | "
                 f"**Key terms:** {row['top_terms'] or 'n/a'}")
        if row["needs_review"]:
            st.warning("Below the confidence threshold: send to manual review.")
        st.write("**Most similar training clauses:**")
        for s in similar_clauses(bundle, row["clause"]):
            st.markdown(f"- *{s['clause_type']}* ({s['risk']}, similarity {s['similarity']}): {s['text'][:250]}...")
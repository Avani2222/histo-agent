"""Minimal demo UI. Run the API first, then: streamlit run ui/app.py"""
import os

import requests
import streamlit as st

API = os.getenv("HISTO_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="HistoAgent", layout="centered")
st.title("HistoAgent")
st.caption("Research prototype for breast histopathology patches. Not for clinical use.")

f = st.file_uploader("Upload a histopathology image patch", type=["png", "jpg", "jpeg", "tif", "tiff"])
if f:
    st.image(f, width=320)
    with st.spinner("Classifying, retrieving evidence, writing report..."):
        r = requests.post(f"{API}/analyze", files={"file": (f.name, f.getvalue(), f.type)}, timeout=120)
    if r.ok:
        out = r.json()
        rep = out["report"]
        st.subheader(f"Finding: {rep['finding']}  ·  confidence {rep['confidence']}")
        st.write(f"Model P(malignant) = {rep['model_probability_malignant']:.3f}")
        st.write(rep["summary"])
        if rep["evidence"]:
            st.markdown("**Evidence**")
            for c in rep["evidence"]:
                st.markdown(f"- [PMID {c['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{c['pmid']}/): {c['supports']}")
        st.markdown("**Limitations**")
        for lim in rep["limitations"]:
            st.markdown(f"- {lim}")
        st.info(rep["recommendation"])
        st.caption(f"{rep['disclaimer']}  ·  {out['timings']['total_s']:.1f}s  ·  flags: {out['flags'] or 'none'}")
    else:
        st.error(f"API error {r.status_code}: {r.text}")

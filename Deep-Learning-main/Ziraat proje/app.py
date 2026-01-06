import streamlit as st
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Load FAQs
def load_faqs():
    faqs = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(DOCS_DIR, filename), "r", encoding="utf-8") as f:
                faqs.append(f.read())
    return faqs

# Search function
def search_faqs(query, faqs, top_n=3):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(faqs + [query])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    return [(faqs[i], float(cosine_sim[i])) for i in top_indices]

# Streamlit UI
st.set_page_config(page_title=" Çalışanlar için Genel BT Problemlerine Çözüm Bulucu", layout="wide")
st.title(" Çalışanlar için Genel BT Problemlerine Çözüm Bulucu")

query = st.text_input("Problemi giriniz:")

if "history" not in st.session_state:
    st.session_state.history = []

if query.strip():
    faqs = load_faqs()
    results = search_faqs(query, faqs)
    st.session_state.history.append(query)
    st.subheader("Search Results")
    df = pd.DataFrame(results, columns=["FAQ", "Similarity"])
    st.write(df)
    if st.button("Export to CSV"):
        export_path = os.path.join(BASE_DIR, "search_results.csv")
        df.to_csv(export_path, index=False)
        st.success(f"Results exported to {export_path}")

st.sidebar.header("Arama geçmişi")
if st.session_state.history:
    for q in st.session_state.history:
        st.sidebar.write(q)
else:
    st.sidebar.write("Arama bulunmadı")

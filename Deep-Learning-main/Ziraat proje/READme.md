# IT Support FAQ Search Tool

This is a simple offline Python application that provides search functionality for an internal IT support FAQ. It allows users to enter a query and retrieves the top 3 matching paragraphs from a set of documents based on TF-IDF vectorization and cosine similarity.

## Features
- **Offline Search:** Everything runs locally; no internet connection is required once installed.
- **TF-IDF & Cosine Similarity:** Uses scikit-learn to rank document paragraphs for relevance:contentReference[oaicite:7]{index=7}:contentReference[oaicite:8]{index=8}.
- **Streamlit UI:** A minimal web interface built with Streamlit.
- **Query History:** Tracks past queries in the sidebar (in-session):contentReference[oaicite:9]{index=9}.
- **CSV Export:** Download the current top results to a CSV file:contentReference[oaicite:10]{index=10}.

## Files
- `app.py`: Main Streamlit application.
- `requirements.txt`: Python dependencies.
- `README.md`: This instructions file.
- `docs/`: Folder containing example FAQ documents in text files.

## Setup Instructions
1. Install Python (version 3.7 or higher recommended).
2. Install dependencies with pip:

import json
import random
import streamlit as st
import re
import json
import numpy as np
import torch
import polars as pl

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="İTÜ Yardım",
    page_icon="itu-yardim-icon.png",
)

# -----------------------------
# Header
# -----------------------------
st.image("itu-yardim.png", width=200)

st.markdown(
    """
    <h2 style="
        text-align: center;
        color: #4D87E5;
        margin-top: 8px;
        margin-bottom: 16px;
        font-weight: 600;
    ">
        Yardım Bileti Oluşturun
    </h2>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Preprocessing functions
# -----------------------------
def normalize_turkish_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("()/", " ")
    text = re.sub(r"[^a-zçğıöşü\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(itu_help: pl.DataFrame) -> pl.DataFrame:
    itu_help = itu_help.filter(
        pl.col("title").is_not_null() & pl.col("content").is_not_null()
    ).unique(subset=["title", "content"], maintain_order=True, keep="first")

    itu_help = itu_help.with_columns(
        pl.col("content")
        .str.split("\nBu makale size yardımcı oldu mu?\n1\n2\n3\n4\n5")
        .list.first()
        .alias("content_cleaned")
    )

    itu_help = itu_help.with_columns(
        pl.col("title").map_elements(normalize_turkish_text).alias("title_normalized"),
        pl.col("content_cleaned")
        .map_elements(normalize_turkish_text)
        .alias("content_normalized"),
    )

    itu_help = itu_help.with_columns(
        pl.col("title_normalized").alias("content_combined")
    )

    return itu_help


# -----------------------------
# Embedding functions
# -----------------------------
def get_sentence_embeddings(texts, model, tokenizer):
    encoded_input = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=128
    )

    with torch.no_grad():
        outputs = model(**encoded_input)
        last_hidden_states = outputs.last_hidden_state

    attention_mask = encoded_input["attention_mask"]
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()

    sum_embeddings = torch.sum(last_hidden_states * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

    return sum_embeddings / sum_mask


def build_faq_embeddings(itu_help, model, tokenizer):
    texts = itu_help["content_combined"].to_list()
    embeddings = get_sentence_embeddings(texts, model, tokenizer)
    return embeddings


# -----------------------------
# Similarity search function
# -----------------------------
def find_similar_FAQs(query, texts, embeddings, model, tokenizer, top_k=5):
    query_emb = get_sentence_embeddings([query], model, tokenizer)
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_idx = np.argsort(sims)[-top_k:][::-1]

    return [{"score": float(sims[i]), "content": texts[i]} for i in top_idx]


# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pl.DataFrame(records)


# -----------------------------
# Load model and tokenizer
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


# -----------------------------
# Load pipeline
# -----------------------------
@st.cache_resource
def load_pipeline(jsonl_path):
    itu_help = load_data(jsonl_path)
    itu_help = preprocess(itu_help)

    titles = itu_help["title"].to_list()
    contents = itu_help["content_cleaned"].to_list()
    texts = itu_help["content_combined"].to_list()

    tokenizer, model = load_model()

    embeddings = get_sentence_embeddings(
        texts,
        model,
        tokenizer,
    )

    return {
        "titles": titles,
        "contents": contents,
        "embeddings": embeddings,
        "model": model,
        "tokenizer": tokenizer,
    }


PIPELINE = load_pipeline("itu_help_data.jsonl")


# -----------------------------
# Search similar FAQs
# -----------------------------
def search_similar(query: str, top_k=5):
    query_norm = normalize_turkish_text(query)

    query_emb = get_sentence_embeddings(
        [query_norm],
        PIPELINE["model"],
        PIPELINE["tokenizer"],
    )

    sims = cosine_similarity(
        query_emb,
        PIPELINE["embeddings"],
    )[0]

    top_idx = np.argsort(sims)[-top_k:][::-1]

    results = []

    for i in top_idx:
        results.append(
            {
                "title": PIPELINE["titles"][i],
                "content": PIPELINE["contents"][i],
            }
        )

    return results


# -----------------------------
# Session state
# -----------------------------
if "step" not in st.session_state:
    st.session_state.step = 1

if "user_query" not in st.session_state:
    st.session_state.user_query = ""

if "results" not in st.session_state:
    st.session_state.results = []


# =================================================
# STEP 1 – Question input
# =================================================
if st.session_state.step == 1:
    st.session_state.user_query = st.text_area(
        "Sorunuzu yazın:",
        placeholder="Örn: Öğrenci belgesini nereden alabilirim?",
        height=120,
        value=st.session_state.user_query,
    )

    if st.button("İleri"):
        if not st.session_state.user_query.strip():
            st.warning("Lütfen bir soru girin.")
        else:
            st.session_state.results = search_similar(
                st.session_state.user_query, top_k=5
            )
            st.session_state.step = 2
            st.rerun()


# =================================================
# STEP 2 – Show similar questions
# =================================================
elif st.session_state.step == 2:
    st.subheader("💡 Daha önce sorulmuş şu sorulardan biri işinizi görebilir:")

    for i, item in enumerate(st.session_state.results, start=1):
        title = item.get("title", "Başlık yok")
        content = item.get("content", "")

        with st.expander(f"{i}. {title}"):
            st.markdown(content.replace("\n", "<br><br>"), unsafe_allow_html=True)

    st.divider()

    col1, col2, col3 = st.columns([1, 2.4, 1])

    with col1:
        if st.button("← Sorumu düzenle"):
            st.session_state.step = 1
            st.rerun()

    with col3:
        if st.button("Devam et →"):
            st.session_state.step = 3
            st.rerun()

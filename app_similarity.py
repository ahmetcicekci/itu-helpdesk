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


def build_faq_embeddings(itu_help: pl.DataFrame, model, tokenizer):
    title_texts = itu_help["title_normalized"].to_list()
    content_texts = itu_help["content_combined"].to_list()

    title_embeddings = get_sentence_embeddings(title_texts, model, tokenizer)
    content_embeddings = get_sentence_embeddings(content_texts, model, tokenizer)

    return {
        "title_embeddings": title_embeddings,
        "content_embeddings": content_embeddings,
        "content_idxs": list(range(len(itu_help))),
        "titles": itu_help["title"].to_list(),
        "contents": itu_help["content_cleaned"].to_list(),
        "content_combined": content_texts,
    }


# -----------------------------
# Similarity search function
# -----------------------------
def find_similar_FAQs(
    query,
    title_embeddings,
    content_embeddings,
    content_idxs,
    content_combined,
    model,
    tokenizer,
    top_k=5,
    threshold=0.7,
):
    query_embedding = get_sentence_embeddings([query], model, tokenizer)

    # --- TITLE SIMILARITY ---
    title_similarities = cosine_similarity(
        query_embedding.reshape(1, -1), title_embeddings
    )[0]

    top_k_indices = np.argsort(title_similarities)[-top_k:][::-1]
    top_k_similarities = title_similarities[top_k_indices]

    valid_indices = [
        top_k_indices[i]
        for i in range(len(top_k_indices))
        if top_k_similarities[i] > threshold
    ]

    valid_similarities = [
        top_k_similarities[i]
        for i in range(len(top_k_similarities))
        if top_k_similarities[i] > threshold
    ]

    search_methods = ["title"] * len(valid_indices)

    # --- FALLBACK TO CONTENT ---
    if len(valid_indices) < top_k:
        content_similarities = cosine_similarity(
            query_embedding.reshape(1, -1), content_embeddings
        )[0]

        fallback_indices = np.argsort(content_similarities)[::-1]
        fallback_indices = [idx for idx in fallback_indices if idx not in valid_indices]

        needed = top_k - len(valid_indices)
        fallback_indices = fallback_indices[:needed]

        valid_indices += fallback_indices
        valid_similarities += list(content_similarities[fallback_indices])
        search_methods += ["combined"] * len(fallback_indices)

    top_k_idxs = [content_idxs[i] for i in valid_indices]
    top_k_contents = [content_combined[i] for i in valid_indices]

    return valid_indices, valid_similarities, top_k_idxs, top_k_contents, search_methods


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
    # 1. Load + preprocess
    itu_help = load_data(jsonl_path)
    itu_help = preprocess(itu_help)

    titles = itu_help["title"].to_list()
    contents = itu_help["content_cleaned"].to_list()

    title_texts = itu_help["title_normalized"].to_list()
    content_texts = itu_help["content_combined"].to_list()

    # 2. Load model
    tokenizer, model = load_model()
    model.eval()

    # 3. Build embeddings
    title_embeddings = get_sentence_embeddings(
        title_texts,
        model,
        tokenizer,
    )

    content_embeddings = get_sentence_embeddings(
        content_texts,
        model,
        tokenizer,
    )

    return {
        "titles": titles,
        "contents": contents,
        "content_combined": content_texts,
        "content_idxs": list(range(len(titles))),
        "title_embeddings": title_embeddings,
        "content_embeddings": content_embeddings,
        "model": model,
        "tokenizer": tokenizer,
    }


PIPELINE = load_pipeline("itu_help_data.jsonl")


# -----------------------------
# Search similar FAQs (HYBRID)
# -----------------------------
def search_similar(query: str, top_k=5):
    query_norm = normalize_turkish_text(query)

    indices, similarities, _, _, methods = find_similar_FAQs(
        query_norm,
        PIPELINE["title_embeddings"],
        PIPELINE["content_embeddings"],
        PIPELINE["content_idxs"],
        PIPELINE["content_combined"],
        PIPELINE["model"],
        PIPELINE["tokenizer"],
        top_k=top_k,
    )

    results = []

    for idx, i in enumerate(indices):
        results.append(
            {
                "title": PIPELINE["titles"][i],
                "content": PIPELINE["contents"][i],
                "score": similarities[idx],
                "method": methods[idx],  # "title" veya "combined"
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

import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_t5():
    tokenizer = T5Tokenizer.from_pretrained("model/T5_e10a5")
    model = T5ForConditionalGeneration.from_pretrained("model/T5_e10a5")
    model.load_state_dict(torch.load("model/best_t5_headline_model_epoch_10.pt", map_location=device))
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_t5_seo():
    tokenizer = T5Tokenizer.from_pretrained("model/T5_SEO_Model")
    model = T5ForConditionalGeneration.from_pretrained("model/T5_SEO_Model")
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_bart():
    tokenizer = BartTokenizer.from_pretrained("xgboost-lover/bart-base-finetuned-inshort-news")
    model = BartForConditionalGeneration.from_pretrained("xgboost-lover/bart-base-finetuned-inshort-news")
    model.to(device)
    model.eval()
    return tokenizer, model
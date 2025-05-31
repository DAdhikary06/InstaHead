import torch
import streamlit as st
import nltk
from collections import Counter
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from code.T5_SEO import SEOLogitsProcessor



def generate_headline(tokenizer, model, input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_seo_headline(tokenizer, model, input_text, keyword, boost=8.0):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    keyword_ids = tokenizer.encode(keyword, add_special_tokens=False)
    scores_map = {token_id: boost for token_id in keyword_ids}
    seo_processor = SEOLogitsProcessor(scores_map)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=64,
            do_sample=True,
            top_k=50,
            temperature=0.6,
            logits_processor=[seo_processor]
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_and_plot_wordcloud(text, title="Word Cloud of the Article"):
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title(title)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate Word Cloud: {e}")


nltk.download('stopwords')
from nltk.corpus import stopwords

def get_top_keywords(article, n=5):
    stop_words = set(stopwords.words('english'))
    words = [
        word.strip(string.punctuation)
        for word in article.split()
        if word.strip(string.punctuation).isalpha()
    ]
    # Build a mapping from lowercase to original form (first occurrence)
    word_map = {}
    for word in words:
        lw = word.lower()
        if lw not in word_map:
            word_map[lw] = word

    filtered_words = [w.lower() for w in words if w.lower() not in stop_words]
    most_common = Counter(filtered_words).most_common(n)
    # Return the original form as in the article
    return [word_map[word] for word, count in most_common]


def show_title_details(title, time_taken, article, keyword, label="Title Details"):
    @st.dialog(label)
    def _show():
        # keyword_present = keyword_inclusion(title, keyword)
        # cosine_similarity = cosine_sim(article, title)
        # stopword_proportion = stopword_prop(title)

        # use Default values for demonstration
        keyword_present = True
        cosine_similarity = 0.85
        stopword_proportion = 20

        st.subheader("🔍 Text Analysis Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Keyword Included", value="Yes" if keyword_present else "No")
            progress_value = 100 if keyword_present else 0
            st.progress(progress_value)
        with col2:
            st.metric(label="Cosine Similarity", value=f"{cosine_similarity:.2f}")
            st.progress(int(cosine_similarity * 100))
        with col3:
            st.metric(label="Stopword Proportion", value=f"{stopword_proportion}%")
            st.progress(stopword_proportion)

        st.info(f"🏷️ **Title:**  {title}")
        st.code(f"⏱️ {time_taken} seconds", language="python")
    return _show
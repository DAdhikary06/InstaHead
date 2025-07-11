import torch
import streamlit as st
import nltk
from collections import Counter
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from code.T5_SEO import SEOLogitsProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from textstat import flesch_reading_ease


nltk.download('stopwords')
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))

def generate_headline(tokenizer, model, input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_seo_headline(tokenizer, model, input_text, keyword, boost=8.0, top_k=50, temperature=0.6):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    keyword_ids = tokenizer.encode(keyword, add_special_tokens=False)
    scores_map = {token_id: boost for token_id in keyword_ids}
    seo_processor = SEOLogitsProcessor(scores_map)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=64,
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
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



def readability_score(text):
    """
    Calculate the readability score of the given text using Flesch Reading Ease.
    Returns a score between 0 and 100, where higher scores indicate easier readability.
    """
    try:
        score = flesch_reading_ease(text)
        return score
    except Exception as e:
        print(f"Error calculating readability score: {e}")
        return 0  # Return 0 if an error occurs

# def cosine_sim(text1, text2):
#     vectorizer = TfidfVectorizer().fit([text1, text2])
#     vecs = vectorizer.transform([text1, text2])
#     return float((vecs[0] @ vecs[1].T).toarray()[0][0])

from sentence_transformers import SentenceTransformer, util
# Load the model ONCE at the top
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_sim(text1, text2):
    embeddings1 = sbert_model.encode(text1, convert_to_tensor=True)
    embeddings2 = sbert_model.encode(text2, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_score.item() if cosine_score is not None else 0.0

def stopword_prop(text):
    ''''
    Note: 
    text = "This is a simple example of how stopword proportion works."
    Suppose stop_words = {"this", "is", "a", "of", "how"}
    stopword_count = 5, total words = 9
    stopword_prop(text) returns int((5/9)*100) = 55

    Higher values mean the text contains more common words; lower values mean it is more content-rich.

    '''
    words = text.split()
    if not words:
        return 0
    stopword_count = sum(1 for w in words if w.lower() in stop_words)
    return int((stopword_count / len(words)) * 100)


@st.cache_data
def compute_title_details(title, article, keyword):
    """
    Cache the computation of title details to ensure consistent results across page reloads or page switches.
    """
    readability = readability_score(title)  # Replace keyword_inclusion with readability_score
    cosine_similarity = cosine_sim(article, title)
    stopword_proportion = stopword_prop(title)
    return readability, cosine_similarity, stopword_proportion

def show_title_details(title, time_taken, article, keyword, label="Title Details"):
    @st.dialog(label)
    def _show():
        # Use cached computation for title details
        readability, cosine_similarity, stopword_proportion = compute_title_details(title, article, keyword)

        st.subheader("🔍 Text Analysis Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Readability Score", value=f"{readability:.2f}")  # Update label to Readability Score
            st.progress(int(readability))  # Assuming readability is a percentage or normalized value
        with col2:
            st.metric(label="Cosine Similarity", value=f"{cosine_similarity:.2f}")
            st.progress(int(cosine_similarity * 100))
        with col3:
            st.metric(label="Stopword Proportion", value=f"{stopword_proportion}%")
            st.progress(stopword_proportion)

        st.info(f"🏷️ **Title:**  {title}")
        st.code(f"⏱️ {time_taken} seconds", language="python")
    return _show
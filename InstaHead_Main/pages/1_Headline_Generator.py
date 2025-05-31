import streamlit as st
from model import load_t5, load_t5_seo, load_bart
from utils import generate_headline, generate_seo_headline, generate_and_plot_wordcloud, get_top_keywords, show_title_details
import time
import matplotlib.pyplot as plt

# --- Load models ONCE at the top ---
t5_tokenizer, t5_model = load_t5()
t5_seo_tokenizer, t5_seo_model = load_t5_seo()
bart_tokenizer, bart_model = load_bart()

st.title("📰 Headline Generator")
category = st.selectbox("Select News Category", ["Sports", "Politics", "Technology", "Business", "Entertainment", "Science"])
article = st.text_area("Paste the news article here", key='article')

# Toggle for SEO-based headline
use_seo = st.checkbox("Use SEO-based headline", key="use_seo")

# --- FIX: Always define keyword, and only update session_state if user selects a new keyword ---
keyword = st.session_state.get('keyword', '')

# Show suggested keywords and SEO keyword input if toggle is on
if use_seo and article.strip():
    keywords = get_top_keywords(article)
    st.markdown("### 🔑 Top 5 Suggested Keywords")
    
    if keywords:
        # Show keywords as a segmented control
        selected_kw = st.segmented_control(
            label="Pick a keyword for SEO optimization",
            options=keywords,
            key="seg_kw"
        )
        # Only update session_state['keyword'] if the user picked a new keyword
        if selected_kw and selected_kw != st.session_state.get('keyword', ''):
            st.session_state['keyword'] = selected_kw
            keyword = selected_kw  # update local variable as well
        st.write("You can also edit the keyword below if you want.")
    else:
        st.info("No keywords could be extracted from the text.")

    # Always show the SEO keyword input when SEO is enabled
    keyword = st.text_input("Enter your SEO keyword (optional)", value=st.session_state.get('keyword', ''), key='keyword')
else:
    # If SEO is not enabled, clear the keyword from session state
    st.session_state['keyword'] = ''
    keyword = ''


if st.button("Generate Headlines"):
    if not st.session_state['article'].strip():
        st.warning("Please paste a news article first.")
    else:
        with st.spinner("Generating headlines..."):
            

            t5_input = f"summarize: {category}: {st.session_state['article']}"
            bart_input = st.session_state['article']
            # Initialize timing
            start_t5 = time.time()
            st.session_state['t5_headline'] = generate_headline(t5_tokenizer, t5_model, t5_input)
            end_t5 = time.time()

            # SEO-biased headline generation (only if toggle is on and keyword is provided)
            if use_seo and keyword.strip():
                start_t5_seo = time.time()
                st.session_state['t5_seo_headline'] = generate_seo_headline(
                    t5_seo_tokenizer, t5_seo_model, t5_input, keyword
                )
                end_t5_seo = time.time()
            else:
                st.session_state['t5_seo_headline'] = None
                end_t5_seo = None

            # BART headline generation
            start_bart = time.time()
            st.session_state['bart_headline'] = generate_headline(bart_tokenizer, bart_model, bart_input)
            end_bart = time.time()

            st.session_state['wordcloud_text'] = st.session_state['article']

            # Optionally store the elapsed times if needed for display or logging
            st.session_state['generation_times'] = {
                "T5 Time (s)": round(end_t5 - start_t5, 2),
                "T5-SEO Time (s)": round(end_t5_seo - start_t5_seo, 2) if end_t5_seo else "N/A",
                "BART Time (s)": round(end_bart - start_bart, 2)
            }

# Show results if headlines are available
if st.session_state.get('t5_headline') and st.session_state.get('bart_headline'):
    tab1, tab2, tab3 = st.tabs(["🧾 Headlines", "📊 Comparisons", "☁️ Word Cloud"])

    with tab1:
        st.success("🧠 Generated Headlines")
        

        # Show T5 Headline
        st.markdown(f"#### 📝 T5 Headline - (⏱️{st.session_state['generation_times']['T5 Time (s)']} sec)")
        if st.button(f"✨ {st.session_state['t5_headline']}", key="t5_details",type='tertiary'):
           show_title_details(
                st.session_state['t5_headline'],
                st.session_state['generation_times']['T5 Time (s)'],
                article,
                keyword,
                label="T5 Headline Details"
            )()
                
        # Show T5-SEO Headline if toggle is on and it exists
        if use_seo and st.session_state.get('t5_seo_headline'):
            st.markdown(f"#### 📝 T5-SEO Headline - (⏱️{st.session_state['generation_times']['T5-SEO Time (s)']} sec)")
            if st.button(f"✨ {st.session_state['t5_seo_headline']}", key="t5_seo_details", type='tertiary'):
                show_title_details(
                    st.session_state['t5_seo_headline'],
                    st.session_state['generation_times']['T5-SEO Time (s)'],
                    article,
                    keyword,
                    label="T5-SEO Headline Details"
                )()

        # BART Headline
        st.markdown(f"#### 📰 BART Headline - (⏱️{st.session_state['generation_times']['BART Time (s)']} sec)")
        if st.button(f"✨ {st.session_state['bart_headline']}", key="bart_details", type='tertiary'):
            show_title_details(
                st.session_state['bart_headline'],
                st.session_state['generation_times']['BART Time (s)'],
                article,
                keyword,
                label="BART Headline Details"
            )()

    with tab2:
        # Show Inference Time 
        st.markdown("### 📊 Inference Time & Word Count Comparison", unsafe_allow_html=True)
        with st.expander("🔍 See Detailed Charts"):
            col1, col2 = st.columns(2)
            with col1:

                st.markdown("### Inference Time Comparison")
                times = [st.session_state['generation_times']["T5 Time (s)"]]
                labels = ['T5']

                if use_seo and st.session_state['generation_times']["T5-SEO Time (s)"] != "N/A":
                    times.append(st.session_state['generation_times']["T5-SEO Time (s)"])
                    labels.append('T5-SEO')

                times.append(st.session_state['generation_times']["BART Time (s)"])
                labels.append('BART')   
                
                # Plot
                fig1, ax1 = plt.subplots()
                bars1 = ax1.bar(labels, times, color=['#1f77b4', '#2ca02c', '#d62728'])

                for bar in bars1:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}s",
                            ha='center', va='bottom', fontsize=10)

                ax1.set_ylabel("Time (seconds)")
                ax1.set_title("Inference Time Comparison: T5 vs BART")
                st.pyplot(fig1)

            with col2:

                # Show Word Count Comparison
                st.markdown("### Word Count Comparison")
                lengths = [len(article.split()), len(st.session_state['t5_headline'].split())]
                labels_wc = ['Article', 'T5 Title']

                if use_seo and st.session_state.get('t5_seo_headline'):
                    lengths.append(len(st.session_state['t5_seo_headline'].split()))
                    labels_wc.append('T5-SEO Title')

                lengths.append(len(st.session_state['bart_headline'].split()))
                labels_wc.append('BART Title')

                # Plot
                fig2, ax2 = plt.subplots()
                bars2 = ax2.bar(labels_wc, lengths, color=['#1f77b4', '#2ca02c', '#d62728'])

                for bar in bars2:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2, height, str(height),
                            ha='center', va='bottom', fontsize=10)

                ax2.set_ylabel("Number of Words")
                ax2.set_title("Word Count Comparison: Article vs Titles")
                st.pyplot(fig2)

    with tab3:
        st.success("Word Cloud of the Article", icon="🖼️")
        generate_and_plot_wordcloud(st.session_state.get('wordcloud_text', ''))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title('📊 Model Analytics')

# Read loss summary data
T5_loss_df = pd.read_csv('output/T5_training_loss_summary.csv')
BART_loss_df = pd.read_csv('output/BART_training_loss_summary.csv')

# Read ROUGE scores data
T5_rouge_df = pd.read_csv('output/T5_rouge_score.csv')

# Show loss summary
def show_loss_summary(df, model_name):
    st.subheader(f"✨ {model_name} - Loss Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 5.2))
        ax.plot(df['Epoch'], df['Train Loss'], marker='o', label='Train Loss')
        ax.plot(df['Epoch'], df['Validation Loss'], marker='o', label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss per Epoch')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


tab1, tab2 = st.tabs(["📉 Loss Summary", "🎯 Score Summary"])

with tab1:
    show_loss_summary(T5_loss_df, "T5")
    st.divider()
    show_loss_summary(BART_loss_df, "BART")

with tab2:
    st.subheader("🤗 Average ROUGE & BLEU Score")
    col1, col2 = st.columns(2)


    with col1:

        # Show as a table
        st.dataframe(T5_rouge_df, use_container_width=True, hide_index=True)

        metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots()
        ax.bar(x - width/2, T5_rouge_df.loc[0, metrics], width, label=T5_rouge_df.loc[0, "Model"])
        ax.bar(x + width/2, T5_rouge_df.loc[1, metrics], width, label=T5_rouge_df.loc[1, "Model"])

        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("F-measure")
        ax.set_title("Average ROUGE Score Comparison")
        ax.legend(loc='upper right', fontsize='small')
        st.pyplot(fig)

    with col2:

    # Bleu Score Comparison
        bleu_data = {
        "Model": ["T5", "BART"],
        "BLEU": [0.52, 0.47]  # Example values
        }
        bleu_df = pd.DataFrame(bleu_data)

        st.dataframe(bleu_df, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots()
        ax.bar(bleu_df["Model"], bleu_df["BLEU"], color=["#1f77b4", "#ff7f0e"])
        ax.set_ylabel("BLEU Score")
        ax.set_title("Average BLEU Score Comparison")

        st.pyplot(fig)
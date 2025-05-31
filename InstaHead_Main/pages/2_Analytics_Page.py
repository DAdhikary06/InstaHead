import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title('📊 Model Analytics')

T5_df = pd.read_csv('output/T5_training_loss_summary.csv')
BERT_df = pd.read_csv('output/BERT_training_loss_summary.csv')

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
    show_loss_summary(T5_df, "T5")
    st.divider()
    show_loss_summary(BERT_df, "BERT")

with tab2:
    st.subheader("✨ Average ROUGE Score Comparison")

    # Manually define the scores (replace with your actual values)
    rouge_data = {
        "Model": ["T5", "BERT"],
        "ROUGE-1": [0.7882, 0.4506],
        "ROUGE-2": [0.6848, 0.3555],
        "ROUGE-L": [0.7672, 0.4218]
    }
    rouge_df = pd.DataFrame(rouge_data)

    # Show as a table
    st.dataframe(rouge_df, use_container_width=True, hide_index=True)

    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, rouge_df.loc[0, metrics], width, label=rouge_df.loc[0, "Model"])
    ax.bar(x + width/2, rouge_df.loc[1, metrics], width, label=rouge_df.loc[1, "Model"])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("F-measure")
    ax.set_title("Average ROUGE Score Comparison")
    ax.legend(loc='upper right')
    st.pyplot(fig)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer
# import time
from tqdm.notebook import tqdm
import os


# Download NLTK resources
nltk.download('punkt')


# # Set the random Seeds

# In[ ]:


''' Note: Set the random seeds means that the every time you run the code, You will get same results '''
# Set random seeds for reproducibility
seed = 4
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# # Define class and their parameter

# In[ ]:


# Define paths and parameters
model_name = "t5-base"  # Can also use 't5-small' for faster training or 't5-large' for better results
max_input_length = 512  # T5 can handle up to 512 tokens
max_output_length = 64  # Headlines are usually short
batch_size = 8 # Adjust based on your GPU memory
epochs = 10
learning_rate = 5.6e-5
weight_decay_rate = 0.01
warmup_steps = 500


# In[ ]:


# define the dataset class for our news data
class NewsDataset(Dataset):
    def __init__(self, articles, headlines, categories, tokenizer, max_input_length, max_output_length):
        self.articles = articles
        self.headlines = headlines
        self.categories = categories
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]
        headline = self.headlines[idx]
        category = self.categories[idx]

        # Prepend task prefix and category for T5
        input_text = f"summarize: {category}: {article}"

        # Tokenize inputs and outputs
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
 )

        output_encoding = self.tokenizer(
            headline,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Convert the target tokens to format expected by T5
        target_ids = output_encoding["input_ids"]
        target_ids[target_ids == 0] = -100  # Ignore padding tokens in loss calculation

        return {
            "input_ids": input_encoding["input_ids"].flatten(),
            "attention_mask": input_encoding["attention_mask"].flatten(),
            "target_ids": target_ids.flatten()

        }


# # Load and Prepare Data

# In[ ]:


def load_and_prepare_data(file_paths, seed=42):
    """Load and prepare the news dataset from multiple CSV files."""
    # Load data from all file paths
    dataframes = [pd.read_csv(path) for path in file_paths]

    # Concatenate all dataframes into a single dataframe
    data = pd.concat(dataframes, ignore_index=True)

    # Shuffle the data
    df = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Dataset Shape
    print(f"DataSet Shape : {df.shape} ")
    print(f"\nColumn Names : {df.columns.to_list()} ")

    # Make sure the column name match your dataset
    # If column names are different , rename it

    column_map ={
        # Map your actual column names to required column names
        # 'your_article_column':'news_article'
        # 'your_headline_column':'news_headline'
        # 'your_category_column':'news_category'

    }

    if column_map:
     df = df.rename(columns=column_map)

    # Ensure the required columns exist
    required_columns = ['news_article', 'news_headline', 'news_category']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset. Please rename your columns.")

    # No of News Categories
    print(f"\nNumber of News Categories: \n{df['news_category'].value_counts()}")

    # Checking NaN values and Handle these values
    nan_articles = df['news_article'].isna().sum()
    nan_headlines = df['news_headline'].isna().sum()

    if nan_articles > 0 or nan_headlines > 0:
      print(f"\nFound {nan_articles} NaN articles and {nan_headlines} NaN headlines")
      print("\nRemoving rows with NaN values...")
      df = df.dropna(subset=['news_article', 'news_headline']) # remove missing or NaN values of listed columns

    # Split into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=seed, stratify=df['news_category'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df['news_category'])

    print(f"\nTrain size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}, DataFrame size: {len(df)}")

    return train_df, val_df, test_df, df


# Define the all paths to the CSV files
paths = [
    '/kaggle/input/inshort-dataset/inshort_news_data-1.csv',
    '/kaggle/input/inshort-dataset/inshort_news_data-2.csv',
    '/kaggle/input/inshort-dataset/inshort_news_data-3.csv',
    '/kaggle/input/inshort-dataset/inshort_news_data-4.csv',
    '/kaggle/input/inshort-dataset/inshort_news_data-5.csv',
    '/kaggle/input/inshort-dataset/inshort_news_data-6.csv'
]

# Load and prepare the data
train_df, val_df, test_df, df = load_and_prepare_data(paths)


# # Create a Train, Test or Validation Dataset

# In[ ]:


# Initialize tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")
# ''' After Save the trained model '''
# save_directory = "/content/drive/MyDrive/T5_Headline_Model"
# tokenizer = T5Tokenizer.from_pretrained(save_directory)

train_dataset = NewsDataset(
    train_df['news_article'].tolist(),
    train_df['news_headline'].tolist(),
    train_df['news_category'].tolist(),  # Include categories
    tokenizer,
    max_input_length,
    max_output_length
)

val_dataset = NewsDataset(
    val_df['news_article'].tolist(),
    val_df['news_headline'].tolist(),
    val_df['news_category'].tolist(),  # Include categories
    tokenizer,
    max_input_length,
    max_output_length
)

test_dataset = NewsDataset(
    test_df['news_article'].tolist(),
    test_df['news_headline'].tolist(),
    test_df['news_category'].tolist(),  # Include categories
    tokenizer,
    max_input_length,
    max_output_length
)


# # Create the DataLoader

# In[ ]:


# Import DataLoader
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Load T5 model or already pretrained model
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
# model = T5ForConditionalGeneration.from_pretrained(save_directory).to(device)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)  # Use AdamW instead of AdamWeightDecay
num_training_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)


# # Train the model

# In[ ]:


# Train the model for a single epochs
def train(model, dataloader, optimizer, scheduler, device):
    model.train() # Puts the model in training mode
    total_loss = 0
    # Iterate through the each batches
    for batch in tqdm(dataloader, desc="Training"):
        # Move input data to the save device as the model
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)

        optimizer.zero_grad() # Clear previous gradients

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# In[ ]:


# Evaluated Loss Value
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_ids
            )

            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(dataloader)


# In[ ]:


# Training loop
best_val_loss = float('inf')
train_losses = []
val_losses = []
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train(model, train_dataloader, optimizer, scheduler, device)
    val_loss = evaluate(model, val_dataloader, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_save_path = f"best_t5_headline_model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved best model to {model_save_path}")

print("\nTraining finished.")


# In[ ]:


# Create a DataFrame from the loss lists
loss_df = pd.DataFrame({
    'Epoch': list(range(1, epochs + 1)),
    'Train Loss': train_losses,
    'Validation Loss': val_losses
})

# Display the table
print("\nLoss Summary:")
print(loss_df.to_string(index=False))


# In[ ]:


loss_df.to_csv("training_loss_summary.csv", index=False)


# In[ ]:


import matplotlib.pyplot as plt

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')

plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, epochs + 1))
plt.legend()
plt.tight_layout()

# Save the plot to a file
plt.savefig('loss_plot.png')

# Show the plot
plt.show()


# # Load the best model after Training

# In[ ]:


# Load the best model for evaluation
best_model_path = f"best_t5_headline_model_epoch_{epoch+1}.pt" # Assuming the last saved model is the best
# If you want the actual best one, you'd need to track the path of `best_val_loss`'s model
# best_model_path = "path_to_the_best_model_you_saved"
try:
  model.load_state_dict(torch.load(best_model_path, map_location=device))
  print(f"Loaded model from {best_model_path}")
except FileNotFoundError:
  print(f"Warning: Best model not found at {best_model_path}. Using the model from the last epoch.")


# # Generating Headline

# In[ ]:


def generate_headline(model, tokenizer, article, category, device, max_length=64):
    """Generate a headline for a given article and category."""
    model.eval()
    input_text = f"summarize: {category}: {article}"
    input_encoding = tokenizer(
        input_text,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_encoding["input_ids"],
            attention_mask=input_encoding["attention_mask"],
            max_length=max_output_length,
            num_beams=5,
            length_penalty=0.6,
            early_stopping=True
        )

    headline = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return headline


# # Generating Headline Calculate Rouge Score for Test Set

# In[ ]:


# Evaluate on the test set and calculate ROUGE scores
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
reference_headlines = test_df['news_headline'].tolist()
generated_headlines = []

for i in tqdm(range(len(test_dataset)), desc="Generating Headlines for Test Set"):
    sample = test_dataset[i]
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

    # Need to reconstruct the article and category to pass to generate_headline
    article = test_df.iloc[i]['news_article']
    category = test_df.iloc[i]['news_category']

    generated_headline = generate_headline(model, tokenizer, article, category, device, max_output_length)
    generated_headlines.append(generated_headline)

# Calculate ROUGE scores
rouge_scores = [scorer.score(ref, gen) for ref, gen in zip(reference_headlines, generated_headlines)]
avg_rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
avg_rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
avg_rougel = np.mean([score['rougeL'].fmeasure for score in rouge_scores])
print(f"\nAverage ROUGE-1 F-measure: {avg_rouge1:.4f}")
print(f"Average ROUGE-2 F-measure: {avg_rouge2:.4f}")
print(f"Average ROUGE-L F-measure: {avg_rougel:.4f}")


# # Generate Headline and WorldCloud based on article

# In[ ]:


from wordcloud import WordCloud

def test_single_article(model, tokenizer, device, max_input_length=512, max_output_length=32):

    # --- Get input from user ---
    print("\n=== Single Article Test ===")
    article = input("Enter article text:\n")
    if not article.strip():
        print("Article cannot be empty.")
        return

    category = input("Enter category (e.g., technology, sports, automobile): ").strip()
    if not category:
        print("Category cannot be empty.")
        return
     # --- Inner function to generate and display word cloud ---
    def generate_and_plot_wordcloud(text, title="Word Cloud"):
        print(f"\nGenerating {title}...")
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(title)
            plt.show()
        except Exception as e:
            print(f"Could not generate Word Cloud: {e}")
            plt.close()
        finally:
            plt.close()
    # --- Prepare input for model ---
    input_text = f"summarize: {category}: {article}"
    input_encoding = tokenizer(
        input_text,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # --- Generate Headline ---
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_encoding["input_ids"],
            attention_mask=input_encoding["attention_mask"],
            max_length=max_output_length,
            num_beams=5,
            length_penalty=0.6,
            early_stopping=True
        )

    headline = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # # --- Generate Word Cloud ---
    generate_and_plot_wordcloud(article, title="Word Cloud of the Article")

    # Return the generated headline
    return print(f"\nGenerated Headline: {headline}")


    

# --- Run the test for a single article ---

# Run custom input test
test_single_article(model, tokenizer, device)


# # Save the Trained Model in directory

# In[ ]:


# Define the directory in the writable Kaggle path
save_directory = "/kaggle/working/T5_Headline_Model"

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    print(f"Created directory: {save_directory}")

# Save the fine-tuned model and tokenizer
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")

# To load the model later
# loaded_model = T5ForConditionalGeneration.from_pretrained(save_directory)
# loaded_tokenizer = T5Tokenizer.from_pretrained(save_directory)


# # Plots the Distribution

# In[ ]:


# Plots the distribution of token lengths for news articles and headlines.
def plot_length_distributions(df):

    if 'news_article' not in df.columns or 'news_headline' not in df.columns:
        print("Error: DataFrame must contain 'news_article' and 'news_headline' columns.")
        return

    # Calculate token lengths (using whitespace tokenization as a proxy)
    # For more accurate tokenization length, you would use the T5Tokenizer
    # Here, we use simple word count as a quicker approximation for visualization
    df['article_length'] = df['news_article'].apply(lambda x: len(str(x).split()))
    df['headline_length'] = df['news_headline'].apply(lambda x: len(str(x).split()))

    plt.figure(figsize=(12, 6))

    # Plot article length distribution
    plt.subplot(1, 2, 1)
    sns.histplot(df['article_length'], bins=50, kde=True)
    plt.title('Distribution of Article Lengths (Words)')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.xlim(0, df['article_length'].quantile(0.99)) # Limit x-axis for better visualization

    # Plot headline length distribution
    plt.subplot(1, 2, 2)
    sns.histplot(df['headline_length'], bins=30, kde=True, color='orange')
    plt.title('Distribution of Headline Lengths (Words)')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.xlim(0, df['headline_length'].quantile(0.99)) # Limit x-axis for better visualization
    plt.tight_layout()
    plt.show()

    return df

''' Re-assign value '''
df=plot_length_distributions(df)

''' Plot the distribution for train/val/test sets separately '''
# plot_length_distributions(train_df)
# plot_length_distributions(val_df)
# plot_length_distributions(test_df)

'''Plot the distribution for train/val/test sets together '''
plot_length_distributions(pd.concat([train_df, val_df, test_df]))


# ## Visualize relationships between Article complexity and Headline quality

# In[ ]:


nltk.download('punkt_tab')



# Correlation matrix between numerical features
numerical_df = df[['article_complexity', 'headline_quality', 'article_length', 'headline_length']] # This are numerical columns

# Drop potential NaN values that might interfere with correlation calculation
numerical_df = numerical_df.dropna()

plt.figure(figsize=(8, 6))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()





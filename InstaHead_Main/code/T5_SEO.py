#!/usr/bin/env python
# coding: utf-8

# ## Model Load

# In[ ]:


# model_path = "/content/drive/MyDrive/T5_e10a5"


# In[ ]:


model_path = "model/T5_SEO_Model"  # Path to the T5 SEO model


# In[ ]:


from transformers import T5ForConditionalGeneration, T5Tokenizer




tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)


# ## Load Data

# In[ ]:


import pandas as pd


# In[ ]:



actual_headline = "Tesla asked to give ₹1 cr to used car buyer for hiding damage in China"


# # Headline Generators

# In[ ]:


import torch
from transformers import LogitsProcessor
from typing import Dict


# In[ ]:


# 1. Keyword Extraction (manual/simple for demo)
article = """Tesla was ordered by a Chinese court to pay over ₹1 crore to the buyer of a used Model S car after concluding it concealed structural damage on the vehicle it sold on its official website. It was reportedly discovered part of the vehicle had been cut and welded back together. Tesla will appeal the ruling to a higher court."""
# article = df.iloc[0]['news_article']
keyword = "Chinese"  # manually extracted for this demo


# In[ ]:


# 5. Prepare input
input_text = "summarize: " + article
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)


# ## Updated LogitsProcessor with Sampling instead of Beam Search  
# Status: **Functional**  
# Conclusion: Working for single token.  

# In[ ]:


# 3. Create scores_map with token ID(s) of the keyword ===
keyword_ids = tokenizer.encode(keyword, add_special_tokens=False)
scores_map = {token_id: 8.0 for token_id in keyword_ids}  # Strong additive bias


# In[ ]:


# 4. Define Additive SEOLogitsProcessor ===
class SEOLogitsProcessor(LogitsProcessor):
    def __init__(self, scores_map: Dict[int, float]):
        self.scores_map = scores_map

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for token_id, boost in self.scores_map.items():
            scores[:, token_id] += boost
        return scores


# In[ ]:


# 5. Generate WITHOUT SEO biasing (baseline) ===
seo_processor = SEOLogitsProcessor(scores_map)
# output_ids_plain = model.generate(
#     input_ids,
#     max_length=20,
#     do_sample=True,
#     top_k=50,
#     temperature=0.9
# )
# title_plain = tokenizer.decode(output_ids_plain[0], skip_special_tokens=True)


# In[ ]:


# 6. Generate WITH SEO biasing ===
output_ids_seo = model.generate(
    input_ids,
    max_length=20,
    do_sample=True,
    top_k=50,
    temperature=0.6,
    logits_processor=[seo_processor]
)
title_seo = tokenizer.decode(output_ids_seo[0], skip_special_tokens=True)


# In[ ]:


# === 7. Show Results ===
# # print("🔹 Without SEO Biasing:", title_plain)
# print("🔹 With SEO Biasing   :", title_seo)
# print("🔹 Keyword Biased Toward:", keyword)


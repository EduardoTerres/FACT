import torch
import clip
from pathlib import Path
from tqdm import tqdm
import numpy as np

# 1. Load CLIP model (for example, CLIP with RN50 architecture)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# 2. Load the list of words from the file
words_file = "./vocab/concreteness_dictionary.txt"
with open(words_file, "r") as f:
    words = [line.strip() for line in f.readlines()]

# 3. Compute embeddings for each word
embeddings = []

# Create a tensor of text inputs (tokenized and transformed for the CLIP model)
text_inputs = clip.tokenize(words).to(device)

# Compute embeddings in batches
batch_size = 64  # Adjust based on your memory capacity
for i in tqdm(range(0, len(words), batch_size), desc="Processing words"):
    batch_text = text_inputs[i:i + batch_size]
    
    # Get the CLIP text features
    with torch.no_grad():
        text_features = model.encode_text(batch_text)
    
    # Normalize embeddings (L2 normalization)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    
    embeddings.append(text_features.cpu())

# 4. Concatenate all embeddings
embeddings = torch.cat(embeddings, dim=0)

# 5. Save the embeddings to a .pth file
output_file = "./vocab/embeddings_clip_ViT_concreteness.pth"
torch.save(embeddings, output_file)

print(f"Saved embeddings to {output_file}")
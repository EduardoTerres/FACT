import torch
import os
from pathlib import Path
from sparse_autoencoder.sparse_autoencoder import SparseAutoencoder  
import numpy as np

def get_concept_name_similarity_matrix(vocab_specific_embedding, dic_vec):
    vocab_specific_embedding = vocab_specific_embedding.to(
        torch.float32)
    dic_vec /= dic_vec.norm(dim=0, keepdim=True)
    similarities = torch.matmul(
        vocab_specific_embedding, dic_vec)
    return similarities.detach().cpu()


vocab_txt_path = os.path.join('./vocab', f"concreteness_dictionary.txt")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load wordnet vocab
wordnet_vocab_dir = "./vocab/embeddings_clip_ViT_concreteness.pth"
vocab_specific_embedding = torch.load(wordnet_vocab_dir).to(device)

sae_path = '/home/eterres/Discover-then-Name/SAE/SAEImg/cc3m/clip_ViT-B16/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/sae_checkpoints/sparse_autoencoder_final.pt'
autoencoder = SparseAutoencoder(n_input_features=512, n_learned_features=4096, n_components=1).to(device)
state_dict = torch.load(sae_path, map_location=device)
autoencoder.load_state_dict(state_dict)

dic_vec = autoencoder.decoder.weight.detach().squeeze()
concept_name_similarity_matrix = get_concept_name_similarity_matrix(
    vocab_specific_embedding,
    dic_vec,
)
all_concept_names = np.genfromtxt(vocab_txt_path, dtype=str, delimiter='\n', autostrip=False)
top_concept_idxs = concept_name_similarity_matrix.argmax(axis=0)
top_concept_values = torch.max(concept_name_similarity_matrix, axis=0).values

with open('./analysis/concept_names_vit.csv', 'w') as f:
    for idx in range(top_concept_idxs.shape[0]):
        name = all_concept_names[top_concept_idxs[idx]]
        cosine_sim = top_concept_values[idx].item()
        f.write(f"{idx},{name},{cosine_sim}\n")



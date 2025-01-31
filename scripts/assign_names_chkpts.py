from dncbm import method_utils
from dncbm import arg_parser
from dncbm import utils
import torch
import os
import torch
from tqdm import tqdm
from sparse_autoencoder.sparse_autoencoder.autoencoder.model import SparseAutoencoder  
import numpy as np

def get_concept_name_similarity_matrix(vocab_specific_embedding, dic_vec):
    vocab_specific_embedding = vocab_specific_embedding.to(
        torch.float32)
    dic_vec /= dic_vec.norm(dim=0, keepdim=True)
    similarities = torch.matmul(
        vocab_specific_embedding, dic_vec)
    return similarities.detach().cpu()


vocab_txt_path = os.path.join('./vocab', f"wordnet.txt")

checkpoints_dir = "/home/eterres/Discover-then-Name/SAE/SAEImg/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/sae_checkpoints"
checkpoints = os.listdir(checkpoints_dir)
checkpoints.sort(key=lambda x: int(x.split('_')[-1][:-3] if x.split('_')[-1][:-3] != 'final' else 0))
print(checkpoints)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load wordnet vocab
wordnet_vocab_dir = './vocab/WORDNET_embeddings_clip_RN50.pth'
vocab_specific_embedding = torch.load(wordnet_vocab_dir).to(device)


for iter, chkpt_name in tqdm(enumerate(checkpoints)):
    print('Loading checkpoint:', chkpt_name)
    autoencoder = SparseAutoencoder(n_input_features=1024, n_learned_features=8192, n_components=1).to(device)
    state_dict = torch.load(os.path.join(checkpoints_dir, chkpt_name), map_location=device)
    autoencoder.load_state_dict(state_dict)

    dic_vec = autoencoder.decoder.weight.detach().squeeze()
    concept_name_similarity_matrix = get_concept_name_similarity_matrix(
        vocab_specific_embedding,
        dic_vec,
    )
    all_concept_names = np.genfromtxt(vocab_txt_path, dtype=str, delimiter='\n', autostrip=False)
    top_concept_idxs = concept_name_similarity_matrix.argmax(axis=0)
    top_concept_values = torch.max(concept_name_similarity_matrix, axis=0).values

    with open(os.path.join(f"wordnet_chkpts/concept_names{2*iter}.csv"), "w") as f:
        for idx in range(top_concept_idxs.shape[0]):
            name = all_concept_names[top_concept_idxs[idx]]
            cosine_sim = top_concept_values[idx]
            # print(f"{idx},{name},{cosine_sim}")
            f.write(f"{idx},{name},{cosine_sim}\n")



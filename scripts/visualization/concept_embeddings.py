import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from dncbm import arg_parser, method_utils
from dncbm.utils import common_init

from sparse_autoencoder.sparse_autoencoder.autoencoder.model import SparseAutoencoder  


def visualize_concept_embeddings(concept_embeddings_np):
    print('Shape', concept_embeddings_np.shape)
    mean = concept_embeddings_np.mean(axis=0)
    std = concept_embeddings_np.std(axis=0)

    print("Shape of mean:", mean.shape)
    # Order the concept strengths based on the mean
    # ordered_indices = np.argsort(mean)
    ordered_indices = np.arange(len(mean))

    # Order the std array based on the ordered indices
    ordered_std = std[ordered_indices]

    # Plot the mean and standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(mean[ordered_indices], label='Mean', color='blue')
    plt.fill_between(range(len(mean)), mean[ordered_indices] - ordered_std, mean[ordered_indices] + ordered_std, color='blue', alpha=0.2, label='Mean ± Std Dev')
    plt.title("Mean and Standard Deviation of Concept Strengths")
    plt.xlabel("Ordered Concepts")
    plt.ylabel("Strength")
    plt.legend()
    plt.savefig("imgs/mean_std_concept_embeddings.png")
    plt.close()

    return mean


parser = arg_parser.get_common_parser()
args = parser.parse_args()
common_init(args)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the SAE weights file
dir = "/home/eterres/Discover-then-Name/probe/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/cc3m/train_val/all_concepts.pth"
concept_strenghts = torch.load(dir)

print("Loaded concepts successfully.")

# Save concept_strengths to a file
# Save to numpy detach first

mean = visualize_concept_embeddings(concept_strenghts)

mean = mean.detach().cpu().numpy()
# torch.save(concept_strenghts, "/home/eterres/Discover-then-Name/cc3m_concept_strenghts.pth")
np.save("/home/eterres/Discover-then-Name/cc3m_concept_strenghts.npy", mean)
print("Saved concept strengths to concept_strengths.npy")



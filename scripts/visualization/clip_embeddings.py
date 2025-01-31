import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from dncbm import arg_parser, method_utils
from dncbm.utils import common_init

from sparse_autoencoder.sparse_autoencoder.autoencoder.model import SparseAutoencoder  


def visualize_clip_embeddings(clip_embeddings_np):
    mean = clip_embeddings_np.mean(axis=0)
    std = clip_embeddings_np.std(axis=0)

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
    plt.savefig("imgs/mean_std_clip_embeddings.png")
    plt.close()


parser = arg_parser.get_common_parser()
args = parser.parse_args()
common_init(args)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the SAE weights file
weights_path = "SAE/SAEImg/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/sae_checkpoints/sparse_autoencoder_final.pt"  # Update with the actual filename if different

# Initialize the model
autoencoder_input_dim = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]]
n_learned_features = int(autoencoder_input_dim * args.expansion_factor)
model = SparseAutoencoder(n_input_features=autoencoder_input_dim, n_learned_features=n_learned_features, n_components=len(args.hook_points)).to(args.device)

# Check if the weights file exists
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"SAE weights file not found at: {weights_path}")

# Load the weights
print(f"Loading SAE weights from {weights_path}...")
state_dict = torch.load(weights_path, map_location=args.device)
model.load_state_dict(state_dict)

# Move model to the specified device
print("SAE weights loaded successfully.")

num_input_nodes = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]]
num_concepts = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]
                                                ] * args.expansion_factor
num_classes = args.probe_nclasses

# train_data = torch.load(
#     osp.join(args.probe_data_dir_activations["img"], "train.pth"))
# train_val_data = torch.load(
#     osp.join(args.probe_data_dir_activations["img"], "train_val.pth"))
# test_data = torch.load(
#     osp.join(args.probe_data_dir_activations["img"], "val.pth"))
# print(f"Getting {args.probe_dataset} features from: {args.probe_features_save_dir}")

data_dir = args.data_dir_activations[args.modality]
cc3m = torch.load(osp.join(data_dir, "train"))

visualize_clip_embeddings(cc3m)
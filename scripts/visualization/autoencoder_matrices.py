import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dncbm import arg_parser
from dncbm.utils import common_init

from sparse_autoencoder.sparse_autoencoder import SparseAutoencoder  

parser = arg_parser.get_common_parser()
args = parser.parse_args()
common_init(args)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the SAE weights file
weights_path = "SAE/SAEImg/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/sae_checkpoints/sparse_autoencoder_final.pt"  # Update with the actual filename if different

# Initialize the model
autoencoder_input_dim = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]]
print('HOOK_POINTS', len(args.hook_points))
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

# encoder_weights = model.encoder._weight[0]

# Convert the encoder weights to a numpy array
# encoder_weights_np = encoder_weights.detach().cpu().numpy()
# encoder_weights_np = np.log(encoder_weights_np + 1)

# Visualize the encoder weights
# plt.imshow(encoder_weights_np, cmap='viridis', aspect='auto')
# plt.colorbar()
# plt.title("Encoder Weights Visualization")
# plt.xlabel("Neurons")
# plt.ylabel("Features")
# plt.savefig("imgs/encoder_weights_visualization.png")
# plt.close()

# decoder_weights = model.decoder._weight[0]

# Convert the decoder weights to a numpy array
# decoder_weights_np = decoder_weights.detach().cpu().numpy()
# decoder_weights_np = np.log(decoder_weights_np + 1)


# Visualize the decoder weights
# plt.imshow(decoder_weights_np, cmap='viridis', aspect='auto')
# plt.colorbar()
# plt.title("Decoder Weights Visualization")
# plt.xlabel("Features")
# plt.ylabel("Neurons")
# plt.savefig("imgs/decoder_weights_visualization.png")
# plt.close()

# indices = np.argmin(decoder_weights_np.mean(axis=-1))
# print("Indices of elements with mean less than -0.2:", indices)

# Visualize the pre-encoder bias
pre_encoder_bias = model.pre_encoder_bias._bias_reference[0]
pre_encoder_bias_np = pre_encoder_bias.detach().cpu().numpy()

# Visualize the encoder bias
plt.figure(figsize=(10, 6))
plt.plot(pre_encoder_bias_np)
plt.title("Pre-encoder Bias Visualization")
plt.xlabel("Index")
plt.ylabel("Bias Value")
plt.savefig("imgs/pre_encoder_bias_visualization.png")
plt.close()

# Visualize bias of the encoder
encoder_bias = model.encoder._bias[0].detach().cpu().numpy()
plt.figure(figsize=(10, 6))
# plt.scatter(encoder_bias, range(len(encoder_bias)))
plt.plot(encoder_bias)
plt.title("Encoder Bias Visualization")
plt.xlabel("Index")
plt.ylabel("Bias Value")
plt.savefig("imgs/encoder_bias_visualization.png")
plt.close()

# Visualize post-decoder bias
post_decoder_bias = model.post_decoder_bias._bias_reference[0]
post_decoder_bias_np = post_decoder_bias.detach().cpu().numpy()

# Visualize the decoder bias
plt.figure(figsize=(10, 6))
plt.plot(post_decoder_bias_np)
plt.title("Post-decoder Bias Visualization")
plt.xlabel("Index")
plt.ylabel("Bias Value")
plt.savefig("imgs/post_decoder_bias_visualization.png")
plt.close()

# Multiplicatoin W_e b
b = model.encoder._weight[0] @ pre_encoder_bias
b_np = b.detach().cpu().numpy()
plt.figure(figsize=(10, 6))
plt.plot(b_np)
plt.title("w_e @ b Visualization")
plt.xlabel("Index")
plt.ylabel("Bias Value")
plt.savefig("imgs/We_b_decoder_bias_visualization.png")
plt.close()

# - W_e @ b + b_e
b_e = model.encoder._bias[0]
b_resta = b_e - b
b_resta_np = b_resta.detach().cpu().numpy()
plt.figure(figsize=(10, 6))
plt.plot(b_resta_np)
plt.title("b_e - w_e @ b Visualization")
plt.xlabel("Index")
plt.ylabel("Bias Value")
plt.savefig("imgs/b_e-W_e_b_decoder_bias_visualization.png")
plt.close()






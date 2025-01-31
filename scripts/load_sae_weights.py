import os
import torch
from sparse_autoencoder.sparse_autoencoder.autoencoder.model import SparseAutoencoder  

def load_sae_weights(model_path, device):
    """
    Loads the SAE model weights from a given path.

    Args:
        model_path (str): Path to the .pth or .pt file containing SAE weights.
        device (str): Device to load the model on ('cpu' or 'cuda').
    
    Returns:
        torch.nn.Module: SAE model with loaded weights.
    """
    # Initialize the model
    model = SparseAutoencoder() 
    
    # Check if the weights file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SAE weights file not found at: {model_path}")
    
    # Load the weights
    print(f"Loading SAE weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Move model to the specified device
    model.to(device)
    print("SAE weights loaded successfully.")
    
    return model

if __name__ == "__main__":
    # Path to the SAE weights file
    weights_path = "SAE/SAEImg/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/sae_checkpoints/sparse_autoencoder_final.pt"  # Update with the actual filename if different
    
    # Specify the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model
    sae_model = load_sae_weights(weights_path, device)
    
    # Print model summary (optional)
    print(sae_model)

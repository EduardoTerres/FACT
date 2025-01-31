import os
import torch
from torch.utils.data import TensorDataset
from dncbm import arg_parser
from sparse_autoencoder.sparse_autoencoder import SparseAutoencoder
from tqdm import tqdm
import os.path as osp
from dncbm.utils import common_init, get_sae_ckpt


def save_concept_strengths_per_group(args):
    """
    Save concept strengths for each group (W@W, W@L, L@W, L@L) using saved features.

    Args:
        args: Command-line arguments.
    """
    group_paths = {
        "W@W": os.path.join(args.probe_data_dir_activations["img"], args.probe_split, "group_features", "W@W_val.pth"),
        "W@L": os.path.join(args.probe_data_dir_activations["img"], args.probe_split, "group_features", "W@L_val.pth"),
        "L@W": os.path.join(args.probe_data_dir_activations["img"], args.probe_split, "group_features", "L@W_val.pth"),
        "L@L": os.path.join(args.probe_data_dir_activations["img"], args.probe_split, "group_features", "L@L_val.pth")
    }

    # Output directory for saving concept strengths
    output_dir = os.path.join(args.probe_cs_save_dir, args.probe_split, "group_concepts")
    os.makedirs(output_dir, exist_ok=True)

    # Load the autoencoder
    autoencoder_input_dim = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]]
    n_learned_features = int(autoencoder_input_dim * args.expansion_factor)
    autoencoder = SparseAutoencoder(
        n_input_features=autoencoder_input_dim,
        n_learned_features=n_learned_features,
        n_components=len(args.hook_points)
    ).to(args.device)
    autoencoder = get_sae_ckpt(args, autoencoder)

    for group, features_path in group_paths.items():
        print(f"Processing group: {group}")
        print(f"Loading features from: {features_path}")

        # Load saved features
        all_features = torch.load(features_path)
        dataset = TensorDataset(all_features)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=False)

        all_concepts = None

        # Pass features through the autoencoder to compute concepts
        with torch.no_grad():
            for features in tqdm(loader, desc=f"Processing {group} batches", unit="batch"):
                features = features[0].to(args.device)
                concepts, reconstructions = autoencoder(features)
                concepts = concepts.squeeze()

                if all_concepts is None:
                    all_concepts = concepts.detach().cpu()
                else:
                    all_concepts = torch.vstack((all_concepts, concepts.detach().cpu()))

        # Save computed concepts
        concept_save_path = os.path.join(output_dir, f"{group}_concepts.pth")
        torch.save(all_concepts, concept_save_path)
        print(f"Saved concepts for {group} at: {concept_save_path}")


if __name__ == "__main__":
    # Initialize arguments
    parser = arg_parser.get_common_parser()
    args = parser.parse_args()
    common_init(args)

    # Save concept strengths for each group
    save_concept_strengths_per_group(args)
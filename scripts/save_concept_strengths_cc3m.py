import os  
import torch
import os
from torch.utils.data import TensorDataset
from dncbm import arg_parser
from sparse_autoencoder.sparse_autoencoder import SparseAutoencoder
from tqdm import tqdm
import os.path as osp

from dncbm.utils import common_init, get_sae_ckpt


def save_concept_strengths(args, is_cc3m=False):
    features_path = "/home/eterres/Discover-then-Name/data/activations_img/cc3m/clip_RN50/out/train_val"
    all_features = torch.load(features_path)

    dataset = TensorDataset(all_features)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=False)

    autoencoder_input_dim = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]]
    n_learned_features = int(autoencoder_input_dim * args.expansion_factor)
    autoencoder = SparseAutoencoder(n_input_features=autoencoder_input_dim, n_learned_features=n_learned_features, n_components=len(args.hook_points)).to(args.device)

    autoencoder = get_sae_ckpt(args, autoencoder)
    all_concepts = None

    len = len(loader)
    with torch.no_grad():
        for features in tqdm(loader[:len]):
            features = features[0].to(args.device)
            concepts, reconstructions = autoencoder(features)
            concepts, reconstructions = concepts.squeeze(), reconstructions.squeeze()
           
            if all_concepts is None:
                all_concepts = concepts.detach().cpu()
            else:
                all_concepts = torch.vstack((all_concepts, concepts.detach().cpu()))

    whole_all_concepts_fname = os.path.join("/home/eterres/Discover-then-Name/probe/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/train_val/all_concepts.pth")
    os.makedirs(osp.dirname(whole_all_concepts_fname), exist_ok=True)
    torch.save(all_concepts, whole_all_concepts_fname)

    print(f"Saved concepts at: {whole_all_concepts_fname}")

if __name__ == "__main__":
    parser = arg_parser.get_common_parser()
    args = parser.parse_args()
    common_init(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_concept_strengths(args, is_cc3m = True)

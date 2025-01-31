import os
import os.path as osp
import copy

import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
import torch.utils.data

from dncbm import arg_parser
from dncbm.utils import common_init
from nltk.corpus import wordnet as wn
import pandas as pd


def load_concepts():
    import pandas as pd
    dir = "/home/eterres/Discover-then-Name/analysis/concreteness/"

    file1_path = 'Brysbaert.xlsx'
    file1_data = pd.read_excel(dir+file1_path, sheet_name=0)

    file2_path = 'concept_names.csv'
    file2_data = pd.read_csv(dir+file2_path, names=['Neuron', 'Word', 'Similarity'])

    selected_words = file2_data['Word']
    filtered_data = file1_data[file1_data['Word'].isin(selected_words)][['Word', 'Conc.M']]

    # Merge the filtered data with the similarity data
    merged_data = pd.merge(filtered_data, file2_data, on='Word')

    sorted_merged_data = merged_data.sort_values(by='Conc.M')
    concrete_concepts = sorted_merged_data[len(sorted_merged_data) // 2:]
    abstract_concepts = sorted_merged_data[:len(sorted_merged_data) // 2]

    abstract_neurons = abstract_concepts['Neuron'].values
    concrete_neurons = concrete_concepts['Neuron'].values

    abstract_neurons = sorted_merged_data[sorted_merged_data['Conc.M'] > 1.0]['Neuron'].values

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # return torch.tensor(abstract_neurons).to(device), torch.tensor(concrete_neurons).to(device)
    
    return abstract_neurons, concrete_neurons


def main(args):
    num_input_nodes = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]]
    num_concepts = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]] * args.expansion_factor
    num_classes = args.probe_nclasses

    probe_dir = os.path.join(args.probe_cs_save_dir, args.probe_config_name)
    checkpoint_save_path = osp.join(probe_dir, "on_concepts_ckpts")

    train_data = torch.load(
        osp.join(args.probe_cs_save_dir, "train", "all_concepts.pth"))
    train_val_data = torch.load(
        osp.join(args.probe_cs_save_dir, "train_val", "all_concepts.pth"))
    test_data = torch.load(
        osp.join(args.probe_cs_save_dir, "val", "all_concepts.pth"))
    print(f"Getting {args.probe_dataset} concepts from: {args.probe_cs_save_dir}")

    train_labels = torch.load(
        osp.join(args.probe_labels_dir["img"], "all_labels_train.pth"))
    train_val_labels = torch.load(
        osp.join(args.probe_labels_dir["img"], "all_labels_train_val.pth"))
    test_labels = torch.load(
        osp.join(args.probe_labels_dir["img"], "all_labels_val.pth"))
    
    print('train', train_data.shape)
    print('train_val_shape', train_val_data.shape)
    print('test_data', test_data.shape)

    # Compute mean and std of the training data
    test_mean = test_data.mean(dim=0)
    test_mean = test_mean.detach().cpu().numpy()
    test_std = test_data.std(dim=0)
    test_std = test_std.detach().cpu().numpy()

    print('test_mean', test_mean.shape)
    # Create pandas dataframe with test_mean and its index
    test_mean_df = pd.DataFrame(zip(range(test_mean.shape[0]), test_mean, test_std), columns=['Neuron', 'mean', 'std'])
    output_csv_path = f"./artists_data/{args.probe_dataset}_test_mean_std.csv"
    test_mean_df.to_csv(output_csv_path, index=False)

parser = arg_parser.get_common_parser()
args = parser.parse_args()
common_init(args)
main(args)

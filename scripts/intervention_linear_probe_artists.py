import os
import os.path as osp
import copy

import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
import torch.utils.data

import statistics

from dncbm import arg_parser
from dncbm.utils import common_init
import torchmetrics
from nltk.corpus import wordnet as wn
import pandas as pd




class ImageWiseCoverageEnergyPercent(torchmetrics.Metric):
    """
    Metric to compute, per image, how many concepts are needed to reach energy_percent of the original logit value.
    Concepts are selected greedily based on contribution to logit.
    Two modes are provided: Contribution to absolute value of the logit, and contribution to the positive part of the logit.
    """

    def __init__(self, energy_percent, use_ground_truth=False, use_only_positive=True):
        super().__init__()
        self.energy_percent = energy_percent
        self.use_ground_truth = use_ground_truth
        self.use_only_positive = use_only_positive
        self.add_state("coverage", default=[])
        self.add_state("coverage_fraction", default=[])

    def update(self, model, inputs, preds, target):
        if self.use_ground_truth:
            target_idx = target
        else:
            target_idx = torch.argmax(preds)
        contribs = model.weight.data[target_idx] * inputs
        assert len(contribs.shape) == 1
        if self.use_only_positive:
            contribs = contribs.clamp(min=0)
        else:
            contribs = contribs.abs()
        contribs = contribs.sort(descending=True).values
        contribs_cumsum = contribs.cumsum(dim=0)
        sufficient_position = torch.where(
            contribs_cumsum >= self.energy_percent*contribs.sum())[0][0]
        self.coverage.append(sufficient_position.item())
        self.coverage_fraction.append(
            (sufficient_position/contribs.shape[0]).item())

    def compute(self):
        if len(self.coverage) == 0:
            return None
        return statistics.fmean(self.coverage)


def eval_model(model, loader, num_classes, classification_loss_name, sparsity_loss_lambda, device, eval_coverage=False):
    """
    Evaluates model accuracy and sparsity during and after training.
    """
    with torch.no_grad():
        model.eval()
        accuracy_top1 = torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes, top_k=1, average="micro").to(device)
        if args.probe_nclasses >= 5:
            accuracy_top5 = torchmetrics.classification.MulticlassAccuracy(
                num_classes=num_classes, top_k=5, average="micro").to(device)
        coverage_energy_wise = {}
        coverage_energy_wise_energies = [0.9, 0.95, 0.99]
        for energy in coverage_energy_wise_energies:
            coverage_energy_wise[energy] = {}
            coverage_energy_wise[energy]["positive"] = ImageWiseCoverageEnergyPercent(
                energy_percent=energy, use_ground_truth=False, use_only_positive=True).to(device)
            coverage_energy_wise[energy]["absolute"] = ImageWiseCoverageEnergyPercent(
                energy_percent=energy, use_ground_truth=False, use_only_positive=False).to(device)
        total_loss = 0
        total_classification_loss = 0
        total_sparsity_loss = 0
        total_batches = 0
        if classification_loss_name == "CE":
            classification_loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        sparsity_loss_fn = torch.nn.L1Loss()
        for batch_idx, (test_X, test_y) in enumerate(tqdm(loader)):
            test_X = test_X.to(model.weight.dtype).to(device)
            test_y = test_y.long().to(device)
            out = model(test_X)
            classification_loss = classification_loss_fn(out, test_y)
            sparsity_loss = sparsity_loss_fn(
                model.weight.flatten(), torch.zeros_like(model.weight.flatten()))
            total_sparsity_loss += sparsity_loss.item()
            loss = classification_loss + sparsity_loss_lambda*sparsity_loss
            total_classification_loss += classification_loss.item()
            total_loss += loss.item()
            accuracy_top1.update(out, test_y)
            if args.probe_nclasses >= 5:
                accuracy_top5.update(out, test_y)
            if eval_coverage:
                for img_idx in range(test_X.shape[0]):
                    img = test_X[img_idx]
                    target = test_y[img_idx]
                    for energy in coverage_energy_wise_energies:
                        coverage_energy_wise[energy]["positive"].update(
                            model, img, out[img_idx], target)
                        coverage_energy_wise[energy]["absolute"].update(
                            model, img, out[img_idx], target)
            total_batches += 1
        zero_count_dict = {}
        zero_count = torch.where(model.weight.data.flatten() == 0)[0].shape[0]
        print(f"Zero count: {zero_count}")
        zero_count_dict[0] = zero_count
        for tol in [1e-9, 1e-6, 1e-3, 1e-1]:
            zero_count_tol = torch.where(
                torch.abs(model.weight.data.flatten()) < tol)[0].shape[0]
            print(
                f"Zero count for tol {tol}: {zero_count_tol}, out of {model.weight.data.flatten().shape[0]}")
            zero_count_dict[tol] = zero_count_tol
        acc_top1 = accuracy_top1.compute()
        if args.probe_nclasses >= 5:
            acc_top5 = accuracy_top5.compute()
        else:
            acc_top5 = 0.0
        if eval_coverage:
            coverage_energy_wise_metric_vals = {}
            for energy in coverage_energy_wise_energies:
                coverage_energy_wise_metric_vals[energy] = {}
                coverage_energy_wise_metric_vals[energy]["positive"] = coverage_energy_wise[energy]["positive"].compute(
                )
                coverage_energy_wise_metric_vals[energy]["absolute"] = coverage_energy_wise[energy]["absolute"].compute(
                )
                print(
                    f"Coverage for energy {energy} positive: {coverage_energy_wise_metric_vals[energy]['positive']}")
                print(
                    f"Coverage for energy {energy} absolute: {coverage_energy_wise_metric_vals[energy]['absolute']}")
        avg_loss = total_loss/total_batches
        avg_class_loss = total_classification_loss/total_batches
        avg_sparse_loss = total_sparsity_loss/total_batches
        model.train()
    if eval_coverage:
        return avg_loss, avg_class_loss, avg_sparse_loss, acc_top1, acc_top5, coverage_energy_wise_metric_vals, zero_count_dict
    return avg_loss, avg_class_loss, avg_sparse_loss, acc_top1, acc_top5, None, zero_count_dict


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


def filter_artists(use_artists):
    wordnet = pd.read_csv('wordnet_depths.csv')

    chkpts_dir = './wordnet_chkpts/'
    final_chpkt = os.path.join(chkpts_dir, 'concept_names_0.csv')
    final_concept_names = pd.read_csv(final_chpkt, names=['Neuron', 'Word', 'Similarity'])

    # Join the final concept names with wordnet depths
    final_concept_names = final_concept_names.merge(wordnet, on='Word', how='inner')

    artist_synset = wn.synsets('artist')[0]
    def filter(word):
        instance = wn.synsets(word.replace(' ', '_'))[0].instance_hypernyms()
        if instance:
            hypernyms = instance[0].hypernyms()
            if hypernyms:
                hypernyms = hypernyms[0]
                if hypernyms == artist_synset:
                    return not use_artists
        return use_artists

    artists_df = final_concept_names[final_concept_names['Word'].apply(filter)]
    neuron_list = artists_df['Neuron'].values
    neuron_list.sort()

    polysemantic_df = final_concept_names[final_concept_names['Synset Count'] < 2]['Neuron'].values
    return [neuron for neuron in range(1, 8192) if neuron not in polysemantic_df]


    # Permutation of 8192
    # import random
    # random.seed(42)
    # neuron_list = random.sample(range(1, 8192), len(neuron_list))
    # remaining_neurons = [neuron for neuron in range(1, 8192) if neuron not in neuron_list]
    # return remaining_neurons
    return neuron_list


def filter_most_activated():
    wordnet = pd.read_csv('wordnet_depths.csv')

    chkpts_dir = 'wordnet_chkpts/'
    final_chpkt = os.path.join(chkpts_dir, 'concept_names_0.csv')
    final_concept_names = pd.read_csv(final_chpkt, names=['Neuron', 'Word', 'Similarity'])

    # Join the final concept names with wordnet depths
    final_concept_names = final_concept_names.merge(wordnet, on='Word', how='inner')

    # Open artist/data/cifar10_test_mean_std.csv
    cifar10_test_mean_std = pd.read_csv('artists_data/cifar10_test_mean_std.csv')
    final_concept_names = final_concept_names.merge(cifar10_test_mean_std, on='Neuron', how='inner')

    # Sort the cifar10_test_mean_std based on the 'mean' column
    final_concept_names = final_concept_names.sort_values(by='std', ascending=False)
    most_activated = final_concept_names.head(128)['Neuron'].values
    return [neuron for neuron in range(1, 8192) if neuron not in most_activated]


def main(args):
    num_input_nodes = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]]
    num_concepts = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]
                                                   ] * args.expansion_factor
    num_classes = args.probe_nclasses

    probe_dir = os.path.join(args.probe_cs_save_dir, args.probe_config_name)
    checkpoint_save_path = osp.join(probe_dir, "on_concepts_ckpts")

    train_val_data = torch.load(
        osp.join(args.probe_cs_save_dir, "train_val", "all_concepts.pth"))
    test_data = torch.load(
        osp.join(args.probe_cs_save_dir, "val", "all_concepts.pth"))
    print(f"Getting {args.probe_dataset} concepts from: {args.probe_cs_save_dir}")

    train_val_labels = torch.load(
        osp.join(args.probe_labels_dir["img"], "all_labels_train_val.pth"))
    test_labels = torch.load(
        osp.join(args.probe_labels_dir["img"], "all_labels_val.pth"))
    
    print('train_val_shape', train_val_data.shape)
    print('test_data', test_data.shape)

    print(set(train_val_labels))
    exit()

    model = torch.nn.Linear(
    num_concepts, num_classes, bias=False)
    model = model.train().to(args.device)

    # Load model checkpoint
    model_path = osp.join(checkpoint_save_path, f"on_concepts_final_{args.probe_config_name}.pt")
    model.load_state_dict(torch.load(model_path, map_location=args.device)['model'])
    
    weight_matrix = model.weight.detach()
    # weight_matrix.mean(dim=0)
    print("Weight matrix shape:", weight_matrix.shape)

    # neuron_list = filter_artists(use_artists=False)
    neuron_list = filter_most_activated()
    print('neuron list len: ', len(neuron_list))

    # Mask
    artist_train_val_data = copy.deepcopy(train_val_data)
    artist_test_data = copy.deepcopy(test_data)
    artist_train_val_data[:, neuron_list] = 0
    artist_test_data[:, neuron_list] = 0
    artist_train_val_data[:, neuron_list] = 0
    artist_test_data[:, neuron_list] = 0

    # Sanity check
    print('Shape test data', artist_test_data.shape, 'Shape train val data', artist_train_val_data.shape)
    print('Must be 0.0:', artist_test_data[:, neuron_list].sum(), artist_train_val_data[:, neuron_list].sum())

    # Datasets
    artist_train_val_dataset = TensorDataset(artist_train_val_data, train_val_labels)
    artist_test_data = TensorDataset(artist_test_data, test_labels)

    abstract_train_val_loader = torch.utils.data.DataLoader(
        artist_train_val_dataset, batch_size=args.probe_train_bs, shuffle=False)
    abstract_test_loader = torch.utils.data.DataLoader(
        artist_test_data, batch_size=args.probe_train_bs, shuffle=False)

    # Evaluation
    _, _, _, val_acc_top1, val_acc_top5, _, _ = eval_model(
        model, abstract_train_val_loader, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)

    _, _, _, test_acc_top1, test_acc_top5, _, _ = eval_model(
        model, abstract_test_loader, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)

    print(f"Val acc top1: {val_acc_top1}, Val acc top5: {val_acc_top5}")
    print(f"Test acc top1: {test_acc_top1}, Test acc top5: {test_acc_top5}")


parser = arg_parser.get_common_parser()
args = parser.parse_args()
common_init(args)
main(args)

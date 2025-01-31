import os
import os.path as osp
import copy

import torch
from tqdm import tqdm
import torchmetrics
import torchmetrics.classification
from torch.utils.data import TensorDataset
import torch.utils.data

from sparse_autoencoder.sparse_autoencoder.metrics.abstract_metric import MetricLocation, MetricResult
import datetime
import statistics
from dncbm.data_utils.probe_classnames import probe_classes_dict

import wandb


from dncbm import arg_parser
from dncbm.utils import common_init
import torchmetrics
import clip

import fcntl
import csv

def read_concept_names(file_path):
    """
    Reads the concept index-to-name mapping from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing concept names.

    Returns:
        dict: A dictionary mapping concept indices to concept names.
    """
    concept_mapping = {}
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            idx, name = int(row[0]), row[1]
            concept_mapping[idx] = name
    return concept_mapping

def get_top_k_weights(model, num_classes, k):
    """
    Returns the top-k concept indices for each class in the model.
    
    Args:
        model (torch.nn.Module): Linear model with weights.
        num_classes (int): Number of classes in the model.
        k (int): Number of top concepts to return.
    
    Returns:
       top_k_weights_indices: dict: Dictionary mapping class indices to lists of top-k concept indices
    """
    top_k_weights_indices = {}
    with torch.no_grad():
        for class_idx in range(num_classes):
            class_weights = model.weight[class_idx, :]
            top_indices = torch.topk(class_weights, k=k).indices
            top_k_weights_indices[class_idx] = top_indices.tolist()
    return top_k_weights_indices

def display_top_concepts(class_idx, top_indices, concept_mapping, weights):
    """
    Displays the top concepts and their corresponding weights for a specific class.

    Args:
        class_idx (int): The class index.
        top_indices (list): List of concept indices to display.
        concept_mapping (dict): Dictionary mapping concept indices to names.
        weights (torch.Tensor): Weights associated with the concepts.
    """
    print(f"Class {class_idx}:")
    for idx in top_indices:
        concept_name = concept_mapping.get(idx, f"Unknown Concept {idx}")
        concept_weight = weights[idx].item()
        print(f"  Concept {idx}: {concept_name} (Weight: {concept_weight:.4f})")

def zero_out_outside_top_k(top_k_weights_indices, model):
    """
    Zero out all weights that are outside the top-k biggest weights for each class.

    Args:
        top_k_weights_indices (dict): Top k concept indices for each class.
        model (torch.nn.Module): Linear model with weights to be masked.
    """
    with torch.no_grad():
        for class_idx, top_k_indices in top_k_weights_indices.items():
            all_indices = set(range(model.weight.shape[1]))
            zero_out_indices = all_indices - set(top_k_indices)  # Indices to zero out
            model.weight[class_idx, list(zero_out_indices)] = 0
    print("Weights outside the top-k have been zeroed out.")

def zero_out_non_bird_concepts(concept_mapping, model, bird_related_keywords):
    """
    Zero out the weights of non-bird-related concepts.

    Args:
        concept_mapping (dict): Mapping of concept indices to concept names.
        model (torch.nn.Module): Linear model with weights to be masked.
        bird_related_keywords (list): List of keywords to identify bird-related concepts.
    """
    bird_related_indices = set()

    # Convert bird keywords to lowercase for case-insensitive matching
    bird_related_keywords = set(keyword.lower() for keyword in bird_related_keywords)

    # Identify bird-related concept indices (Exact Match Only)
    for idx, name in concept_mapping.items():
        lower_name = name.lower().strip()
        
        # Check if the concept name **exactly** matches a keyword in the list
        if lower_name in bird_related_keywords:
            bird_related_indices.add(idx)

    print("\nBird-related concepts identified:")
    for idx in bird_related_indices:
        print(f"  Concept {idx}: {concept_mapping[idx]}")

    # Zero out weights of non-bird concepts
    with torch.no_grad():
        all_indices = set(range(model.weight.shape[1]))
        non_bird_indices = all_indices - bird_related_indices  # Indices not related to birds
        model.weight[:, list(non_bird_indices)] = 0
    print("Weights of non-bird-related concepts have been zeroed out.")

def zero_out_bird_related_concepts(concept_mapping, model, bird_related_keywords):
    """
    Zero out all weights of bird-related concepts.

    Args:
        concept_mapping (dict): Mapping of concept indices to concept names.
        model (torch.nn.Module): Linear model with weights to be masked.
        bird_related_keywords (list): List of keywords to identify bird-related concepts.
    """
    bird_related_indices = set()

    # Convert bird keywords to lowercase for exact, case-insensitive matching
    bird_related_keywords = set(keyword.lower() for keyword in bird_related_keywords)

    # Identify bird-related concept indices (Exact Match Only)
    for idx, name in concept_mapping.items():
        lower_name = name.lower().strip()

        # Check if the concept name **exactly** matches a keyword in the list
        if lower_name in bird_related_keywords:
            bird_related_indices.add(idx)

    print("\nBird-related concepts to be removed:")
    for idx in bird_related_indices:
        print(f"  Concept {idx}: {concept_mapping[idx]}")

    # Zero out weights of bird-related concepts
    with torch.no_grad():
        model.weight[:, list(bird_related_indices)] = 0
    print("Weights of bird-related concepts have been zeroed out.")

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


def main(args):

    if args.probe_on_features:
        wandb_project_name_prefix = f"Probe_training_on_features"
        probe_dir = os.path.join(
            args.probe_features_save_dir, args.probe_config_name)
        checkpoint_save_path = osp.join(
            probe_dir, "on_features_ckpts")
    else:
        wandb_project_name_prefix = f"Probe_training_on_concepts"
        probe_dir = os.path.join(
            args.probe_cs_save_dir, args.probe_config_name)
        checkpoint_save_path = osp.join(
            probe_dir, "on_concepts_ckpts")

    if args.use_wandb:
        wandb_project_name = f"{wandb_project_name_prefix}_{args.sae_dataset}_{args.img_enc_name_for_saving}_{args.hook_points[0]}_{args.probe_dataset}_{datetime.datetime.now().strftime('%Y-%m-%d')}{args.save_suffix}"
        wandb_dir = os.path.join(probe_dir, ".cache/")
        os.makedirs(wandb_dir, exist_ok=True)

        wandb.init(
            project=wandb_project_name,
            entity=args.wandb_entity,
            name=args.config_name+args.probe_config_name,
            dir=wandb_dir,
            config=args,
        )

    os.makedirs(checkpoint_save_path, exist_ok=True)

    num_input_nodes = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]]
    num_concepts = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]
                                                   ] * args.expansion_factor
    num_classes = args.probe_nclasses

    if args.probe_on_features:
        train_data = torch.load(
            osp.join(args.probe_data_dir_activations["img"], "train"))
        train_val_data = torch.load(
            osp.join(args.probe_data_dir_activations["img"], "train_val"))
        test_data = torch.load(
            osp.join(args.probe_data_dir_activations["img"], "val"))
        print(
            f"Getting {args.probe_dataset} features from: {args.probe_features_save_dir}")
    else:
        train_data = torch.load(
            osp.join(args.probe_cs_save_dir, "train", "all_concepts.pth"))
        train_val_data = torch.load(
            osp.join(args.probe_cs_save_dir, "train_val", "all_concepts.pth"))
        test_data = torch.load(
            osp.join(args.probe_cs_save_dir, "val", "all_concepts.pth"))
        print(
            f"Getting {args.probe_dataset} concepts from: {args.probe_cs_save_dir}")
        test_data_ww = "probe/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/waterbirds/train/group_concepts/W@W_concepts.pth"
        test_data_wl = "probe/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/waterbirds/train/group_concepts/W@L_concepts.pth"
        test_data_lw = "probe/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/waterbirds/train/group_concepts/L@W_concepts.pth"
        test_data_ll = "probe/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/waterbirds/train/group_concepts/L@L_concepts.pth"

    train_labels = torch.load(
        osp.join(args.probe_labels_dir["img"], "all_labels_train.pth"))
    train_val_labels = torch.load(
        osp.join(args.probe_labels_dir["img"], "all_labels_train_val.pth"))
    test_labels = torch.load(
        osp.join(args.probe_labels_dir["img"], "all_labels_val.pth"))
    test_labels_ww = "data/activations_img/waterbirds/train/group_features/W@W_all_labels_val.pth"
    test_labels_wl = "data/activations_img/waterbirds/train/group_features/W@L_all_labels_val.pth"
    test_labels_lw = "data/activations_img/waterbirds/train/group_features/L@W_all_labels_val.pth"
    test_labels_ll = "data/activations_img/waterbirds/train/group_features/L@L_all_labels_val.pth"

    train_dataset = TensorDataset(train_data, train_labels)
    train_val_dataset = TensorDataset(train_val_data, train_val_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    test_dataset_ww = TensorDataset(torch.load(test_data_ww), torch.load(test_labels_ww))
    test_dataset_wl = TensorDataset(torch.load(test_data_wl), torch.load(test_labels_wl))
    test_dataset_lw = TensorDataset(torch.load(test_data_lw), torch.load(test_labels_lw))
    test_dataset_ll = TensorDataset(torch.load(test_data_ll), torch.load(test_labels_ll))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.probe_train_bs, shuffle=True)
    train_val_loader = torch.utils.data.DataLoader(
        train_val_dataset, batch_size=args.probe_train_bs, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.probe_train_bs, shuffle=False)
    test_loader_ww = torch.utils.data.DataLoader(test_dataset_ww, batch_size=args.probe_train_bs, shuffle=False)
    test_loader_wl = torch.utils.data.DataLoader(test_dataset_wl, batch_size=args.probe_train_bs, shuffle=False)
    test_loader_lw = torch.utils.data.DataLoader(test_dataset_lw, batch_size=args.probe_train_bs, shuffle=False)
    test_loader_ll = torch.utils.data.DataLoader(test_dataset_ll, batch_size=args.probe_train_bs, shuffle=False)
    
    if args.probe_on_features:
        prefix = 'on_features'
        model_only_birds = torch.nn.Linear(num_input_nodes, num_classes, bias=False)
        model_no_birds = torch.nn.Linear(num_input_nodes, num_classes, bias=False)
        model_normal = torch.nn.Linear(num_input_nodes, num_classes, bias=False)
    else:
        prefix = 'on_concepts'
        model_only_birds = torch.nn.Linear(num_concepts, num_classes, bias=False)
        model_no_birds = torch.nn.Linear(num_concepts, num_classes, bias=False)
        model_normal = torch.nn.Linear(num_concepts, num_classes, bias=False)

    # Print model architecture
    print(model_only_birds)
    model_only_birds = model_only_birds.train().to(args.device)
    model_no_birds = model_no_birds.train().to(args.device)
    model_normal = model_normal.train().to(args.device)

    # Load the saved state dict
    state_dict_path = osp.join(checkpoint_save_path, f"{prefix}_final_{args.probe_config_name}.pt")

    # Load the checkpoint and extract the model's state dict
    checkpoint = torch.load(state_dict_path, map_location=args.device)
    state_dict = checkpoint['model']  # Extract only the model weights
    print ("Accuracy top 1:", checkpoint['test_acc_top1'], "Loss:", checkpoint['test_loss'])

    # Load the state dict into the model
    model_only_birds.load_state_dict(state_dict)
    model_no_birds.load_state_dict(state_dict)
    model_normal.load_state_dict(state_dict)

    # Switch to evaluation mode if needed
    model_only_birds = model_only_birds.eval()
    model_no_birds = model_no_birds.eval()
    model_normal = model_normal.eval()

    # File containing the concept index-to-name mapping
    concept_names_file = os.path.join(args.save_dir["img"], f"concept_names.csv")
    
    # Load concept index-to-name mapping
    concept_mapping = read_concept_names(concept_names_file)
    k = 10
    #! Get top k largest weights (top k concepts) for each class in the dataset
    top_k_weights_indices = get_top_k_weights(model_normal, num_classes, k)

    # Display the top k concepts for each class
    print(f"Top {k} concepts for each class:")
    for class_idx, indices in top_k_weights_indices.items():
        display_top_concepts(class_idx, indices, concept_mapping, model_normal.weight[class_idx])

    # Define bird-related keywords for manual filtering
    bird_related_keywords = [
        "bird", "sparrow", "parrot", "owl", "eagle", "ducks", "ibis", "penguin", "crane", "falcon", "birds", "swan", "peacock",
        "duck", "ravens", "chickens", "penguins", "emu"
    ]
    
    # Zero out weights outside the top k for each class
    zero_out_outside_top_k(top_k_weights_indices, model_only_birds)
    zero_out_outside_top_k(top_k_weights_indices, model_no_birds)

    # Select and mask bird-related concepts
    zero_out_non_bird_concepts(concept_mapping, model_only_birds, bird_related_keywords)
    zero_out_bird_related_concepts(concept_mapping, model_no_birds, bird_related_keywords)

    # Evaluate the model on the test set

    # ONLY BIRDS AVERAGE
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model_only_birds, test_loader, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
    print ("Model only birds on average:")
    print(f"Test loss: {test_loss}, Test classification loss: {test_class_loss}, Test sparsity loss: {test_sparse_loss}, Test top-1 accuracy: {test_acc_top1}, Test top-5 accuracy: {test_acc_top5}")

    # NO BIRDS AVERAGE
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model_no_birds, test_loader, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
    print ("Model no birds on average:")
    print(f"Test loss: {test_loss}, Test classification loss: {test_class_loss}, Test sparsity loss: {test_sparse_loss}, Test top-1 accuracy: {test_acc_top1}, Test top-5 accuracy: {test_acc_top5}")
    
    # NORMAL AVERAGE
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model_normal, test_loader, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
    print ("Model NORMAL on average:")
    print(f"Test loss: {test_loss}, Test classification loss: {test_class_loss}, Test sparsity loss: {test_sparse_loss}, Test top-1 accuracy: {test_acc_top1}, Test top-5 accuracy: {test_acc_top5}")

    ######################################

    # NO BiRDS WW
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model_no_birds, test_loader_ww, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
    print ("Model no birds WW:")
    print(f"Test loss: {test_loss}, Test classification loss: {test_class_loss}, Test sparsity loss: {test_sparse_loss}, Test top-1 accuracy: {test_acc_top1}, Test top-5 accuracy: {test_acc_top5}")
    
    # NO BiRDS WL
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model_no_birds, test_loader_wl, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
    print ("Model no birds WL:")
    print(f"Test loss: {test_loss}, Test classification loss: {test_class_loss}, Test sparsity loss: {test_sparse_loss}, Test top-1 accuracy: {test_acc_top1}, Test top-5 accuracy: {test_acc_top5}")
    
    # NO BiRDS LW
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model_no_birds, test_loader_lw, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
    print ("Model no birds LW:")
    print(f"Test loss: {test_loss}, Test classification loss: {test_class_loss}, Test sparsity loss: {test_sparse_loss}, Test top-1 accuracy: {test_acc_top1}, Test top-5 accuracy: {test_acc_top5}")
    
    # NO BiRDS LL
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model_no_birds, test_loader_ll, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
    print ("Model no birds LL:")
    print(f"Test loss: {test_loss}, Test classification loss: {test_class_loss}, Test sparsity loss: {test_sparse_loss}, Test top-1 accuracy: {test_acc_top1}, Test top-5 accuracy: {test_acc_top5}")
    
    ######################################

    # ONLY BIRDS WW
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model_only_birds, test_loader_ww, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
    print ("Model only birds WW:")
    print(f"Test loss: {test_loss}, Test classification loss: {test_class_loss}, Test sparsity loss: {test_sparse_loss}, Test top-1 accuracy: {test_acc_top1}, Test top-5 accuracy: {test_acc_top5}")
    
    # ONLY BIRDS WL
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model_only_birds, test_loader_wl, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
    print ("Model only birds WL:")
    print(f"Test loss: {test_loss}, Test classification loss: {test_class_loss}, Test sparsity loss: {test_sparse_loss}, Test top-1 accuracy: {test_acc_top1}, Test top-5 accuracy: {test_acc_top5}")
    
    # ONLY BIRDS LW
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model_only_birds, test_loader_lw, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
    print ("Model only birds LW:")
    print(f"Test loss: {test_loss}, Test classification loss: {test_class_loss}, Test sparsity loss: {test_sparse_loss}, Test top-1 accuracy: {test_acc_top1}, Test top-5 accuracy: {test_acc_top5}")
    
    # ONLY BIRDS LL
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model_only_birds, test_loader_ll, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
    print ("Model only birds LL:")
    print(f"Test loss: {test_loss}, Test classification loss: {test_class_loss}, Test sparsity loss: {test_sparse_loss}, Test top-1 accuracy: {test_acc_top1}, Test top-5 accuracy: {test_acc_top5}")
    
    ######################################

    # NORMAL WW
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model_normal, test_loader_ww, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
    print ("Model NORMAL WW:")
    print(f"Test loss: {test_loss}, Test classification loss: {test_class_loss}, Test sparsity loss: {test_sparse_loss}, Test top-1 accuracy: {test_acc_top1}, Test top-5 accuracy: {test_acc_top5}")
    
    # NORMAL WL
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model_normal, test_loader_wl, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
    print ("Model NORMAL WL:")
    print(f"Test loss: {test_loss}, Test classification loss: {test_class_loss}, Test sparsity loss: {test_sparse_loss}, Test top-1 accuracy: {test_acc_top1}, Test top-5 accuracy: {test_acc_top5}")
    
    # NORMAL LW
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model_normal, test_loader_lw, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
    print ("Model NORMAL LW:")
    print(f"Test loss: {test_loss}, Test classification loss: {test_class_loss}, Test sparsity loss: {test_sparse_loss}, Test top-1 accuracy: {test_acc_top1}, Test top-5 accuracy: {test_acc_top5}")
    
    # NORMAL LL
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model_normal, test_loader_ll, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
    print ("Model NORMAL LL:")
    print(f"Test loss: {test_loss}, Test classification loss: {test_class_loss}, Test sparsity loss: {test_sparse_loss}, Test top-1 accuracy: {test_acc_top1}, Test top-5 accuracy: {test_acc_top5}")
    
parser = arg_parser.get_common_parser()
args = parser.parse_args()
common_init(args)
main(args)

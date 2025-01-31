import os
import os.path as osp
import copy

import torch
from tqdm import tqdm
import torchmetrics
import torchmetrics.classification
from torch.utils.data import TensorDataset
import torch.utils.data

import datetime
import statistics
from dncbm.data_utils.probe_classnames import probe_classes_dict
import dncbm.config as config

from dncbm import arg_parser
from dncbm.utils import common_init, get_img_model
import torchmetrics
from get_closest_concepts import get_concepts

from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import csv
import numpy as np
import clip

from sentence_transformers import SentenceTransformer

from abc import ABC, abstractmethod

class SimilarityModel(ABC):
    def __init__(self):
        self.suggested_threshold = 0.5

    @abstractmethod
    def get_embedding(self, text):
        pass
    
    # Cosine similarity
    def get_similarity(self, embedding1, embedding2):
        if embedding1 is None or embedding2 is None:
            raise ValueError(f"Could not compute embeddings")
        embedding1 = F.normalize(embedding1, p=2, dim=-1)
        embedding2 = F.normalize(embedding2, p=2, dim=-1)
        return torch.cosine_similarity(embedding1, embedding2).item()

class BertSimilarityModel(SimilarityModel):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_model.eval()
        self.suggested_threshold = 0.75

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        # Use pooler_output for better sentence-level embeddings
        return outputs.pooler_output.squeeze()
    
class GloVeSimilarityModel(SimilarityModel):
    def __init__(self):
        super().__init__()
        self.suggested_threshold = 0.3
        glove_path = "data/glove.6B.300d.txt"
        if not os.path.exists(glove_path):
            raise FileNotFoundError(f"GloVe file not found at {glove_path}")

        self.embeddings = {}
        all_embeddings = []
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype=np.float32)
                self.embeddings[word] = vector
                all_embeddings.append(vector)
        self.embeddings['<unk>'] = np.mean(all_embeddings, axis=0)  # Mean of all vectors

    def get_embedding(self, text):
        embeddings = [self.embeddings[word] for word in text.split() if word in self.embeddings]
        if embeddings:
            return torch.tensor(np.mean(embeddings, axis=0))
        else:
            return torch.tensor(self.embeddings['<unk>'])

class ClipSimilarityModel(SimilarityModel):
    def __init__(self):
        super().__init__()
        self.clip_encoder = get_img_model(args)[0]
        self.suggested_threshold = 0.7
    
    def get_embedding(self, text):
        try:
            # Tokenize the text prompts (convert them to a tensor of token IDs)
            text_tokens = clip.tokenize(text).to(args.device)

            # Encode the text using the CLIP text encoder
            with torch.no_grad():
                text_features = self.clip_encoder.encode_text(text_tokens)

            # Normalize the embeddings to unit vectors
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features.squeeze(0)
        except Exception as e:
            print(f"Error: {e}")

class SentenceTransformerSimilarityModel(SimilarityModel):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.suggested_threshold = 0.4

    def get_embedding(self, text):
        return torch.tensor(self.model.encode(text, convert_to_numpy=True))

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

def display_remaining_concepts(close_concepts_in_embedding_space):
    """
    Displays the top concepts and their corresponding weights for a specific class.

    Args:
        class_idx (int): The class index.
        top_indices (list): List of concept indices to display.
        concept_mapping (dict): Dictionary mapping concept indices to names.
        weights (torch.Tensor): Weights associated with the concepts.
    """
    for class_name, concepts in close_concepts_in_embedding_space.items():
        print(f"\nClass: {class_name}")
        for concept_info in concepts:
            print(f"   Concept_idx: {concept_info['concept_idx']}, Concept: {concept_info['concept']}, Similarity: {concept_info['similarity']:.4f}")

def get_top_k_biggest_weights (args, model, k=10):
    """
    Get the top-k biggest weights for each class in the model.

    Args:
        args (argparse.Namespace): Command-line arguments.
        model (torch.nn.Module): Linear model with weights.
        k (int): Number of top weights to display.
    """
    class_names = probe_classes_dict[args.probe_dataset]
    top_k_weights_indices = {}
    with torch.no_grad():
        for class_idx in range(args.probe_nclasses):
            class_weights = model.weight[class_idx, :]
            top_indices = torch.topk(class_weights, k=k).indices
            top_k_weights_indices[class_names[class_idx]] = top_indices.tolist()
    return top_k_weights_indices

def zero_out_unrelated_concepts(closest_concepts, model):
    """
    Zero out all weights in the model that are not in the indices present in closest_concepts,
    ensuring at least one weight (the largest) is preserved.

    Args:
        closest_concepts (dict): A dictionary where keys are class names and values are lists of dictionaries.
                                 Each dictionary contains "concept_idx" (index), "concept_name", and "similarity".
        model (torch.nn.Module): Linear model with weights to be masked.
    """
    with torch.no_grad():
        for class_idx, (class_name, close_concepts) in enumerate(closest_concepts.items()):
            # Get the indices of the concepts that are related to the class
            related_indices = [concept_info["concept_idx"] for concept_info in close_concepts]

            # If no related indices are found, preserve the largest weight
            if not related_indices:
                max_weight_idx = torch.argmax(torch.abs(model.weight[class_idx])).item()
                related_indices = [max_weight_idx]

            # Zero out the weights for unrelated concepts
            for idx in range(model.weight.shape[1]):
                if idx not in related_indices:
                    model.weight[class_idx, idx] = 0

            # Ensure at least one weight is preserved (largest if none of the related_indices are preserved)
            if all(model.weight[class_idx, idx] == 0 for idx in related_indices):
                max_weight_idx = torch.argmax(torch.abs(model.weight[class_idx])).item()
                model.weight[class_idx, max_weight_idx] = model.weight[class_idx, max_weight_idx].detach()

    print("Weights for unrelated concepts have been zeroed out, ensuring at least one weight is preserved.")
    return model

def get_close_concepts_in_embeddings_space(similarity_model, top_k_concepts, concept_mapping):
    """
    Efficiently retrieve concepts that are close to the class names in the BERT embedding space.

    Args:
        similarity_model: A model providing embeddings and similarity computations.
        top_k_concepts (dict): Dictionary mapping class names to lists of closest concept indices.
        concept_mapping (dict): Dictionary mapping concept indices to concept names.

    Returns:
        dict: A dictionary mapping each class name to its closest concepts in the BERT space.
    """

    # Extract class names and retrieve all embeddings at once
    class_names = list(top_k_concepts.keys())
    class_embeddings = torch.stack([similarity_model.get_embedding(name) for name in class_names])  # Shape: [num_classes, embedding_dim]

    # Flatten the list of all unique concept indices and retrieve embeddings at once
    unique_concept_indices = list(set(idx for indices in top_k_concepts.values() for idx in indices))
    concept_names = [concept_mapping[idx] for idx in unique_concept_indices]
    concept_embeddings = torch.stack([similarity_model.get_embedding(name) for name in concept_names])  # Shape: [num_concepts, embedding_dim]

    # Normalize embeddings for cosine similarity calculation
    class_embeddings = torch.nn.functional.normalize(class_embeddings, p=2, dim=1)  # [num_classes, embedding_dim]
    concept_embeddings = torch.nn.functional.normalize(concept_embeddings, p=2, dim=1)  # [num_concepts, embedding_dim]

    # Compute cosine similarity matrix in one step (efficient batch operation)
    similarity_matrix = torch.matmul(class_embeddings, concept_embeddings.T)  # Shape: [num_classes, num_concepts]

    # Map concept indices back to their original names
    concept_index_to_name = {idx: name for idx, name in zip(unique_concept_indices, concept_names)}

    # Filter results based on the threshold
    close_concepts_in_BERT_space = {
        class_name: [
            {"concept_idx": unique_concept_indices[j], "concept": concept_index_to_name[unique_concept_indices[j]], "similarity": similarity_matrix[i, j].item()}
            for j in range(len(unique_concept_indices)) if similarity_matrix[i, j] > similarity_model.suggested_threshold
        ]
        for i, class_name in enumerate(class_names)
    }

    return close_concepts_in_BERT_space

def get_similarity_model(args):
    if args.probe_similarity_model == "bert":
        return BertSimilarityModel()
    elif args.probe_similarity_model == "glove":
        return GloVeSimilarityModel()
    elif args.probe_similarity_model == "clip":
        return ClipSimilarityModel()
    elif args.probe_similarity_model == "sentence_transformer":
        return SentenceTransformerSimilarityModel()
    elif args.probe_similarity_model == "wordnet":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown similarity model: {args.probe_similarity_model}")

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

    train_labels = torch.load(
        osp.join(args.probe_labels_dir["img"], "all_labels_train.pth"))
    train_val_labels = torch.load(
        osp.join(args.probe_labels_dir["img"], "all_labels_train_val.pth"))
    test_labels = torch.load(
        osp.join(args.probe_labels_dir["img"], "all_labels_val.pth"))

    train_dataset = TensorDataset(train_data, train_labels)
    train_val_dataset = TensorDataset(train_val_data, train_val_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.probe_train_bs, shuffle=True)
    train_val_loader = torch.utils.data.DataLoader(
        train_val_dataset, batch_size=args.probe_train_bs, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.probe_train_bs, shuffle=False)

    if args.probe_on_features:
        prefix = 'on_features'
        model = torch.nn.Linear(num_input_nodes, num_classes, bias=False)
    else:
        prefix = 'on_concepts'
        model = torch.nn.Linear(num_concepts, num_classes, bias=False)

    print (f"Loading model {prefix} from: {checkpoint_save_path}")

    model = model.train().to(args.device)
    state_dict_path = osp.join(checkpoint_save_path, f"{prefix}_final_{args.probe_config_name}.pt")
    
    # Load concept names from trained SAE
    concept_names_file = os.path.join(args.save_dir["img"], f"concept_names.csv")
    concept_mapping = read_concept_names(concept_names_file)    

    # Load the checkpoint and extract the model's state dict
    checkpoint = torch.load(state_dict_path, map_location=args.device)
    state_dict = checkpoint['model']  # Extract only the model weights
    print ("Accuracy top 1:", checkpoint['test_acc_top1'], "Accuracy top 5:", checkpoint['test_acc_top5'], "Loss:", checkpoint['test_loss'])
    # Load the state dict into the model
    model.load_state_dict(state_dict)

    # Switch to evaluation mode if needed
    model = model.eval()

    similarity_model = get_similarity_model(args)

    top_k_concepts = get_top_k_biggest_weights(args, model, k=2000)

    close_concepts_in_embedding_space = get_close_concepts_in_embeddings_space(similarity_model, top_k_concepts, concept_mapping)

    # Print how many concepts have been selected for each class
    print(f"Number of concepts selected for each class:")
    for class_name, close_concepts in close_concepts_in_embedding_space.items():
        print(f"Class: {class_name}, Number of concepts: {len(close_concepts)}")

    # Print the resulting dictionary
    display_remaining_concepts(close_concepts_in_embedding_space)

    # Zero out unrelated concepts
    model = zero_out_unrelated_concepts(close_concepts_in_embedding_space, model)

    # Evaluate the model on the test set
    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model, test_loader, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)

    print (f"test_loss: {test_loss}, test_class_loss: {test_class_loss}, test_sparse_loss: {test_sparse_loss}, test_acc_top1: {test_acc_top1}, test_acc_top5: {test_acc_top5}")

parser = arg_parser.get_common_parser()
args = parser.parse_args()
common_init(args)
main(args)

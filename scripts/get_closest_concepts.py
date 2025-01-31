from dncbm import method_utils
from dncbm import arg_parser
from dncbm import utils
from dncbm.data_utils import probe_classnames
from dncbm.utils import common_init
import os
import torch
import clip

def load_word_embeddings(args, word_list):
    """
    Load embeddings for words in the word list.

    Args:
        word_list (list): List of words.
        all_embeddings (torch.Tensor): Concept embedding tensor.
        vocab_txt_all (list): Vocabulary corresponding to the embeddings.

    Returns:
        torch.Tensor: Embeddings of the given words.
    """
    word_embeddings = []
    clip_encoder = utils.get_img_model(args)[0]
    for sentence in word_list:
        try:
            # Tokenize the text prompts (convert them to a tensor of token IDs)
            text_tokens = clip.tokenize(sentence).to(args.device)

            # Encode the text using the CLIP text encoder
            with torch.no_grad():
                text_features = clip_encoder.encode_text(text_tokens)

            # Normalize the embeddings to unit vectors
            text_features /= text_features.norm(dim=-1, keepdim=True)
            word_embeddings.append(text_features)
        except Exception as e:
            print(f"Error: {e}")

    return torch.stack(word_embeddings)

def load_concept_names(concept_csv_path):
    """
    Load concept names from the concept_names.csv file.

    Args:
        concept_csv_path (str): Path to the concept_names.csv file.

    Returns:
        list: List of concept names.
    """
    concept_names = []
    with open(concept_csv_path, "r") as f:
        for line in f:
            values = line.strip().split(",")  # Split the line by commas
            if len(values) == 3:  # Ensure there are exactly three values
                name = values[1]  # Get the middle value
                concept_names.append(name)
            else:
                print(f"Skipping line due to unexpected format: {line.strip()}")
    return concept_names

def get_concepts (args, k=5):
    # Initialize arguments
    parser = arg_parser.get_common_parser()
    args = parser.parse_args()
    utils.common_init(args)

    # Paths to embeddings and vocabulary
    embeddings_path = os.path.join(
        args.vocab_dir, f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth")
    vocab_txt_path = os.path.join(args.vocab_dir, f"clipdissect_20k.txt")

    # Initialize method object
    method_obj = method_utils.get_method("ours", args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path)

    # Load concept names
    concept_csv_path = os.path.join(args.save_dir["img"], "concept_names.csv")
    concept_names = load_concept_names(concept_csv_path)

    # Load word list
    class_names = probe_classnames.probe_classes_dict[args.probe_dataset]

    # Load word embeddings
    class_names_embeddings = load_word_embeddings(args, class_names).to(args.device)

    # Get top-k closest concepts
    closest_concepts = {}
    dic_vec = method_obj.all_dic_vec.to(args.device)  # Ensure dic_vec is on the correct device
    dic_vec = dic_vec.to(torch.float32)  # Convert dic_vec to float32
    dic_vec /= dic_vec.norm(dim=0, keepdim=True)  # Normalize concept embeddings

    for class_name, class_name_embedding in zip(class_names, class_names_embeddings):
        closest_concepts[class_name] = []
        class_name_embedding = class_name_embedding.to(args.device)  # Ensure class_name_embedding is on the correct device
        class_name_embedding = class_name_embedding.to(torch.float32)  # Convert to float32
        class_name_embedding /= class_name_embedding.norm()  # Normalize embedding

        similarities = torch.matmul(dic_vec.T, class_name_embedding.squeeze(0))  # Compute cosine similarities
        top_k = torch.topk(similarities, k=k, dim=0)  # Get top-k similarities and indices

        # Convert generator to a list and append to closest_concepts
        closest_concepts[class_name] = [(idx, concept_names[idx], similarities[idx].item()) for idx in top_k.indices]

    # Print top-k closest concepts for each class name, max 100
    for class_name, close_concepts in closest_concepts.items():
        print(f"\nClass Name: {class_name}")
        print("Closest concepts:")
        for rank, (idx, concept, similarity) in enumerate(close_concepts, start=1):
            if rank > 100:
                break
            print(f"  {rank}. Concept_idx: {idx}, Concept: {concept}, Similarity: {similarity:.4f}")

    return closest_concepts

if __name__ == "__main__":
    parser = arg_parser.get_common_parser()
    args = parser.parse_args()
    common_init(args)
    get_concepts(args, k=10)
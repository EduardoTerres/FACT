import os
import numpy as np
from dncbm import method_utils, arg_parser, utils
from dncbm import method_utils
from dncbm import arg_parser
from dncbm import utils
from dncbm import config

def load_similarity_matrix_and_vocab(args):
    """
    Loads the concept-name similarity matrix and the vocabulary.
    """
    embeddings_path = os.path.join(args.vocab_dir, f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth")
    vocab_txt_path = os.path.join(args.vocab_dir, f"clipdissect_20k.txt")

    method_obj = method_utils.get_method(
        "ours", args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path, use_fixed_sae=True
    )
    similarity_matrix = method_obj.get_concept_name_similarity_matrix()[0]
    all_concept_names = method_obj.vocab_txt_all[0]
    return similarity_matrix, all_concept_names

def sample_nodes(similarity_matrix):
    """
    Samples nodes from specified bins based on similarity scores.
    """
    # Compute mean similarity for each node (columns in similarity matrix)
    top_concepts_values = similarity_matrix.max(axis=0)
    top_concept_idxs = similarity_matrix.argmax(axis=0)


    # Sort nodes by similarity
    sorted_indices = np.argsort(top_concepts_values)

    # Define bins
    high_bin = top_concept_idxs[sorted_indices][:2000]
    low_bin = top_concept_idxs[sorted_indices][:-2000]
    intermediate_bin = top_concept_idxs[sorted_indices][2000:-2000]

    # Sample nodes uniformly at random from each bin
    sampled_high = np.random.choice(high_bin, size=5, replace=False)
    sampled_intermediate = np.random.choice(intermediate_bin, size=5, replace=False)
    sampled_low = np.random.choice(low_bin, size=5, replace=False)

    return sampled_high, sampled_intermediate, sampled_low

def user_study(args):
    """
    Main function to conduct the user study.
    """
    utils.common_init(args)

    # Load similarity matrix and concept names
    similarity_matrix, all_concept_names = load_similarity_matrix_and_vocab(args)

    # Define bins and samples per bin
    '''
    if args.model_type == "SAE":
        num_nodes = 8192
        bins = {"high": 2000, "low": 2000}
        samples_per_bin = {"high": 5, "intermediate": 5, "low": 5}
    elif args.model_type == "CLIP":
        num_nodes = 1024
        bins = {"high": 250, "low": 250}
        samples_per_bin = {"high": 3, "intermediate": 3, "low": 3}
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    '''
    # Sample nodes
    sampled_high, sampled_intermediate, sampled_low = sample_nodes(similarity_matrix)
    names_dic =  {
        "high": [all_concept_names[sampled_high[i]] for i in range(5)],
        "intermediate": [all_concept_names[sampled_intermediate[i]] for i in range(5)],
        "low": [all_concept_names[sampled_low[i]] for i in range(5)],
    }
    print(names_dic)
    return names_dic

if __name__ == "__main__":
    parser = arg_parser.get_common_parser()
    parser.add_argument("--model_type", type=str, default="SAE", choices=["SAE", "CLIP"], help="Type of model (SAE or CLIP)")
    args = parser.parse_args()

    # Conduct user study
    user_study(args)

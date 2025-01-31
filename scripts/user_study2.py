import os
import numpy as np
from dncbm import method_utils, arg_parser, utils
from dncbm import method_utils
from dncbm import arg_parser
from dncbm import utils
from dncbm import config
import torch


parser = arg_parser.get_common_parser()
args = parser.parse_args()
utils.common_init(args)
embeddings_path = os.path.join(args.vocab_dir, f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth")
vocab_txt_path = os.path.join(args.vocab_dir, f"clipdissect_20k.txt")
method_obj = method_utils.get_method("ours", args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path)
concept_name_similarity_matrix = method_obj.get_concept_name_similarity_matrix()[0]
print("matrix: ", concept_name_similarity_matrix.shape)
# 20k vocabulary
all_concept_names = method_obj.vocab_txt_all[0]
# 8k indexes indicating the number of the vocabulary word that has maximum score
vocab_idxs = torch.argmax(concept_name_similarity_matrix, dim=0).numpy()
print("matrix: ", concept_name_similarity_matrix.shape)
# 8k top scores 
top_concepts_values = concept_name_similarity_matrix.max(dim=0).values
top_concepts_values= np.array(top_concepts_values)
print('top_concept_idxs: ',vocab_idxs.shape, vocab_idxs.max())
print('top_concept_values: ', top_concepts_values.shape)
# Sort nodes by similarity
# 8k indices for sorting the concept values
sorted_indices = np.argsort(top_concepts_values)
print('sorted_indices: ',sorted_indices.shape, sorted_indices.max())
# top_conepts_idxs sorted
#sorted_concept = vocab_idxs[sorted_indices]
#high_bin = vocab_idxs[:2000]
#low_bin = vocab_idxs[:-2000]
#intermediate_bin = vocab_idxs[2000:-2000]
high_bin = sorted_indices[:2000]
low_bin = sorted_indices[:-2000]
intermediate_bin = sorted_indices[2000:-2000]
print('high: ', high_bin.shape)
print('top 20 aligned concepts: ',high_bin[:20])
for el in high_bin[:20]:
    print(all_concept_names[vocab_idxs[el]])
for x in range(20):
    # Sample nodes/concepts uniformly at random from each bin
    sampled_high = np.random.choice(high_bin, size=5, replace=False)
    sampled_intermediate = np.random.choice(intermediate_bin, size=5, replace=False)
    sampled_low = np.random.choice(low_bin, size=5, replace=False)
    print('samples: ',sampled_high, sampled_intermediate, sampled_low)
    names_dic =  {
            "high": [all_concept_names[vocab_idxs[sampled_high[i]]] for i in range(5)],
            "intermediate": [all_concept_names[vocab_idxs[sampled_intermediate[i]]] for i in range(5)],
            "low": [all_concept_names[vocab_idxs[sampled_low[i]] ]for i in range(5)],
        }
    scores_dic =  {
            "high": [top_concepts_values[sampled_high[i]] for i in range(5)],
            "intermediate": [top_concepts_values[sampled_intermediate[i]] for i in range(5)],
            "low": [top_concepts_values[sampled_low[i]]for i in range(5)],
        }
    
    
    print(names_dic)
    print(scores_dic)

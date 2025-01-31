import os  
import clip
import time
import torchvision
import torch
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from dncbm.utils import common_init, get_probe_dataset
from dncbm import arg_parser, method_utils, config
from dncbm.visualization.canvas import Canvas

# Start time for execution
start_time = time.time()

# Argument parsing and common initialization
parser = arg_parser.get_common_parser()
args = parser.parse_args()
common_init(args, disable_make_dirs=True)
topk = 4  # Top 4 images to display for each concept

# Paths to embeddings and vocab files
embeddings_path = osp.join(args.vocab_dir, f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth")
vocab_txt_path  = osp.join(args.vocab_dir, "clipdissect_20k.txt")

# Load method object for processing
#method_obj = method_utils.get_method(args.method_name, args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path, device='cpu')
#vocab_specific_embedding = method_obj.all_embeddings[0]
name_map = {"full-imagenet": "ImageNet", "cifar10": "CIFAR10", "cifar100": "CIFAR100", "places365": "Places365", "cc3m": "CC3M"}

# Visualization setup
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

dataset_list = ['full-imagenet', 'places365']  # Only using ImageNet and Places365 datasets

bpad = 50
spad = 5
imsize = 100
left = 40
top = 25
rows = len(dataset_list)  # Number of rows per concept
subcols = 4  # Number of images per concept
cols = 5  # # num of concepts 
fontsize = 30

# Directory paths for saving results
data_dir = osp.join(config.analysis_dir, 'user_study', args.img_enc_name_for_saving, 'data')
vis_dir = osp.join(config.analysis_dir, 'user_study', args.img_enc_name_for_saving, 'vis')



# Load top image IDs dictionary (which contains topk activations for each dataset)
dict_fname = f"top{topk}_img_ids_dict_{args.probe_split}.pt"
top_img_ids_dict = torch.load(osp.join(data_dir, dict_fname))

# Preprocess setup using CLIP model
_, preprocess = clip.load(args.img_enc_name[5:], device='cpu')
un_normalize = torchvision.transforms.Normalize(
    (-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711),
    (1/0.26862954, 1/0.26130258, 1/0.27577711)
)

# Load real datasets for ImageNet and Places365
real_dataset_list = []
for dataset_idx, dataset_name in enumerate(dataset_list):
    args.probe_dataset = dataset_name
    common_init(args, disable_make_dirs=True)
    dataset = get_probe_dataset(args.probe_dataset, args.probe_split, args.probe_dataset_root_dir, preprocess_fn=preprocess)
    real_dataset_list.append(dataset)

print(f"Will save plots at {vis_dir}")
print(f"will load data from {data_dir}")

# Directory setup for saving visualizations
os.makedirs(vis_dir, exist_ok=True)

def draw_in_canvas(names, fname, cols=2):
    """
    Draws and saves the images in a 4x2 grid on the canvas for each concept.
    """
    Canvas.DEFAULTS["use_latex"] = True
    Canvas.DEFAULTS["save_path"] = vis_dir
    canvas = Canvas.make(width=cols * ((subcols * (imsize + spad) - spad) + bpad) - bpad, 
                         height=rows * (imsize + spad) - spad, 
                         left=left, top=top)
    major_cols = canvas.horizontal_split(cols, bpad)  # Split into major columns for ImageNet and Places365

    for idx, col in enumerate(major_cols):
        col.set_title(names[idx][1], y_off=0.1, fontsize=fontsize)
        col.set_title(f"Index {names[idx][0]}", y_off=0.015, fontweight="normal")
       
    with canvas.zoom((left, canvas.extent[1], top, canvas.extent[-1])):
        for grid_idx, subgrid in enumerate([col.even_grid(rows, subcols, spad) for col in major_cols]):  # Different concepts
            for row_idx, row in enumerate(subgrid):
                start_idx = names[grid_idx][0] * topk
                end_idx = start_idx + topk
                img_ids = top_img_ids_dict[dataset_list[row_idx]][start_idx:end_idx]
              
                for col_idx, cell in enumerate(row):
                    img_id = img_ids[col_idx]
                    if img_id != -1:
                        img = real_dataset_list[row_idx][img_id][0].unsqueeze(0)
                        img = un_normalize(img)[0].detach().cpu().permute(1, 2, 0).numpy()
                        cell.imshow(img.clip(0, 1))
                        
                        if not grid_idx and not col_idx:
                            cell.set_ylabel(name_map[dataset_list[row_idx]], rotation=0, fontsize=fontsize)       
    canvas.save(fname)

'''
concept_names = [['reebok', 'projectors', 'hogtied', 'taipei', 'godsmack'],  ['bouquet', 'participants', 'courtyard', 'longitude', 'landscaping'],  ['eminem', 'details', 'woodstock', 'magnets', 'judaism']]
# Concept samples: each group contains concept indices
concept_samples = [
    [10649, 7390, 10270, 13165, 11438],  # Group 1
    [9878, 2598, 10101, 9955, 10311],    # Group 2
    [5368, 226, 15196, 10332, 13831]     # Group 3
]
group_category = ['high', 'intermediate', 'low']

# Process each concept in the given samples and visualize
for group_idx, group in enumerate(concept_samples):
    group_concepts = concept_names[group_idx]
    names = tuple(zip(group, group_concepts))
    
    # Create file name and draw the images for the concept
    fname = f"fig_group{group_category[group_idx]}.pdf"
    draw_in_canvas(names, fname, cols=cols)

print(f"Time taken: {np.round(time.time() - start_time, 3)}")
'''

concept_names = torch.load(osp.join(data_dir, "concept_names.pt"))
sim_values = torch.load(osp.join(data_dir, "sim_vals.pt"))

sorted_values, sorted_indices = torch.sort(sim_values, descending=True)
sorted_concept_names = concept_names[sorted_indices]

start_idx= 0
end_idx = 15
for index in range(start_idx, end_idx, cols):
    print(f"Total plottings to be done: {index}/{end_idx}")
    node_ids_tmp = sorted_indices[index: index+cols]
    concept_names_tmp = sorted_concept_names[index: index+cols]

    names = tuple(zip(node_ids_tmp, concept_names_tmp))
    node_ids_tmp = node_ids_tmp.tolist()
    ugly_fname = f"fig{index:04d}_node_ids{node_ids_tmp}.pdf"
    fname = ugly_fname.replace('[', '_').replace(']', '_').replace(',', '').replace(' ', '_')
    draw_in_canvas(names, fname, cols=cols) 

print(f"Time taken: {np.round(time.time()-start_time,3)}")
import os
import pandas as pd
from pathlib import Path

def create_symlink_folders_with_classes(csv_file, output_dir):
    # Load the CSV
    data = pd.read_csv(csv_file, delimiter=';')
    
    # Create base directories for train, val, and test
    for split in ['train', 'val', 'test']:
        split_dir = Path(output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for class_name in ['waterbird', 'landbird']:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)

    # Create symbolic links
    for _, row in data.iterrows():
        img_path = Path(row['img_filename'])
        split = row['split']
        y = row['y']

        # Determine the split folder
        if split == 0:
            split_folder = 'train'
        elif split == 1:
            split_folder = 'val'
        elif split == 2:
            split_folder = 'test'
        else:
            raise ValueError(f"Invalid split value: {split}")

        # Determine the class folder
        class_folder = 'waterbird' if y == 1 else 'landbird'
        
        # Define target directory and symlink
        target_dir = Path(output_dir) / split_folder / class_folder
        target_symlink = target_dir / img_path.name

        # Create the symlink
        os.symlink(img_path.resolve(), target_symlink)

# Usage
csv_file = "metadata_edited.csv"
output_dir = "/scratch-shared/eterres/waterbirds/output_split_folders"
create_symlink_folders_with_classes(csv_file, output_dir)
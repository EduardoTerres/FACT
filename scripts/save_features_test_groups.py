import os
import torch
from tqdm.auto import tqdm
import os.path as osp
from dncbm.utils import common_init, get_img_model
from dncbm import arg_parser
import torchvision


class FetchFeaturesForGroups:
    def __init__(self, args=None):
        self.model, self.preprocess = get_img_model(args)
        self.args = args

    def get_probe_out(self, loader):
        """
        Extract features and labels from the dataset loader.
        """
        self.args.device = torch.device(self.args.device)
        count = 0
        with torch.no_grad():
            labels = None
            out = None

            for (inputs, label_batch) in tqdm(loader, desc="Processing batches", unit="batch"):
                count += inputs.shape[0]
                inputs = inputs.to(self.args.device)

                if out is None:
                    out = self.model.encode_image(inputs).detach().cpu()
                    labels = label_batch
                else:
                    out = torch.vstack((out, self.model.encode_image(inputs).detach().cpu()))
                    labels = torch.hstack((labels, label_batch))

                print(f" total data points: {count}")
            assert labels.shape[0] == out.shape[0], "Mismatch between labels and output features."
        return out, labels

    def save_group_features(self):
        """
        Save features and labels for each group into separate directories.
        """
        group_paths = {
            "W@W": "/scratch-shared/eterres/waterbirds/test_groups/W@W",
            "W@L": "/scratch-shared/eterres/waterbirds/test_groups/W@L",
            "L@W": "/scratch-shared/eterres/waterbirds/test_groups/L@W",
            "L@L": "/scratch-shared/eterres/waterbirds/test_groups/L@L"
        }

        # Directories for activations and labels
        activations_dir = osp.join(self.args.probe_data_dir_activations["img"], self.args.probe_split, "group_features")
        labels_dir = osp.join(self.args.probe_split_idxs_dir["img"], self.args.probe_split, "group_features")
        os.makedirs(activations_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for group, path in group_paths.items():
            print(f"Processing group: {group}")

            # Prepare dataset
            dataset = torchvision.datasets.ImageFolder(path, transform=self.preprocess)

            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.args.batch_size, shuffle=False)

            # Extract features and labels
            features, labels = self.get_probe_out(loader)

            # Save features and labels to separate directories
            feature_path = os.path.join(activations_dir, f"{group}_val.pth")
            label_path = os.path.join(labels_dir, f"{group}_all_labels_val.pth")
            torch.save(features, feature_path)
            torch.save(labels, label_path)

            print(f"Saved features for {group} to {feature_path}")
            print(f"Saved labels for {group} to {label_path}")


if __name__ == '__main__':
    # Initialize arguments
    parser = arg_parser.get_common_parser()
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()
    common_init(args)

    # Process group features
    fetch_act = FetchFeaturesForGroups(args)
    fetch_act.save_group_features()
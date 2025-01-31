

autoencoder_input_dim_dict = {'clip_RN50_out': 1024,
                              'clip_ViT-B16_out': 512,  
                              'clip_ViT-L14_out': 768, }

data_dir_root = './data'
save_dir_root = './SAE'
probe_cs_save_dir_root = './probe'
vocab_dir = './vocab'
analysis_dir = './analysis'



probe_dataset_root_dir_dict = {
    "places365": "/scratch-shared/eterres/places365",
    "imagenet": "/scratch-shared/eterres/imagenet/tiny-imagenet-200",
    "full-imagenet": "/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder",
    "cifar10": "./data/activations_img/cifar10",
    "cifar100": "./data/activations_img/cifar100",
    "waterbirds": "/scratch-shared/eterres/waterbirds/output_split_folders",
}

probe_dataset_nclasses_dict = {"places365": 365,
                               "imagenet": 200,
                               "full-imagenet": 1000,
                               "cifar10": 10,
                               "cifar100": 100,
                               "waterbirds": 2}

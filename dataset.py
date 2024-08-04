from torch.utils.data import Dataset, WeightedRandomSampler, Subset
from PIL import Image
import numpy as np

# Define the custom dataset
class ISICDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.classes = sorted(set(labels))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_random_subset_without_given_indices(dataset, num_samples, indices_to_exclude):
    all_indices = list(range(len(dataset)))
    indices_to_include = list(set(all_indices) - set(indices_to_exclude))
    random_indices = np.random.choice(indices_to_include, num_samples, replace=False)

    random_subset = Subset(dataset, random_indices)

    return random_subset, random_indices
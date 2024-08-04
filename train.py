import os
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from torchvision.models import vit_b_16

from utils import fix_random_seeds, clip_gradients, compute_knn_accuracy
from dataset import ISICDataset, get_random_subset_without_given_indices
from torch.utils.data import WeightedRandomSampler, Subset
from dino import DataAugmentationDINO, MultiCropWrapper, DINOHead, DINOLoss

import wandb

def main():
    print("DINO Training started")
    print(r"""  
    ^..^      /
    /_/\_____/
       /\   /\
      /  \ /  \  
          """)



    # 1. Start a W&B Run


    #  2. Capture a dictionary of hyperparameters
    config = {
                    "epochs": 100, 
                    "learning_rate": 0.0005 * 16 / 256, 
                    "batch_size": 64, 
                    "momentum_teacher": 0.995, 
                    "embedding_size": 1024,
                    "warmup_teacher_temp": 0.04,
                    "teacher_temp": 0.04,
                    "warmup_teacher_temp_epochs": 0,
                    "student_temp": 0.1,
                    "center_momentum": 0.9,
                    "local_crops_number": 4,
                    "log_number": 1000,
                }
    
    run = wandb.init(
        project="DINO ISIC24",
        tags=["dino", "isic", "vit"],
        config=config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    fix_random_seeds(42)

    metadata = pd.read_csv('data/metadata.csv')
    labels = metadata['malignant'].values.astype(int)
    files = [f"data/ISIC_2024_Training_Input/{f}" for f in os.listdir('data/ISIC_2024_Training_Input') if f.endswith('.jpg')]

    transform = DataAugmentationDINO(global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4), local_crops_number=8)

    dataset = ISICDataset(files, labels, transform=transform)
    dataset_loader = DataLoader(dataset, batch_size=wandb.config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)


    targets = torch.tensor(dataset.labels)
    class_counts = torch.bincount(targets)
    num_zeros = class_counts[0].item()
    num_ones = class_counts[1].item()

    # Create weight tensor
    weights = torch.ones(len(targets))
    weights[targets == 1] = num_zeros / num_ones

    sampler = WeightedRandomSampler(weights, 500, replacement=False)

    # sample all the indices via the weighted sampler
    indices = list(sampler)

    sampler = WeightedRandomSampler(weights, 5000, replacement=False)
    indices_train = list(sampler)

    # remove from train indices the indices that are in the validation indices
    indices_train = [idx for idx in indices_train if idx not in indices]

    # get the subset of the dataset
    val_knn_subset = Subset(dataset, indices)

    # get the remaining indices
    train_knn_subset = Subset(dataset, indices_train)

    student = vit_b_16(weights=None)
    teacher = vit_b_16(weights=None)

    teacher.load_state_dict(student.state_dict())

    student = MultiCropWrapper(student, DINOHead(768, 1024))
    teacher = MultiCropWrapper(teacher, DINOHead(768, 1024))

    student = student.cuda()
    teacher = teacher.cuda()

    for p in teacher.parameters():
        p.requires_grad = False

    dino_loss = DINOLoss(
        wandb.config["embedding_size"],
        wandb.config["local_crops_number"]+2,
        wandb.config["warmup_teacher_temp"],
        wandb.config["teacher_temp"],
        wandb.config["warmup_teacher_temp_epochs"],
        wandb.config["epochs"]
    )

    dino_loss = dino_loss.cuda()

    lr = wandb.config["learning_rate"]
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-6)
    momentum_teacher = wandb.config["momentum_teacher"]

    log_number = wandb.config["log_number"]

    epochs = wandb.config["epochs"]

    for e in range(epochs):
        num_batches = 0
        for images, _ in tqdm(dataset_loader):
            images = [img.cuda() for img in images]
            
            with torch.autocast(device_type="cuda"):
                student_output = student(images)
                teacher_output = teacher(images[:2])

                loss = dino_loss(student_output, teacher_output, e)

            wandb.log({"loss": loss})

            optimizer.zero_grad()
            loss.backward()
            clip_gradients(student)
            optimizer.step()

            with torch.no_grad():
                for student_ps, teacher_ps in zip(
                    student.parameters(), teacher.parameters()
                ):
                    teacher_ps.data.mul_(momentum_teacher)
                    teacher_ps.data.add_(
                        (1 - momentum_teacher) * student_ps.detach().data
                    )

            num_batches += 1

            if (num_batches % log_number) == 0:
                print(f"Calculating KNN accuracy for report")
                acc, preds, train_lbls, val_lbls = compute_knn_accuracy(student.backbone, train_knn_subset, val_knn_subset, device, 64)
                wandb.log({"knn_acc": acc})

        # save both the student and the teacher after each epoch
        torch.save(student.state_dict(), f"models/student_{e}.pth")
        torch.save(teacher.state_dict(), f"models/teacher_{e}.pth")

        run.log_model(path="models/student_{e}.pth", name=f"student_{e}")
        run.log_model(path="models/teacher_{e}.pth", name=f"teacher_{e}")


if __name__ == "__main__":
    main()
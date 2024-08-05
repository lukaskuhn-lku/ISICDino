import os
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from torchvision.models import vit_b_16

from utils import fix_random_seeds, clip_gradients, compute_knn_accuracy
from dataset import ISICDataset
from torch.utils.data import Subset
from dino import DataAugmentationDINO, MultiCropWrapper, DINOHead, Loss
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

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
                "train_size": 10000,
                "epochs": 100, 
                "learning_rate": 1e-3, 
                "batch_size": 32, 
                "momentum_teacher": 0.995, 
                "embedding_size": 1024,
                "warmup_teacher_temp": 0.04,
                "teacher_temp": 0.04,
                "warmup_teacher_temp_epochs": 0,
                "student_temp": 0.1,
                "center_momentum": 0.9,
                "local_crops_number": 8,
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

    total_train_size = wandb.config["train_size"]

    norm_only = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_train = ISICDataset(files, labels, transform=transform)
    dataset_knn = ISICDataset(files, labels, transform=norm_only)

    targets = torch.tensor(labels)
    # get all indices where targets is 1
    positive_indices = torch.where(targets == 1)[0]

    # get 10.000 random indices where targets is 0
    negative_indices = torch.where(targets == 0)[0]
    negative_indices_train = negative_indices[torch.randperm(negative_indices.size(0))[:total_train_size-len(positive_indices)]]

    # combine positive and negative indices
    indices = torch.cat([positive_indices, negative_indices_train])

    train_dataset = Subset(dataset_train, indices)

    # get 50% of the positive indices
    positive_indices_knn_val = positive_indices[torch.randperm(positive_indices.size(0))[:len(positive_indices)//2]]

    # fill up to 1000 indices with negative indices
    negative_indices_knn_val = negative_indices[torch.randperm(negative_indices.size(0))[:1000-len(positive_indices_knn_val)]]

    knn_val_dataset = Subset(dataset_knn, torch.cat([positive_indices_knn_val, negative_indices_knn_val]))

    # get the rest of the positive indices
    positive_indices_knn_train = positive_indices[torch.randperm(positive_indices.size(0))[len(positive_indices)//2:]]

    # fill up to 5000 indices with negative indices
    negative_indices_knn_train = negative_indices[torch.randperm(negative_indices.size(0))[1000-len(positive_indices_knn_val):(5000-len(positive_indices_knn_train))+(1000-len(positive_indices_knn_val))]]

    knn_train_dataset = Subset(dataset_knn, torch.cat([positive_indices_knn_train, negative_indices_knn_train]))

    dataset_loader = DataLoader(train_dataset, batch_size=wandb.config["batch_size"], shuffle=True)

    student = vit_b_16(weights=None)
    teacher = vit_b_16(weights=None)

    teacher.load_state_dict(student.state_dict())

    student = MultiCropWrapper(student, DINOHead(768, wandb.config["embedding_size"]))
    teacher = MultiCropWrapper(teacher, DINOHead(768, wandb.config["embedding_size"]))

    student = student.to(device)
    teacher = teacher.to(device)

    for p in teacher.parameters():
        p.requires_grad = False

    dino_loss = Loss(
        wandb.config["embedding_size"],
        wandb.config["teacher_temp"],
        wandb.config["student_temp"],
        wandb.config["center_momentum"],
    )

    dino_loss = dino_loss.to(device)

    lr = wandb.config["learning_rate"]
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-6)
    momentum_teacher = wandb.config["momentum_teacher"]

    epochs = wandb.config["epochs"]

    for e in range(epochs):
        num_batches = 0
        for images, _ in tqdm(dataset_loader):
            images = [img.to(device) for img in images]
            
            with torch.autocast(device_type="cuda"):
                student_output = student(images)
                teacher_output = teacher(images[:2])

                loss = dino_loss(student_output, teacher_output)

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

        print(f"Calculating KNN accuracy for report")
        acc, preds, train_lbls, val_lbls = compute_knn_accuracy(student.backbone, knn_train_dataset, knn_val_dataset, device, 64)
        wandb.log({"knn_acc": acc})

        # save both the student and the teacher after each epoch
        torch.save(student.state_dict(), f"models/student_{e}.pth")
        torch.save(teacher.state_dict(), f"models/teacher_{e}.pth")

        # save only the backbone of the student
        torch.save(student.backbone.state_dict(), f"models/student_backbone_{e}.pth")

        run.log_model(path="models/student_backbone_{e}.pth", name=f"student_backbone_{e}")

if __name__ == "__main__":
    main()
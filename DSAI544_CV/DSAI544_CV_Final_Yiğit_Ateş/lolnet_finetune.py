import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import polars as pl
from pathlib import Path
from PIL import Image
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import json

WANDB_API_KEY = "PUT_YOUR_API_KEY_HERE"
WANDB_MODE = "online"

METADATA_PATH = Path("metadata.csv")
FRAMES_DIR = Path("processed_frames")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_VIDEOS = ["kled_top", "zed_jungle", "galio_mid", "kaisa_adc"]
TEST_VIDEOS = ["fiora_top", "trundle_jungle", "katarina_mid", "twitch_adc"]

BIN_FINE_MAPPING = {"0-5min": 0, "5-10min": 1, "10-15min": 2, "15-20min": 3, 
                     "20-25min": 4, "25-30min": 5, "30+min": 6}
BIN_COARSE_MAPPING = {"early_game": 0, "mid_game": 1, "late_game": 2}

BIN_FINE_LABELS = ["0-5min", "5-10min", "10-15min", "15-20min", "20-25min", "25-30min", "30+min"]
BIN_COARSE_LABELS = ["early_game", "mid_game", "late_game"]
ROLE_LABELS = ["top", "jungle", "mid", "adc"]

def get_role(video_id):
    if "_top" in video_id:
        return "top"
    elif "_jungle" in video_id:
        return "jungle"
    elif "_mid" in video_id:
        return "mid"
    elif "_adc" in video_id:
        return "adc"
    return "unknown"

def get_train_videos(metadata_path):
    df = pl.read_csv(metadata_path)
    all_videos = df.select("video_id").unique().to_series().to_list()
    train_videos = [v for v in all_videos if v not in VAL_VIDEOS and v not in TEST_VIDEOS]
    return train_videos

TRAIN_VIDEOS = get_train_videos(METADATA_PATH)

try:
    wandb.login(key=WANDB_API_KEY)
except:
    print("⚠️  W&B login failed, using offline mode")
    WANDB_MODE = "offline"

class LoLDataset(Dataset):
    def __init__(self, metadata_df, transform=None, task="regression"):
        self.metadata = metadata_df
        self.transform = transform
        self.task = task
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata[idx]
        img_path = FRAMES_DIR / row["relative_path"]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        if self.task == "regression":
            label = torch.tensor(row["timestamp_seconds"], dtype=torch.float32)
        elif self.task == "fine":
            label = torch.tensor(BIN_FINE_MAPPING[row["bin_fine"]], dtype=torch.long)
        elif self.task == "coarse":
            label = torch.tensor(BIN_COARSE_MAPPING[row["bin_coarse"]], dtype=torch.long)
        
        return image, label, row["video_id"], row["bin_fine"], row["bin_coarse"]

def create_datasets(metadata_path, task="regression"):
    df = pl.read_csv(metadata_path)
    
    train_df = df.filter(pl.col("video_id").is_in(TRAIN_VIDEOS)).to_dicts()
    val_df = df.filter(pl.col("video_id").is_in(VAL_VIDEOS)).to_dicts()
    test_df = df.filter(pl.col("video_id").is_in(TEST_VIDEOS)).to_dicts()
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = LoLDataset(train_df, transform=train_transform, task=task)
    val_dataset = LoLDataset(val_df, transform=val_transform, task=task)
    test_dataset = LoLDataset(test_df, transform=val_transform, task=task)
    
    return train_dataset, val_dataset, test_dataset

def create_model(task="regression", num_classes=None):
    model = models.resnet50(pretrained=True)
    
    if task == "regression":
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model.to(DEVICE)

def train_epoch(model, dataloader, criterion, optimizer, task):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels, _, _, _ in tqdm(dataloader, desc="Training"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        if task == "regression":
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if task == "regression":
            all_preds.extend(outputs.detach().cpu().numpy())
        else:
            all_preds.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    
    if task == "regression":
        mae = mean_absolute_error(all_labels, all_preds)
        return avg_loss, mae
    else:
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, acc

def validate_simple(model, dataloader, criterion, task):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _, _, _ in tqdm(dataloader, desc="Validation"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            
            if task == "regression":
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            if task == "regression":
                all_preds.extend(outputs.cpu().numpy())
            else:
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    
    if task == "regression":
        mae = mean_absolute_error(all_labels, all_preds)
        return avg_loss, mae
    else:
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, acc

def validate_detailed(model, dataloader, criterion, task):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_video_ids = []
    all_bins_fine = []
    all_bins_coarse = []
    
    with torch.no_grad():
        for images, labels, video_ids, bins_fine, bins_coarse in tqdm(dataloader, desc="Validation"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            
            if task == "regression":
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            if task == "regression":
                all_preds.extend(outputs.cpu().numpy())
            else:
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_video_ids.extend(video_ids)
            all_bins_fine.extend(bins_fine)
            all_bins_coarse.extend(bins_coarse)
    
    avg_loss = running_loss / len(dataloader)
    
    return avg_loss, all_preds, all_labels, all_video_ids, all_bins_fine, all_bins_coarse

def compute_detailed_metrics(preds, labels, video_ids, bins_fine, bins_coarse, task):
    results = {}
    
    if task == "regression":
        overall_mae = mean_absolute_error(labels, preds)
        results["overall_mae"] = overall_mae
        
        role_metrics = {}
        for role in ROLE_LABELS:
            role_mask = [get_role(vid) == role for vid in video_ids]
            if sum(role_mask) > 0:
                role_preds = [p for p, m in zip(preds, role_mask) if m]
                role_labels = [l for l, m in zip(labels, role_mask) if m]
                role_metrics[role] = mean_absolute_error(role_labels, role_preds)
        results["per_role_mae"] = role_metrics
        
        bin_fine_metrics = {}
        for bin_label in BIN_FINE_LABELS:
            bin_mask = [b == bin_label for b in bins_fine]
            if sum(bin_mask) > 0:
                bin_preds = [p for p, m in zip(preds, bin_mask) if m]
                bin_labels_vals = [l for l, m in zip(labels, bin_mask) if m]
                bin_fine_metrics[bin_label] = mean_absolute_error(bin_labels_vals, bin_preds)
        results["per_bin_fine_mae"] = bin_fine_metrics
        
        bin_coarse_metrics = {}
        for bin_label in BIN_COARSE_LABELS:
            bin_mask = [b == bin_label for b in bins_coarse]
            if sum(bin_mask) > 0:
                bin_preds = [p for p, m in zip(preds, bin_mask) if m]
                bin_labels_vals = [l for l, m in zip(labels, bin_mask) if m]
                bin_coarse_metrics[bin_label] = mean_absolute_error(bin_labels_vals, bin_preds)
        results["per_bin_coarse_mae"] = bin_coarse_metrics
        
    else:
        overall_acc = accuracy_score(labels, preds)
        results["overall_accuracy"] = overall_acc
        
        role_metrics = {}
        for role in ROLE_LABELS:
            role_mask = [get_role(vid) == role for vid in video_ids]
            if sum(role_mask) > 0:
                role_preds = [p for p, m in zip(preds, role_mask) if m]
                role_labels = [l for l, m in zip(labels, role_mask) if m]
                role_metrics[role] = accuracy_score(role_labels, role_preds)
        results["per_role_accuracy"] = role_metrics
        
        bin_fine_metrics = {}
        for bin_label in BIN_FINE_LABELS:
            bin_mask = [b == bin_label for b in bins_fine]
            if sum(bin_mask) > 0:
                bin_preds = [p for p, m in zip(preds, bin_mask) if m]
                bin_labels_vals = [l for l, m in zip(labels, bin_mask) if m]
                bin_fine_metrics[bin_label] = accuracy_score(bin_labels_vals, bin_preds)
        results["per_bin_fine_accuracy"] = bin_fine_metrics
        
        bin_coarse_metrics = {}
        for bin_label in BIN_COARSE_LABELS:
            bin_mask = [b == bin_label for b in bins_coarse]
            if sum(bin_mask) > 0:
                bin_preds = [p for p, m in zip(preds, bin_mask) if m]
                bin_labels_vals = [l for l, m in zip(labels, bin_mask) if m]
                bin_coarse_metrics[bin_label] = accuracy_score(bin_labels_vals, bin_preds)
        results["per_bin_coarse_accuracy"] = bin_coarse_metrics
    
    return results

def save_results(results, task_name, task_type):
    results_file = RESULTS_DIR / f"{task_name}_results.json"
    
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {str(k): float(v) if isinstance(v, (np.float32, np.float64, np.int64)) else v 
                                         for k, v in value.items()}
        else:
            serializable_results[key] = float(value) if isinstance(value, (np.float32, np.float64, np.int64)) else value
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✅ Results saved to {results_file}")

def log_detailed_metrics(results, task, prefix, task_name):
    if task == "regression":
        wandb.log({f"{prefix}_overall_mae": results["overall_mae"]})
        
        for role, mae in results["per_role_mae"].items():
            wandb.log({f"{prefix}_mae_{role}": mae})
        
        for bin_label, mae in results["per_bin_fine_mae"].items():
            wandb.log({f"{prefix}_mae_fine_{bin_label}": mae})
        
        for bin_label, mae in results["per_bin_coarse_mae"].items():
            wandb.log({f"{prefix}_mae_coarse_{bin_label}": mae})
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        roles = list(results["per_role_mae"].keys())
        role_maes = list(results["per_role_mae"].values())
        axes[0].bar(roles, role_maes)
        axes[0].set_title("MAE by Role")
        axes[0].set_ylabel("MAE (seconds)")
        
        bins_fine = list(results["per_bin_fine_mae"].keys())
        bin_fine_maes = list(results["per_bin_fine_mae"].values())
        axes[1].bar(range(len(bins_fine)), bin_fine_maes)
        axes[1].set_xticks(range(len(bins_fine)))
        axes[1].set_xticklabels(bins_fine, rotation=45)
        axes[1].set_title("MAE by Fine Bins")
        axes[1].set_ylabel("MAE (seconds)")
        
        bins_coarse = list(results["per_bin_coarse_mae"].keys())
        bin_coarse_maes = list(results["per_bin_coarse_mae"].values())
        axes[2].bar(bins_coarse, bin_coarse_maes)
        axes[2].set_title("MAE by Coarse Bins")
        axes[2].set_ylabel("MAE (seconds)")
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{task_name}_breakdown.png", dpi=150, bbox_inches='tight')
        wandb.log({f"{prefix}_breakdown": wandb.Image(fig)})
        plt.close()
        
    else:
        wandb.log({f"{prefix}_overall_accuracy": results["overall_accuracy"]})
        
        for role, acc in results["per_role_accuracy"].items():
            wandb.log({f"{prefix}_accuracy_{role}": acc})
        
        for bin_label, acc in results["per_bin_fine_accuracy"].items():
            wandb.log({f"{prefix}_accuracy_fine_{bin_label}": acc})
        
        for bin_label, acc in results["per_bin_coarse_accuracy"].items():
            wandb.log({f"{prefix}_accuracy_coarse_{bin_label}": acc})
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        roles = list(results["per_role_accuracy"].keys())
        role_accs = list(results["per_role_accuracy"].values())
        axes[0].bar(roles, role_accs)
        axes[0].set_title("Accuracy by Role")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_ylim([0, 1])
        
        bins_fine = list(results["per_bin_fine_accuracy"].keys())
        bin_fine_accs = list(results["per_bin_fine_accuracy"].values())
        axes[1].bar(range(len(bins_fine)), bin_fine_accs)
        axes[1].set_xticks(range(len(bins_fine)))
        axes[1].set_xticklabels(bins_fine, rotation=45)
        axes[1].set_title("Accuracy by Fine Bins")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim([0, 1])
        
        bins_coarse = list(results["per_bin_coarse_accuracy"].keys())
        bin_coarse_accs = list(results["per_bin_coarse_accuracy"].values())
        axes[2].bar(bins_coarse, bin_coarse_accs)
        axes[2].set_title("Accuracy by Coarse Bins")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{task_name}_breakdown.png", dpi=150, bbox_inches='tight')
        wandb.log({f"{prefix}_breakdown": wandb.Image(fig)})
        plt.close()

def generate_gradcam_samples(model, test_dataset, task_type, task_name, num_samples=6):
    model.eval()
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    fig, axes = plt.subplots(2, num_samples, figsize=(18, 6))
    
    indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    for idx, sample_idx in enumerate(indices):
        image_tensor, label, video_id, bin_fine, bin_coarse = test_dataset[sample_idx]
        
        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        
        if task_type == "regression":
            grayscale_cam = cam(input_tensor=input_tensor)
        else:
            targets = [ClassifierOutputTarget(label.item())]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
        grayscale_cam = grayscale_cam[0, :]
        
        img_path = FRAMES_DIR / f"{video_id}" / test_dataset.metadata[sample_idx]["filename"]
        original_img = Image.open(img_path).convert("RGB").resize((224, 224))
        rgb_img = np.array(original_img) / 255.0
        
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        axes[0, idx].imshow(original_img)
        axes[0, idx].axis('off')
        axes[0, idx].set_title(f"Original", fontsize=8)
        
        axes[1, idx].imshow(visualization)
        axes[1, idx].axis('off')
        axes[1, idx].set_title(f"Grad-CAM", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{task_name}_gradcam.png", dpi=150, bbox_inches='tight')
    return fig

def train_model_simple(task_name, task_type, num_classes=None, epochs=20, lr=1e-4, batch_size=32):
    wandb.init(project="lol-time-prediction", name=f"{task_name}", mode=WANDB_MODE, reinit=True)
    
    train_dataset, val_dataset, test_dataset = create_datasets(METADATA_PATH, task=task_type)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = create_model(task=task_type, num_classes=num_classes)
    
    if task_type == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    wandb.config.update({
        "task": task_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "optimizer": "AdamW",
        "model": "ResNet50",
        "train_videos": len(TRAIN_VIDEOS),
        "val_videos": len(VAL_VIDEOS),
        "test_videos": len(TEST_VIDEOS)
    })
    
    best_val_metric = float('inf') if task_type == "regression" else 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_metric = train_epoch(model, train_loader, criterion, optimizer, task_type)
        val_loss, val_metric = validate_simple(model, val_loader, criterion, task_type)
        
        scheduler.step(val_loss)
        
        if task_type == "regression":
            print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_metric:.2f}s")
            print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_metric:.2f}s")
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_metric,
                "val_loss": val_loss,
                "val_mae": val_metric,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                torch.save(model.state_dict(), f"best_{task_name}.pth")
                wandb.run.summary["best_val_mae"] = val_metric
        else:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_metric:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metric:.4f}")
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metric,
                "val_loss": val_loss,
                "val_accuracy": val_metric,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                torch.save(model.state_dict(), f"best_{task_name}.pth")
                wandb.run.summary["best_val_accuracy"] = val_metric
    
    model.load_state_dict(torch.load(f"best_{task_name}.pth"))
    test_loss, test_metric = validate_simple(model, test_loader, criterion, task_type)
    
    if task_type == "regression":
        print(f"\nTest MAE: {test_metric:.2f}s ({test_metric/60:.2f} minutes)")
        wandb.run.summary["test_mae"] = test_metric
        return test_metric
    else:
        print(f"\nTest Accuracy: {test_metric:.4f}")
        wandb.run.summary["test_accuracy"] = test_metric
        return test_metric
    
    wandb.finish()

def train_model_detailed(task_name, task_type, num_classes=None, epochs=20, lr=1e-4, batch_size=32):
    wandb.init(project="lol-time-prediction", name=f"{task_name}", mode=WANDB_MODE, reinit=True)
    
    train_dataset, val_dataset, test_dataset = create_datasets(METADATA_PATH, task=task_type)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = create_model(task=task_type, num_classes=num_classes)
    
    if task_type == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    wandb.config.update({
        "task": task_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "optimizer": "AdamW",
        "model": "ResNet50",
        "train_videos": len(TRAIN_VIDEOS),
        "val_videos": len(VAL_VIDEOS),
        "test_videos": len(TEST_VIDEOS)
    })
    
    best_val_metric = float('inf') if task_type == "regression" else 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_metric = train_epoch(model, train_loader, criterion, optimizer, task_type)
        val_loss, val_preds, val_labels, val_video_ids, val_bins_fine, val_bins_coarse = validate_detailed(model, val_loader, criterion, task_type)
        
        if task_type == "regression":
            val_metric = mean_absolute_error(val_labels, val_preds)
        else:
            val_metric = accuracy_score(val_labels, val_preds)
        
        scheduler.step(val_loss)
        
        if task_type == "regression":
            print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_metric:.2f}s")
            print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_metric:.2f}s")
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_metric,
                "val_loss": val_loss,
                "val_mae": val_metric,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                torch.save(model.state_dict(), f"best_{task_name}.pth")
                wandb.run.summary["best_val_mae"] = val_metric
        else:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_metric:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metric:.4f}")
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metric,
                "val_loss": val_loss,
                "val_accuracy": val_metric,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                torch.save(model.state_dict(), f"best_{task_name}.pth")
                wandb.run.summary["best_val_accuracy"] = val_metric
    
    model.load_state_dict(torch.load(f"best_{task_name}.pth"))
    test_loss, test_preds, test_labels, test_video_ids, test_bins_fine, test_bins_coarse = validate_detailed(model, test_loader, criterion, task_type)
    
    test_results = compute_detailed_metrics(test_preds, test_labels, test_video_ids, test_bins_fine, test_bins_coarse, task_type)
    
    if task_type == "regression":
        print(f"\nOverall Test MAE: {test_results['overall_mae']:.2f}s")
        print(f"\nPer-Role MAE:")
        for role, mae in test_results["per_role_mae"].items():
            print(f"  {role}: {mae:.2f}s")
        print(f"\nPer-Bin Fine MAE:")
        for bin_label, mae in test_results["per_bin_fine_mae"].items():
            print(f"  {bin_label}: {mae:.2f}s")
        print(f"\nPer-Bin Coarse MAE:")
        for bin_label, mae in test_results["per_bin_coarse_mae"].items():
            print(f"  {bin_label}: {mae:.2f}s")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(test_labels, test_preds, alpha=0.5)
        ax.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], 'r--')
        ax.set_xlabel("Actual Time (seconds)")
        ax.set_ylabel("Predicted Time (seconds)")
        ax.set_title(f"{task_name} - Test Set Predictions")
        plt.savefig(RESULTS_DIR / f"{task_name}_scatter.png", dpi=150, bbox_inches='tight')
        wandb.log({"test_scatter": wandb.Image(fig)})
        plt.close()
    else:
        print(f"\nOverall Test Accuracy: {test_results['overall_accuracy']:.4f}")
        print(f"\nPer-Role Accuracy:")
        for role, acc in test_results["per_role_accuracy"].items():
            print(f"  {role}: {acc:.4f}")
        print(f"\nPer-Bin Fine Accuracy:")
        for bin_label, acc in test_results["per_bin_fine_accuracy"].items():
            print(f"  {bin_label}: {acc:.4f}")
        print(f"\nPer-Bin Coarse Accuracy:")
        for bin_label, acc in test_results["per_bin_coarse_accuracy"].items():
            print(f"  {bin_label}: {acc:.4f}")
        
        cm = confusion_matrix(test_labels, test_preds)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{task_name} - Confusion Matrix")
        plt.savefig(RESULTS_DIR / f"{task_name}_confusion.png", dpi=150, bbox_inches='tight')
        wandb.log({"confusion_matrix": wandb.Image(fig)})
        plt.close()
    
    log_detailed_metrics(test_results, task_type, prefix="test", task_name=task_name)
    
    save_results(test_results, task_name, task_type)
    
    print("\nGenerating Grad-CAM visualizations...")
    gradcam_fig = generate_gradcam_samples(model, test_dataset, task_type, task_name, num_samples=6)
    wandb.log({"gradcam_visualization": wandb.Image(gradcam_fig)})
    plt.close()
    
    wandb.finish()
    print(f"\n{'='*60}")
    print(f"✅ {task_name} training complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    
    print("="*60)
    print("DATASET INFO")
    print("="*60)
    print(f"Training videos: {len(TRAIN_VIDEOS)}")
    print(f"Validation videos: {len(VAL_VIDEOS)}")
    print(f"Test videos: {len(TEST_VIDEOS)}")
    print(f"Train: {TRAIN_VIDEOS}")
    print(f"Val: {VAL_VIDEOS}")
    print(f"Test: {TEST_VIDEOS}")
    print("="*60)
    
    print("\n" + "="*60)
    print("PHASE 1: HYPERPARAMETER TUNING (9 RUNS)")
    print("="*60)
    
    learning_rates = [1e-4, 3e-4, 1e-3]
    
    lr_results = {
        "regression": {},
        "fine_bins": {},
        "coarse_bins": {}
    }
    
    print("\n[1/3] Regression LR Tuning")
    for lr in learning_rates:
        print(f"\n--- Testing LR={lr} for Regression ---")
        test_metric = train_model_simple(f"regression_lr{lr}", "regression", epochs=25, lr=lr, batch_size=32)
        lr_results["regression"][f"lr_{lr}"] = float(test_metric)
        wandb.finish()
    
    print("\n[2/3] Fine Bins LR Tuning")
    for lr in learning_rates:
        print(f"\n--- Testing LR={lr} for Fine Bins ---")
        test_metric = train_model_simple(f"fine_bins_lr{lr}", "fine", num_classes=7, epochs=25, lr=lr, batch_size=32)
        lr_results["fine_bins"][f"lr_{lr}"] = float(test_metric)
        wandb.finish()
    
    print("\n[3/3] Coarse Bins LR Tuning")
    for lr in learning_rates:
        print(f"\n--- Testing LR={lr} for Coarse Bins ---")
        test_metric = train_model_simple(f"coarse_bins_lr{lr}", "coarse", num_classes=3, epochs=25, lr=lr, batch_size=32)
        lr_results["coarse_bins"][f"lr_{lr}"] = float(test_metric)
        wandb.finish()
    
    with open(RESULTS_DIR / "lr_tuning_summary.json", 'w') as f:
        json.dump(lr_results, f, indent=2)
    
    print("\n" + "="*60)
    print("LR TUNING COMPLETE - RESULTS:")
    print("="*60)
    print(json.dumps(lr_results, indent=2))
    
    best_lr_regression = min(lr_results["regression"].items(), key=lambda x: x[1])[0].replace("lr_", "")
    best_lr_fine = min(lr_results["fine_bins"].items(), key=lambda x: -x[1])[0].replace("lr_", "")
    best_lr_coarse = min(lr_results["coarse_bins"].items(), key=lambda x: -x[1])[0].replace("lr_", "")
    
    print(f"\nBest LRs:")
    print(f"  Regression: {best_lr_regression}")
    print(f"  Fine Bins: {best_lr_fine}")
    print(f"  Coarse Bins: {best_lr_coarse}")
    
    print("\n" + "="*60)
    print("PHASE 2: FINAL MODELS WITH DETAILED ANALYSIS (3 RUNS)")
    print("="*60)
    
    print(f"\n[1/3] Training final regression model (LR={best_lr_regression})")
    train_model_detailed("regression_final", "regression", epochs=25, lr=float(best_lr_regression), batch_size=32)
    
    print(f"\n[2/3] Training final fine_bins model (LR={best_lr_fine})")
    train_model_detailed("fine_bins_final", "fine", num_classes=7, epochs=25, lr=float(best_lr_fine), batch_size=32)
    
    print(f"\n[3/3] Training final coarse_bins model (LR={best_lr_coarse})")
    train_model_detailed("coarse_bins_final", "coarse", num_classes=3, epochs=25, lr=float(best_lr_coarse), batch_size=32)
    
    print("\n" + "="*60)
    print("ALL TRAINING COMPLETE!")
    print("="*60)
    print(f"\nResults saved in: {RESULTS_DIR}")
    print("\nGenerated files:")
    for file in sorted(RESULTS_DIR.glob("*")):
        print(f"  - {file.name}")
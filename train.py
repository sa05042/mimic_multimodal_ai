import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import AutoTokenizer
import yaml
from dataset import MIMICCXRWithLabels
from model import BioViLClassifier, clip_contrastive_loss
from report_generation import ReportGenerator
from utils import set_seed, save_checkpoint
import open_clip

# Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
set_seed(config["seed"])
device = config["device"]

# Data
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
dataset = MIMICCXRWithLabels(config["preprocessed_csv"], transform, tokenizer, config["max_length"], config["num_classes"])
train_size = int(config["train_val_split"] * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

# Model
base_model, _, _ = open_clip.create_model_and_transforms("BiomedVLP-BioViL-T", pretrained=True)
base_model = base_model.to(device)
base_model.eval()
model_cls = BioViLClassifier(base_model, num_classes=config["num_classes"]).to(device)

# Report generator
report_gen = ReportGenerator(base_model, config["report_model_name"], device=device)

# Optimizer & loss
optimizer = torch.optim.AdamW(model_cls.parameters(), lr=config["lr"])
criterion = torch.nn.BCEWithLogitsLoss()

def train_epoch(loader):
    model_cls.train()
    total_loss = 0
    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs, img_feat, txt_feat = model_cls(images, input_ids)
        cls_loss = criterion(outputs, labels)
        clip_loss = clip_contrastive_loss(img_feat, txt_feat)

        if config["mode"] == "contrastive":
            loss = clip_loss
        elif config["mode"] == "classification":
            loss = cls_loss
        elif config["mode"] == "joint":
            loss = cls_loss + config["alpha"] * clip_loss
        elif config["mode"] == "report_generation":
            # Placeholder for report generation fine-tuning
            loss = cls_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_epoch(loader):
    model_cls.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs, img_feat, txt_feat = model_cls(images, input_ids)
            cls_loss = criterion(outputs, labels)
            clip_loss = clip_contrastive_loss(img_feat, txt_feat)

            if config["mode"] == "contrastive":
                loss = clip_loss
            elif config["mode"] == "classification":
                loss = cls_loss
            elif config["mode"] == "joint":
                loss = cls_loss + config["alpha"] * clip_loss
            elif config["mode"] == "report_generation":
                loss = cls_loss
            total_loss += loss.item()
    return total_loss / len(loader)

# Training loop
best_val_loss = float("inf")
for epoch in range(config["epochs"]):
    train_loss = train_epoch(train_loader)
    val_loss = validate_epoch(val_loader)
    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model_cls, config["output_dir"])
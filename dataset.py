import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class MIMICCXRWithLabels(Dataset):
    def __init__(self, csv_file, transform=None, tokenizer=None, max_length=128, num_classes=14):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes
        self.df.fillna("", inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["file_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        report = row["report_text"]
        if self.tokenizer:
            tokens = self.tokenizer(
                report,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
        else:
            input_ids = None
            attention_mask = None
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "report_text": report
        }
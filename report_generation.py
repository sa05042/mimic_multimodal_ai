import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ReportGenerator:
    def __init__(self, image_encoder, report_model_name="google/medmimic-t5-base", device="cuda"):
        self.image_encoder = image_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(report_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(report_model_name).to(device)
        self.device = device

    def generate_report(self, images, max_length=512):
        self.model.eval()
        with torch.no_grad():
            img_emb = self.image_encoder.encode_image(images.to(self.device))
            # Convert embeddings into prompt for LLM (simple placeholder)
            prefix_texts = ["Image embedding prefix"] * images.size(0)
            inputs = self.tokenizer(prefix_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model.generate(**inputs, max_length=max_length)
            reports = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return reports
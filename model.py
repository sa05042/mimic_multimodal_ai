import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

class BioViLClassifier(nn.Module):
    def __init__(self, base_model, num_classes=14):
        super().__init__()
        self.base_model = base_model
        vision_dim = base_model.visual_proj_dim
        text_dim = base_model.text_proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(vision_dim + text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, input_ids):
        img_feat = self.base_model.encode_image(image)
        txt_feat = self.base_model.encode_text(input_ids)
        fused = torch.cat([img_feat, txt_feat], dim=-1)
        out = self.classifier(fused)
        return out, img_feat, txt_feat

def clip_contrastive_loss(image_feats, text_feats, temperature=0.07):
    image_norm = F.normalize(image_feats, dim=1)
    text_norm  = F.normalize(text_feats, dim=1)
    logits = image_norm @ text_norm.T / temperature
    labels = torch.arange(len(logits)).to(image_norm.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2
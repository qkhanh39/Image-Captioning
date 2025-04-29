import torch.nn.functional as F
from torchvision import models, transforms
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from ultils.ultils import *
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
i2w, w2i = load_i2w_and_w2i()
embedding_matrix = load_embedding_matrix()


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg              = models.vgg16(pretrained=True)
        self.features    = vgg.features
        self.avgpooling  = vgg.avgpool
        self.flatten     = nn.Flatten()

        for param in self.features.parameters():
            param.requires_grad = False
    def forward(self, x):
        out = self.features(x)
        out = self.avgpooling(out)
        out = self.flatten(out)

        return out


class VGGLSTM(nn.Module):
    def __init__(self, input_feature_size, vocab_size, embed_size, hidden_size):
        super(VGGLSTM, self).__init__()
        self.vgg = VGG()
        self.feature_projector = nn.Linear(input_feature_size, embed_size)
        self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, image, captions):
        features = self.vgg(image)
        projected_features = self.feature_projector(features)  # [batch, embed_size]
        projected_features = projected_features.unsqueeze(1)  # [batch, 1, embed_size]

        captions_embed = self.embed(captions)  # [batch, seq_len, embed_size]
        lstm_input = torch.cat((projected_features, captions_embed), dim=1)

        lstm_out, _ = self.lstm(lstm_input)
        out = self.fc(lstm_out[:, -1, :])  # Output from the last time step
        return out

def load_model():
    model = VGGLSTM(input_feature_size=512*7*7, vocab_size=len(w2i), embed_size=200, hidden_size=512)
    checkpoint = torch.load("assets/MLP.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def generate_caption(model, image):
    transform = load_transform()
    model = model
    image = image
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dim

    caption = generate_caption_beam_for_mlp(model, image_tensor, i2w, w2i, device=device)
    return caption


# if __name__ == "__main__":
#     model = load_model()
#     caption = generate_caption(model, "statics/test.jpg")
#     print(caption)


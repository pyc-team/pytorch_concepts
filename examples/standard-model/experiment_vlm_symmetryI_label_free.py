import os
import csv

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

image_folder = "./test_data"
output_file = "./clip_experiment/results.csv"
model_name = "openai/clip-vit-base-patch32"

os.makedirs('./clip_experiment', exist_ok=True)

# Load model and processor
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

# Load images
image_files = sorted(
    f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
)

# Encode text prompt once
text_inputs = processor(text=["red"], return_tensors="pt", padding=True)
with torch.inference_mode():
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Compute dot product for each image and save
rows = []
for fname in image_files:
    image = Image.open(os.path.join(image_folder, fname)).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    dot_product = (text_features @ image_features.T).item()
    print(f"{fname}: {dot_product:.4f}")
    rows.append({"model": model_name, "image": fname, "dot_product": dot_product})

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "image", "dot_product"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nResults saved to {output_file}")

import os
import csv
import shutil
import itertools

import pandas as pd
import torch
from collections import Counter
from PIL import Image, ImageDraw
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

image_folder = "./test_data"
output_file = "./vlm_experiment/results.csv"

os.makedirs('./vlm_experiment', exist_ok=True)

# Pairwise prompt: model sees a side-by-side image labelled A (left) and B (right)
PROMPT = (
    "You are shown two images side by side, labelled A (left) and B (right). "
    "Which image shows a STRONGER / HIGHER degree of red color? "
    "Reply with a single letter: A or B."
)

# Load the 20 images, sorted so filenames are deterministic
image_files = sorted(
    f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
)[:20]


def make_side_by_side(img_a: Image.Image, img_b: Image.Image, label_height: int = 20) -> Image.Image:
    """Concatenate two images horizontally with A/B labels."""
    w, h = img_a.width, img_a.height
    canvas = Image.new("RGB", (w * 2 + 10, h + label_height), (255, 255, 255))
    canvas.paste(img_a, (0, label_height))
    canvas.paste(img_b, (w + 10, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((w // 2 - 4, 2), "A", fill=(0, 0, 0))
    draw.text((w + 10 + w // 2 - 4, 2), "B", fill=(0, 0, 0))
    return canvas


def get_majority_vote(predictions):
    """Get the most frequent prediction from a list, return None if no valid predictions."""
    valid_predictions = [p for p in predictions if p in ("A", "B")]
    if not valid_predictions:
        return None
    counter = Counter(valid_predictions)
    return counter.most_common(1)[0][0]


def _seed_hf_cache(local_path: str, model_name: str) -> None:
    """Copy *.py files from a local model folder into the HuggingFace dynamic-module
    cache so that transformers can resolve relative imports at load time."""
    cache_dir = os.path.expanduser(
        f"~/.cache/huggingface/modules/transformers_modules/{model_name}"
    )
    os.makedirs(cache_dir, exist_ok=True)
    for fname in os.listdir(local_path):
        if fname.endswith(".py"):
            src = os.path.join(local_path, fname)
            dst = os.path.join(cache_dir, fname)
            shutil.copy2(src, dst)


def run_vlm_batch(model_name, local_path, mode="standard"):
    print(f"\n--- Running {model_name} (pairwise) ---")

    if model_name == "moondream2":
        _seed_hf_cache(local_path, model_name)
        model = AutoModelForCausalLM.from_pretrained(local_path, trust_remote_code=True).to("mps")
        proc = AutoTokenizer.from_pretrained(local_path)
    elif model_name == "qwen":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            local_path, torch_dtype=torch.float16, device_map="mps"
        )
        proc = AutoProcessor.from_pretrained(local_path, use_fast=False)

    model.eval()
    images = {name: Image.open(os.path.join(image_folder, name)).convert("RGB") for name in image_files}
    pairs = list(itertools.permutations(image_files, 2))
    print(f"Running {len(pairs)} pairwise comparisons...")

    if mode == "prototype" and model_name == "moondream2":
        model_name = "moondream2_proto"
        proto_folder = './train_data'
        proto_image_files = sorted(
            f for f in os.listdir(proto_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )[:20]
        proto_images = {name: Image.open(os.path.join(proto_folder, name)).convert("RGB") for name in proto_image_files}
        distances = pd.DataFrame(columns=proto_image_files, index=image_files)
        if os.path.exists('./vlm_experiment/prototype_distances.csv'):
            distances = pd.read_csv('./vlm_experiment/prototype_distances.csv', index_col=0)
        else:
            for img_name, img in proto_images.items():
                print(f"  Running prototype image {img_name} through vision encoder...")
                with torch.inference_mode():
                    prototype = model.model._run_vision_encoder(img)

                    for img_name_test, img_test in images.items():
                        print(f"    Running test image {img_name_test} through vision encoder...")
                        test_tensor = model.model._run_vision_encoder(img_test)
                        dist = torch.norm(prototype - test_tensor, p=2)
                        distances.loc[img_name_test, img_name] = dist.item()

            distances.to_csv('./vlm_experiment/prototype_distances.csv')

        img_ranking = distances.values.argmin(axis=1)
        results = []
        for img_a_name, img_b_name in pairs:
            results.append({
                "model": model_name,
                "image_a": img_a_name,
                "image_b": img_b_name,
                "prediction": "A" if img_ranking[image_files.index(img_a_name)] > img_ranking[image_files.index(img_b_name)] else "B",
            })

        return results

    results = []
    for img_a_name, img_b_name in pairs:
        combined = make_side_by_side(images[img_a_name], images[img_b_name])

        # Run the same question 3 times for robustness
        predictions = []
        for trial in range(1):
            try:
                with torch.inference_mode():
                    if model_name == "moondream2":

                        enc = model.encode_image(combined)
                        answer = model.answer_question(enc, PROMPT, proc)

                    elif model_name == "qwen":
                        combined.thumbnail((512, 512), Image.LANCZOS)
                        messages = [{
                            "role": "user",
                            "content": [
                                {"type": "image", "image": combined},
                                {"type": "text", "text": PROMPT},
                            ],
                        }]
                        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = proc(text=[text], images=[combined], return_tensors="pt").to("mps")
                        out = model.generate(
                            **inputs,
                            max_new_tokens=5,
                            do_sample=True,
                            temperature=0.7,
                            use_cache=True
                        )
                        generated = out[0][inputs["input_ids"].shape[-1]:]
                        answer = proc.decode(generated, skip_special_tokens=True).strip()

                letter = next((c for c in answer.upper() if c in ("A", "B")), None)
                predictions.append(letter)
                print(f"  {img_a_name} vs {img_b_name} trial {trial+1} -> {answer.strip()!r}  (letter: {letter})")
            except Exception as e:
                print(f"  Error on {img_a_name} vs {img_b_name} trial {trial+1}: {e}")
                predictions.append(None)

        # Get majority vote from the 3 trials
        final_prediction = get_majority_vote(predictions)
        print(f"  Final prediction for {img_a_name} vs {img_b_name}: {final_prediction} (from {predictions})")

        results.append({
            "model": model_name,
            "image_a": img_a_name,
            "image_b": img_b_name,
            "prediction": final_prediction,  # "A" means A is more red, "B" means B is more red
        })

    del model
    torch.mps.empty_cache()
    return results


# Run models
all_data = []
all_data.extend(run_vlm_batch("moondream2", "./models/moondream2", mode="prototype"))
all_data.extend(run_vlm_batch("moondream2", "./models/moondream2"))
all_data.extend(run_vlm_batch("qwen", "./models/qwen2-vl"))

# Save to CSV
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "image_a", "image_b", "prediction"])
    writer.writeheader()
    writer.writerows(all_data)

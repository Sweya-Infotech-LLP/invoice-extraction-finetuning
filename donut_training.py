# !pip install transformers datasets accelerate torch torchvision datasets 

import os
import json
import io
import torch
import gc
from PIL import Image
from datasets import Dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Trainer,
    TrainingArguments,
)

gc.collect()
torch.cuda.empty_cache()

with open("donut_dataset.json", "r") as f:
    samples = json.load(f)

hf_dataset = Dataset.from_list(samples)

def load_image(example):
    """Loads the image from a file path into a PIL Image object."""
    image_path = example["image"]
    example["image"] = Image.open(image_path).convert("RGB")
    return example

hf_dataset = hf_dataset.map(load_image)

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

task_prompt = "<s_invoice>"
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(task_prompt)
model.config.pad_token_id = processor.tokenizer.pad_token_id

def preprocess(example):
    """
    Prepares the data for the model.
    - Resizes image and creates pixel_values.
    - Tokenizes ground_truth to create labels.
    """
    image_bytes = example["image"]["bytes"]
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((384, 384))
    pixel_values = processor(image, return_tensors="pt").pixel_values[0]

    labels = processor.tokenizer(
        example["ground_truth"],
        max_length=512, 
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids[0]

    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

tokenized_dataset = hf_dataset.map(preprocess, remove_columns=["image", "ground_truth"])
tokenized_dataset.set_format("torch")

training_args = TrainingArguments(
    output_dir="./donut-invoice-custom",
    per_device_train_batch_size=1,
    num_train_epochs=5,
    fp16=True,                         
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,       
    optim="adamw_8bit",               
    logging_steps=100,
    report_to="none",
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("Starting training with memory optimizations...")
trainer.train()
print("Training done successfully!")


model.save_pretrained("./donut-invoice-custom")
processor.save_pretrained("./donut-invoice-custom")

print("Model saved successfully")
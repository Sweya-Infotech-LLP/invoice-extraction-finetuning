import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel


processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image_path = "./output_images/34a340f1-page_1.png"
image = Image.open(image_path).convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

task_prompt = "<s_invoice>"
decoder_input_id = processor.tokenizer.convert_tokens_to_ids(task_prompt)

model.eval()
with torch.no_grad():
    outputs = model.generate(
        pixel_values,
        decoder_start_token_id=decoder_input_id,
        max_length=512,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id
    )

prediction = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print("\n🔍 Predicted Output:")
print(prediction)
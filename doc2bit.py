# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# # Use the original base repository, not the mradermacher GGUF fork
# model_id = "erax-ai/EraX-VL-2B-V1.5"

# # Load the model
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_id, 
#     torch_dtype="auto", 
#     device_map="auto" # Automatically offloads to GPU if available
# )

# # Load the processor for handling text and images
# processor = AutoProcessor.from_pretrained(model_id)

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pdf2image import convert_from_path

# 1. Load the model and processor
model_id = "erax-ai/EraX-VL-2B-V1.5"
print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype = "auto",
    device_map = "auto"
)
processor = AutoProcessor.from_pretrained(model_id)

def extract_text_from_pdf(pdf_path):
    # 2. Convert PDF pages to PIL images
    print(f"Converting {pdf_path} to images ...")
    # you can limit with first_page and last_page  arguments if it's a huge PDF
    poppler_dir = r"./bin"
    pages = convert_from_path(pdf_path, poppler_path = poppler_dir)

    extracted_full_text = []

    # 3. Iterate through each page and perform OCR
    for page_num, page_image in enumerate(pages, start = 1):
        print(f"Processing page {page_num}/{len(pages)}...")

        # format the prompt for qwen2-vl architecture
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": page_image},
                    {"type": "text", "text": "Extract all the text from this image"}
                ],
            }
        ]

        # prepare inputs using the processor
        text = processor.apply_chat_template(
            messages, tokenize = False, add_generation_prompt = True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text = [text],
            images = image_inputs,
            videos = video_inputs,
            padding = True,
            return_tensors = "pt",
        )

        # move inputs to the same device as the model
        inputs = inputs.to(model.device)

        # 4. Generate the OCR output
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens = 15000)

        # trim the prompt tokens from the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # decode the generated text
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens =  True, clean_up_tokenization = False
        )

        # save the text for this page
        page_text = f"--- Page {page_num} ---\n{output_text[0]}\n"
        extracted_full_text.append(page_text)
    
    return "\n".join(extracted_full_text)


# --- run the script ---
if __name__ == "__main__":
    pdf_file = "data/BA.12.2004.HS-ST.NGUYEN VAN TRONG-MAI XUAN CUONG-PHAM VAN NHO/KL_DIEU TRA.pdf"
    final_text = extract_text_from_pdf(pdf_file)
    print("\n\n=== OCR Extraction Result ===")
    print(final_text)
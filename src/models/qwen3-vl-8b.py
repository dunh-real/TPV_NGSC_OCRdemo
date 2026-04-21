import os
import json
from PIL import Image
from pdf2image import convert_from_path
import ollama

poppler_dir = r"../../bin"

def ocr_pdf_with_bbox(pdf_path, model_name = "qwen3-vl:8b"):
    pages = convert_from_path(pdf_path, poppler_path = poppler_dir, dpi = 300)
    results = []

    for i, page in enumerate(pages):
        temp_image_path = f"temp_page_{i}.png"
        page.save(temp_image_path, "PNG")

        prompt = (
            "Detect and recognize, extract all text in this image."
            "For each text segment, provide the bouding box in [ymin, xmin, ymax, xmax] format."
            "Return only a JSON list of objects: {'text': '...', 'bbox': [y1, x1, y2, x2]}."
        )

        response = ollama.chat(
            model = model_name,
            messages = [{
                'role': 'user',
                'content': prompt,
                'images': [temp_image_path]
            }],
            options = {
                'temperature': 0.1, # bọn m đặt cái này thấp thôi k là tọa độ bbox nhảy disco giờ
                'num_predict': 4096
            }
        )

        content = response['message']['content']
        results.append({
            "page": i + 1,
            "raw_output": content
        })
        os.remove(temp_image_path)
    
    return results

# để anh ví dụ cho bọn m xem này :))
pdf_file = "../../data/BA.12.2004.HS-ST.NGUYEN VAN TRONG-MAI XUAN CUONG-PHAM VAN NHO/KL_DIEU TRA.pdf"
final_result = ocr_pdf_with_bbox(pdf_file)
for item in final_result:
    print(f"\n--- Trang {item['page']} ---")
    print(item['raw_output'])

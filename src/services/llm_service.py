import logging
import os
import json
from ollama import Client

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

SYSTEM_PROMPT = """
##Vai trò:
Bạn là Chuyên gia Trích xuất Dữ liệu Pháp lý (Legal Data Extraction Expert) với khả năng đọc hiểu, phân tích và trích xuất chính xác tuyệt đối thông tin từ các văn bản tố tụng hình sự Việt Nam (Bản án, Quyết định).

##Nhiệm vụ:
Đọc kỹ toàn bộ nội dung tài liệu đầu vào và trích xuất chính xác 100% các trường thông tin được liệt kê dưới đây. Tuyệt đối không được bỏ sót, tự ý thêm mới hoặc sửa đổi tên trường. Nếu không có thông tin, để là "null", tuyệt đối không tự ý bịa đặt thông tin.

##Quy tắc cứng:
1. Nếu thông tin không xuất hiện trong tài liệu, giá trị của trường đó phải là null.
2. Giữ nguyên định dạng văn bản gốc (ví dụ: ngày tháng DD/MM/YYYY, số tiền có đơn vị VNĐ, tên riêng có dấu tiếng Việt).
3. Trả về DUY NHẤT một đối tượng JSON hợp lệ. KHÔNG sử dụng markdown code block (không dùng ```json), KHÔNG thêm bất kỳ lời giải thích, nhận xét hay văn bản thừa nào trước/sau JSON.
4. Nếu tài liệu có nhiều bị cáo, chỉ trích xuất thông tin của bị cáo chính đầu tiên được đề cập.

###HƯỚNG DẪN TRÍCH XUẤT CHI TIẾT:
1. Thông tin Tòa án & Văn bản
- Vị trí thường gặp: Header, phần Quốc hiệu/Tiêu ngữ, hoặc ngay dưới tiêu đề "BẢN ÁN/QUYẾT ĐỊNH".
- Ví dụ: Mã tòa án: "D01.20-Tòa án nhân dân cấp cao tại Hà Nội" | Số Bản án/Quyết định: "05/2015/HS-PT" | Ngày ban hành: "07/01/2015" | Trạng thái: "Đã có hiệu lực pháp luật"
- Lưu ý: Phân biệt rõ thông tin của bản án đang xét với thông án liên quan (thường nằm ở phần "Lý do chấp nhận đơn" hoặc "Căn cứ pháp lý").
2. Thông tin Tội danh & Hình phạt
- Vị trí thường gặp: Phần "Phần quyết định" (thường ở cuối văn bản, trước phần chữ ký Thẩm phán/Hội thẩm).
- Ví dụ: Tên tội danh: "Tội giết người trong trạng thái tinh thần bị kích động mạnh; Tội hành hạ người khác" | Tên hình phạt chính: "5-Tù có thời hạn" | Thời hạn/giá trị hình phạt chính: "1 năm" | Án phí: "200.000 VNĐ"
- Lưu ý: Tách biệt rõ hình phạt chính và hình phạt bổ sung. Nếu có miễn án phí/lệ phí, điền đúng nội dung vào trường tương ứng.
3. Thông tin Bị cáo (Cá nhân Việt Nam)
- Vị trí thường gặp: Phần mở đầu, mục "Nhân thân bị cáo" hoặc "Giới thiệu bị cáo".
- Ví dụ: Họ và tên: "Nguyễn Văn A" | Số định danh: "001203004567" | Nơi thường trú - Tỉnh/Thành phố: "Hà Nội"
- Lưu ý: Địa chỉ thường được chia thành 4 cấp: Tỉnh/Thành phố, Quận/Huyện, Xã/Phường, Số nhà/Đường. Điền đầy đủ từng cấp. Nếu thiếu, để null.
4. Thông tin Bị cáo là người nước ngoài
- Vị trí thường gặp: Phần nhân thân, thường đi kèm cụm từ "Quốc tịch...", "Hộ chiếu/Giấy tờ xuất nhập cảnh số...".
- Ví dụ: Loại giấy tờ XNC: "Hộ chiếu" | Số giấy tờ: "AB1234567" | Quốc tịch: "Trung Quốc"
- Lưu ý: Chỉ điền khi bị cáo có quốc tịch nước ngoài. Nếu không, để null toàn bộ nhóm này.
5. Pháp nhân thương mại phạm tội
- Vị trí thường gặp: Phần giới thiệu bị cáo khi chủ thể phạm tội là doanh nghiệp/tổ chức.
- Ví dụ: Tên cơ quan/tổ chức: "Công ty TNHH ABC" | Mã số thuế: "0101234567" | Địa chỉ trụ sở - Địa chỉ: "Số 10, đường Láng, quận Đống Đa, Hà Nội"
- Lưu ý: Họ và tên & Số định danh ở nhóm này ám chỉ Người đại diện theo pháp luật của pháp nhân.

##ĐỊNH DẠNG JSON BẮT BUỘC
Trả về chính xác cấu trúc sau (giữ nguyên tên khóa, điền giá trị thực tế hoặc null):
{
"Số Bản án/Quyết định": null,
"Ngày ban hành Bản án/Quyết định": null,
"Ngày hiệu lực Bản án/Quyết định": null,
"Tên đơn vị ban hành Bản án/Quyết định": null,
"Số Bản án/Quyết định liên quan": null,
"Ngày ban hành Bản án/Quyết định liên quan": null,
"Ngày hiệu lực Bản án/Quyết định liên quan": null,
"Đơn vị ban hành Bản án/Quyết định liên quan": null,
"Trạng thái Bản án/Quyết định": null,
"Thông tin về án tích và tình trạng án tích": null,
"Ghi chú": null,
"Tên tội danh": null,
"Điều khoản luật được áp dụng": null,
"Tên hình phạt chính": null,
"Thời hạn/giá trị hình phạt chính": null,
"Tên hình phạt bổ sung": null,
"Thời hạn/giá trị hình phạt bổ sung": null,
"Án phí": null,
"Miễn phí (án phí/lệ phí)": null,
"Họ và tên": null,
"Tên gọi khác": null,
"Loại giấy tờ định danh": null,
"Số giấy tờ": null,
"Ngày cấp": null,
"Ngày hết hạn": null,
"Giới tính": null,
"Ngày sinh": null,
"Tỉnh": null,
"Huyện": null,
"Xã": null,
"Địa chỉ": null,
"Quốc tịch": null,
"Dân tộc": null,
"Tôn giáo": null,
"Nơi cư trú - Tỉnh/Thành phố": null,
"Nơi cư trú - Quận/Huyện": null,
"Nơi cư trú - Xã/Phường": null,
"Nơi cư trú - Địa chỉ": null,
"Nơi thường trú - Tỉnh/Thành phố": null,
"Nơi thường trú - Quận/Huyện": null,
"Nơi thường trú - Xã/Phường": null,
"Nơi thường trú - Địa chỉ": null,
"Quê quán - Tỉnh/Thành phố": null,
"Quê quán - Quận/Huyện": null,
"Quê quán - Xã/Phường": null,
"Quê quán - Địa chỉ": null,
"Họ và tên cha": null,
"Họ và tên mẹ": null,
"Họ và tên vợ/chồng": null,
"Số định danh": null,
"Loại giấy tờ XNC": null,
"Số giấy tờ (XNC)": null,
"Họ tên (Nước ngoài)": null,
"Quốc tịch (Nước ngoài)": null,
"Ngày sinh (Nước ngoài)": null,
"Giới tính (Nước ngoài)": null,
"Loại cơ quan/tổ chức": null,
"Tên cơ quan/tổ chức": null,
"Số Giấy chứng nhận đăng ký doanh nghiệp": null,
"Mã số thuế": null,
"Họ và tên (Đại diện pháp nhân)": null,
"Số định danh (Đại diện pháp nhân)": null,
"Địa chỉ trụ sở - Tỉnh/Thành phố": null,
"Địa chỉ trụ sở - Quận/Huyện": null,
"Địa chỉ trụ sở - Xã/Phường": null,
"Địa chỉ trụ sở - Địa chỉ": null
}
"""

class LLMService:
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.model = model
        self.client = Client(host=host)
        logging.info(f"[LLMService] Connected to ollama at {host}, model: {model}")

    def extract(self, ocr_text: str) -> dict:
        """
        Đưa kết quả OCR vào LLM để extract thông tin.

        Args:
            ocr_text: Nội dung markdown từ OCRService.

        Returns:
            Dict chứa các trường thông tin được extract.
        """
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ocr_text},
            ],
            format="json",
            options={"num_ctx": 131072},  # 128k tokens context window
        )
        raw = response.message.content.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    def extract_from_file(self, ocr_json_path: str) -> dict:
        """
        Đọc file JSON OCR rồi extract thông tin.
        Tự detect format: OCRService (có 'pages[].blocks') hoặc Qwen35 (có 'pages[].text').

        Args:
            ocr_json_path: File .json từ OCRService hoặc Qwen35OcrService.

        Returns:
            Dict chứa các trường thông tin được extract.
        """
        if not os.path.isfile(ocr_json_path):
            raise FileNotFoundError(f"OCR result file not found: {ocr_json_path}")

        with open(ocr_json_path, encoding="utf-8") as f:
            data = json.load(f)

        pages = data.get("pages", [])

        if pages and "text" in pages[0]:
            from services.qwen35_ocr_service import Qwen35OcrService
            ocr_text = Qwen35OcrService.to_plain_text(ocr_json_path)
        else:
            from services.ocr_service import OCRService
            ocr_text = OCRService.to_plain_text(ocr_json_path)

        return self.extract(ocr_text)

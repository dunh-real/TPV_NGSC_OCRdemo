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

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from typing import Optional, List


class DiaChi(BaseModel):
    """Cấu trúc dùng chung cho các trường địa chỉ (Cấp tỉnh, huyện, xã, chi tiết)"""
    tinh_thanh_pho: Optional[str] = Field(None, description="Tỉnh hoặc Thành phố trực thuộc trung ương")
    quan_huyen: Optional[str] = Field(None, description="Quận, Huyện, Thị xã hoặc Thành phố thuộc tỉnh")
    xa_phuong: Optional[str] = Field(None, description="Xã, Phường hoặc Thị trấn")
    dia_chi_chi_tiet: Optional[str] = Field(None, description="Số nhà, đường phố, thôn xóm (Địa chỉ cụ thể)")

class GiayToDinhDanh(BaseModel):
    """Cấu trúc dùng chung cho giấy tờ tùy thân"""
    loai_giay_to: Optional[str] = Field(None, description="Loại giấy tờ định danh (VD: CCCD, CMND, Hộ chiếu)")
    so_giay_to: Optional[str] = Field(None, description="Số của giấy tờ định danh")
    ngay_cap: Optional[str] = Field(None, description="Ngày cấp giấy tờ (Định dạng DD/MM/YYYY nếu có)")
    ngay_het_han: Optional[str] = Field(None, description="Ngày hết hạn của giấy tờ (Định dạng DD/MM/YYYY nếu có)")


class ThongTinChungBanAn(BaseModel):
    """Thông tin chung về Bản án/Quyết định (Cột 1 đến 23)"""
    so_ban_an_quyet_dinh: Optional[str] = Field(None, description="Số Bản án/Quyết định (1)")
    ngay_ban_hanh: Optional[str] = Field(None, description="Ngày ban hành Bản án/Quyết định (2)")
    ngay_hieu_luc: Optional[str] = Field(None, description="Ngày hiệu lực Bản án/Quyết định (3)")
    ten_don_vi_ban_hanh: Optional[str] = Field(None, description="Tên đơn vị ban hành (Tòa án) (4,5)")
    so_ban_an_lien_quan: Optional[str] = Field(None, description="Số Bản án/Quyết định liên quan (6)")
    ngay_ban_hanh_lien_quan: Optional[str] = Field(None, description="Ngày ban hành Bản án/Quyết định liên quan (7)")
    ngay_hieu_luc_lien_quan: Optional[str] = Field(None, description="Ngày hiệu lực Bản án/Quyết định liên quan (8)")
    don_vi_ban_hanh_lien_quan: Optional[str] = Field(None, description="Đơn vị ban hành Bản án/Quyết định liên quan (9)")
    trang_thai_ban_an: Optional[str] = Field(None, description="Trạng thái Bản án/Quyết định (10)")
    thong_tin_an_tich: Optional[str] = Field(None, description="Thông tin về án tích và tình trạng án tích (11)")
    ghi_chu: Optional[str] = Field(None, description="Ghi chú chung (12)")
    ten_toi_danh: Optional[str] = Field(None, description="Tên tội danh bị kết án (13,14)")
    dieu_khoan_luat_ap_dung: Optional[str] = Field(None, description="Điều khoản luật được áp dụng (15)")
    ten_hinh_phat_chinh: Optional[str] = Field(None, description="Tên hình phạt chính (16,17)")
    thoi_han_gia_tri_hinh_phat_chinh: Optional[str] = Field(None, description="Thời hạn/giá trị hình phạt chính (18)")
    ten_hinh_phat_bo_sung: Optional[str] = Field(None, description="Tên hình phạt bổ sung (19,20)")
    thoi_han_gia_tri_hinh_phat_bo_sung: Optional[str] = Field(None, description="Thời hạn/giá trị hình phạt bổ sung (21)")
    an_phi: Optional[str] = Field(None, description="Án phí phải nộp (22)")
    mien_phi: Optional[str] = Field(None, description="Miễn phí (án phí/lệ phí) (23)")

class ThongTinBiCao(BaseModel):
    """Thông tin bị cáo (Cột 24.1 đến 24.15)"""
    ho_va_ten: Optional[str] = Field(None, description="Họ và tên bị cáo (24.1)")
    ten_goi_khac: Optional[str] = Field(None, description="Tên gọi khác/Bí danh (24.2)")
    giay_to_dinh_danh: Optional[GiayToDinhDanh] = Field(None, description="Thông tin giấy tờ định danh (24.3.1 - 24.3.4)")
    gioi_tinh: Optional[str] = Field(None, description="Giới tính (24.4)")
    ngay_sinh: Optional[str] = Field(None, description="Ngày tháng năm sinh (24.5)")
    noi_sinh: Optional[DiaChi] = Field(None, description="Thông tin nơi sinh (Tỉnh, Huyện, Xã, Địa chỉ) (24.6.1 - 24.6.4)")
    quoc_tich: Optional[str] = Field(None, description="Quốc tịch (24.7)")
    dan_toc: Optional[str] = Field(None, description="Dân tộc (24.8)")
    ton_giao: Optional[str] = Field(None, description="Tôn giáo (24.9)")
    noi_cu_tru: Optional[DiaChi] = Field(None, description="Nơi cư trú hiện tại (24.10.1 - 24.10.4)")
    noi_thuong_tru: Optional[DiaChi] = Field(None, description="Nơi đăng ký thường trú (24.11.1 - 24.11.4)")
    que_quan: Optional[DiaChi] = Field(None, description="Quê quán (24.12.1 - 24.12.4)")
    ho_ten_cha: Optional[str] = Field(None, description="Họ và tên cha (24.13)")
    ho_ten_me: Optional[str] = Field(None, description="Họ và tên mẹ (24.14)")
    ho_ten_vo_chong: Optional[str] = Field(None, description="Họ và tên vợ/chồng (24.15)")

class ThongTinBiCaoNuocNgoai(BaseModel):
    """Dành cho bị cáo là người nước ngoài (Cột 25.1 đến 25.7)"""
    so_dinh_danh: Optional[str] = Field(None, description="Số định danh cá nhân người nước ngoài (25.1)")
    loai_giay_to_xnc: Optional[str] = Field(None, description="Loại giấy tờ Xuất nhập cảnh (25.2)")
    so_giay_to: Optional[str] = Field(None, description="Số giấy tờ Xuất nhập cảnh (25.3)")
    ho_ten: Optional[str] = Field(None, description="Họ tên người nước ngoài (25.4)")
    quoc_tich: Optional[str] = Field(None, description="Quốc tịch (25.5)")
    ngay_sinh: Optional[str] = Field(None, description="Ngày sinh (25.6)")
    gioi_tinh: Optional[str] = Field(None, description="Giới tính (25.7)")

class PhapNhanThuongMai(BaseModel):
    """Pháp nhân thương mại phạm tội (Cột 26.1 đến 26.6.4)"""
    loai_co_quan_to_chuc: Optional[str] = Field(None, description="Loại cơ quan/tổ chức (26.1)")
    ten_co_quan_to_chuc: Optional[str] = Field(None, description="Tên cơ quan/tổ chức (26.2)")
    so_gcn_dang_ky_doanh_nghiep: Optional[str] = Field(None, description="Số Giấy chứng nhận đăng ký doanh nghiệp (26.3)")
    ma_so_thue: Optional[str] = Field(None, description="Mã số thuế (26.4)")
    ho_ten_nguoi_dai_dien: Optional[str] = Field(None, description="Họ và tên người đại diện pháp luật (26.5.1)")
    so_dinh_danh_nguoi_dai_dien: Optional[str] = Field(None, description="Số định danh người đại diện (26.5.2)")
    dia_chi_tru_so: Optional[DiaChi] = Field(None, description="Địa chỉ trụ sở chính (26.6.1 - 26.6.4)")


class DuLieuBanAn(BaseModel):
    """
    Class Root chứa toàn bộ thông tin được bóc tách từ file tài liệu.
    Sử dụng List[] cho phần thông tin bị cáo vì một bản án có thể có nhiều bị cáo hoặc nhiều pháp nhân.
    """
    thong_tin_chung: ThongTinChungBanAn
    danh_sach_bi_cao: Optional[List[ThongTinBiCao]] = Field(default_factory=list, description="Danh sách các bị cáo (người Việt Nam) trong bản án")
    danh_sach_bi_cao_nuoc_ngoai: Optional[List[ThongTinBiCaoNuocNgoai]] = Field(default_factory=list, description="Danh sách bị cáo là người nước ngoài")
    danh_sach_phap_nhan_pham_toi: Optional[List[PhapNhanThuongMai]] = Field(default_factory=list, description="Danh sách pháp nhân thương mại phạm tội")


class ExtractService:
    def __init__(self):
        self.llm_model = OLLAMA_MODEL
        self.structured_llm = self.llm_model.get_llm().with_structured_output(DuLieuBanAn)
    
    async def extract_document(self,document:str):
        prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một trợ lý pháp lý và chuyên gia phân tích dữ liệu chuyên nghiệp. 
        Nhiệm vụ của bạn là đọc nội dung văn bản OCR từ một bản án/quyết định của tòa án và 
        trích xuất thông tin chính xác theo định dạng được yêu cầu.
        Nếu thông tin nào không có trong văn bản, hãy để trống (null). Không được tự bịa đặt dữ liệu.
        TUYỆT ĐỐI không chèn thêm bất kỳ hội thoại, giải thích hay lời chào nào (phải trả về đúng chuẩn parser JSON)."""),
        ("user", "Nội dung OCR của bản án:\n\n{document}")
    ])
        chain = prompt | self.structured_llm
        result = await chain.ainvoke({"document": document})
        return result
    
        
        
import logging
import os
import json
from ollama import Client
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List
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




class DiaChi(BaseModel):
    tinh_thanh_pho: Optional[str] = Field(None, description="Tỉnh hoặc Thành phố trực thuộc trung ương")
    quan_huyen: Optional[str] = Field(None, description="Quận, Huyện, Thị xã hoặc Thành phố thuộc tỉnh")
    xa_phuong: Optional[str] = Field(None, description="Xã, Phường hoặc Thị trấn")
    dia_chi_chi_tiet: Optional[str] = Field(None, description="Số nhà, đường phố, thôn xóm (Địa chỉ cụ thể)")

class GiayToDinhDanh(BaseModel):
    loai_giay_to: Optional[str] = Field(None, description="Loại giấy tờ định danh (VD: CCCD, CMND, Hộ chiếu)")
    so_giay_to: Optional[str] = Field(None, description="Số của giấy tờ định danh")
    ngay_cap: Optional[str] = Field(None, description="Ngày cấp giấy tờ (Định dạng DD/MM/YYYY nếu có)")
    ngay_het_han: Optional[str] = Field(None, description="Ngày hết hạn của giấy tờ (Định dạng DD/MM/YYYY nếu có)")


class ThongTinChungBanAn(BaseModel):
    """Thông tin chung, áp dụng cho toàn bộ văn bản (Chỉ bóc tách 1 lần)"""
    so_ban_an_quyet_dinh: Optional[str] = Field(None, description="Số Bản án/Quyết định")
    ngay_ban_hanh: Optional[str] = Field(None, description="Ngày ban hành Bản án/Quyết định")
    ngay_hieu_luc: Optional[str] = Field(None, description="Ngày hiệu lực Bản án/Quyết định")
    ten_don_vi_ban_hanh: Optional[str] = Field(None, description="Tên đơn vị ban hành (Tòa án)")
    so_ban_an_lien_quan: Optional[str] = Field(None, description="Số Bản án/Quyết định liên quan")
    ngay_ban_hanh_lien_quan: Optional[str] = Field(None, description="Ngày ban hành Bản án/Quyết định liên quan")
    ngay_hieu_luc_lien_quan: Optional[str] = Field(None, description="Ngày hiệu lực Bản án/Quyết định liên quan")
    don_vi_ban_hanh_lien_quan: Optional[str] = Field(None, description="Đơn vị ban hành Bản án/Quyết định liên quan")
    trang_thai_ban_an: Optional[str] = Field(None, description="Trạng thái Bản án/Quyết định")
    ghi_chu: Optional[str] = Field(None, description="Ghi chú chung")


class ThongTinKetAn(BaseModel):
    """Án tích, Tội danh và Hình phạt - Riêng biệt cho từng người"""
    thong_tin_an_tich: Optional[str] = Field(None, description="Thông tin về án tích và tình trạng án tích")
    ten_toi_danh: List[str] = Field(
        default_factory=list,
        description="Danh sách các tội danh bị kết án"
    )
    dieu_khoan_luat_ap_dung: List[str] = Field(
        default_factory=list,
        description="Danh sách các điều, khoản, điểm luật được áp dụng"
    )
    ten_hinh_phat_chinh: Optional[str] = Field(None, description="Tên hình phạt chính")
    thoi_han_gia_tri_hinh_phat_chinh: Optional[str] = Field(None, description="Thời hạn/giá trị hình phạt chính")
    ten_hinh_phat_bo_sung: List[str] = Field(
        default_factory=list,
        description="Danh sách tên các hình phạt bổ sung"
    )
    thoi_han_gia_tri_hinh_phat_bo_sung: Optional[str] = Field(None, description="Thời hạn/giá trị hình phạt bổ sung")
    an_phi: Optional[str] = Field(None, description="Án phí phải nộp")
    mien_phi: Optional[str] = Field(None, description="Miễn phí (án phí/lệ phí)")

class ThongTinBiCao(BaseModel):
    ho_va_ten: Optional[str] = Field(None, description="Họ và tên bị cáo")
    ten_goi_khac: Optional[str] = Field(None, description="Tên gọi khác/Bí danh")
    giay_to_dinh_danh: Optional[GiayToDinhDanh] = Field(None, description="Thông tin giấy tờ định danh")
    gioi_tinh: Optional[str] = Field(None, description="Giới tính")
    ngay_sinh: Optional[str] = Field(None, description="Ngày tháng năm sinh")
    noi_sinh: Optional[DiaChi] = Field(None, description="Thông tin nơi sinh (Tỉnh, Huyện, Xã, Địa chỉ)")
    quoc_tich: Optional[str] = Field(None, description="Quốc tịch")
    dan_toc: Optional[str] = Field(None, description="Dân tộc")
    ton_giao: Optional[str] = Field(None, description="Tôn giáo")
    noi_cu_tru: Optional[DiaChi] = Field(None, description="Nơi cư trú hiện tại")
    noi_thuong_tru: Optional[DiaChi] = Field(None, description="Nơi đăng ký thường trú")
    que_quan: Optional[DiaChi] = Field(None, description="Quê quán")
    ho_ten_cha: Optional[str] = Field(None, description="Họ và tên cha")
    ho_ten_me: Optional[str] = Field(None, description="Họ và tên mẹ")
    ho_ten_vo_chong: Optional[str] = Field(None, description="Họ và tên vợ/chồng")

class ThongTinBiCaoNuocNgoai(BaseModel):
    so_dinh_danh: Optional[str] = Field(None, description="Số định danh cá nhân người nước ngoài")
    loai_giay_to_xnc: Optional[str] = Field(None, description="Loại giấy tờ Xuất nhập cảnh")
    so_giay_to: Optional[str] = Field(None, description="Số giấy tờ Xuất nhập cảnh")
    ho_ten: Optional[str] = Field(None, description="Họ tên người nước ngoài")
    quoc_tich: Optional[str] = Field(None, description="Quốc tịch")
    ngay_sinh: Optional[str] = Field(None, description="Ngày sinh")
    gioi_tinh: Optional[str] = Field(None, description="Giới tính")

class PhapNhanThuongMai(BaseModel):
    loai_co_quan_to_chuc: Optional[str] = Field(None, description="Loại cơ quan/tổ chức")
    ten_co_quan_to_chuc: Optional[str] = Field(None, description="Tên cơ quan/tổ chức")
    so_gcn_dang_ky_doanh_nghiep: Optional[str] = Field(None, description="Số Giấy chứng nhận đăng ký doanh nghiệp")
    ma_so_thue: Optional[str] = Field(None, description="Mã số thuế")
    ho_ten_nguoi_dai_dien: Optional[str] = Field(None, description="Họ và tên người đại diện pháp luật")
    so_dinh_danh_nguoi_dai_dien: Optional[str] = Field(None, description="Số định danh người đại diện")
    dia_chi_tru_so: Optional[DiaChi] = Field(None, description="Địa chỉ trụ sở chính")


class HoSoDoiTuong(BaseModel):
    """Đại diện cho 1 đối tượng bị xét xử"""
    thong_tin_ket_an: ThongTinKetAn = Field(..., description="Tội danh, hình phạt và án phí của riêng đối tượng này")
    bi_cao_vn: Optional[ThongTinBiCao] = Field(None, description="Điền nếu là công dân VN")
    bi_cao_nuoc_ngoai: Optional[ThongTinBiCaoNuocNgoai] = Field(None, description="Điền nếu là người nước ngoài")
    phap_nhan_pham_toi: Optional[PhapNhanThuongMai] = Field(None, description="Điền nếu là pháp nhân")

class DuLieuBanAn(BaseModel):
    """Class Root chứa toàn bộ thông tin bóc tách từ 1 văn bản Bản án/Quyết định"""
    thong_tin_chung: ThongTinChungBanAn = Field(..., description="Thông tin chung của bản án")
    danh_sach_doi_tuong: List[HoSoDoiTuong] = Field(
        default_factory=list,
        description="Danh sách những người/pháp nhân bị xét xử trong bản án này"
    )


class ExtractService:
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        api_base = host if host.endswith("/v1") else f"{host.rstrip('/')}/v1"
        
        self.llm = ChatOpenAI(
            base_url = api_base,
            api_key = "ollama",
            model = model,
            temperature = 0,
            max_tokens = 8192,
            model_kwargs={
                "extra_body": {
                    "options": {
                        "num_ctx": 131072 
                    }
                }
            }
        )
        self.structured_llm = self.llm.with_structured_output(DuLieuBanAn)
    
    async def extract_document(self,document:str):
        prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một trợ lý pháp lý và chuyên gia phân tích dữ liệu chuyên nghiệp.
Nhiệm vụ của bạn là đọc nội dung văn bản OCR từ một bản án/quyết định của tòa án và trích xuất thông tin.TRÍCH XUẤT CÁC NỘI
DUNG TỪ VĂN BẢN ĐẦU VÀO CHÍNH XÁC NHẤT, KHÔNG ĐƯỢC BỎ XÓT, KHÔNG ĐƯỢC BỊA ĐẶT DỮ LIỆU CHỈ LẤY TỪ VĂN BẢN VÀ TRẢ
RA THẬT ĐẦY ĐỦ NHẤT.  PHẦN THÔNG TIN KẾT ÁN CỦA TỪNG BỊ CÁO PHẢI LIỆT KÊ ĐẦY ĐỦ CÁC HÌNH PHẠT CHÍNH, HÌNH PHẠT BỔ SUNG VD: PHẠT TIỀN, BỒI THƯỜNG .... CÁC ĐIỀU KHOẢN ÁP DÙNG PHẢI ĐẦY ĐỦ
VỚI NỘI DUNG TRONG VĂN BẢN. Nếu thông tin nào không có trong văn bản, hãy để trống (null). KHÔNG được tự bịa đặt dữ liệu.
        TUYỆT ĐỐI không chèn thêm bất kỳ hội thoại, giải thích hay lời chào nào (phải trả về đúng chuẩn parser JSON)."""),
        ("user", "Nội dung OCR của bản án:\n\n{document}")
    ])
        chain = prompt | self.structured_llm
        result = await chain.ainvoke({"document": document})
        return result
    
        
        
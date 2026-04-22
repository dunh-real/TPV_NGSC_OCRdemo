import logging
import os
import json
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

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


class LLMService:
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        llm = ChatOllama(model=model, base_url=host)
        self.structured_llm = llm.with_structured_output(DuLieuBanAn)
        logging.info(f"[LLMService] Connected to ollama at {host}, model: {model}")

    def extract(self, ocr_text: str) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là Chuyên gia Trích xuất Dữ liệu Pháp lý (Legal Data Extraction Expert) với khả năng đọc hiểu, phân tích và trích xuất chính xác tuyệt đối thông tin từ các văn bản tố tụng hình sự Việt Nam (Bản án, Quyết định). 
        Nhiệm vụ của bạn là đọc nội dung văn bản OCR từ một bản án/quyết định của tòa án và trích xuất thông tin chính xác theo định dạng được yêu cầu.
        Nếu thông tin nào không có trong văn bản, hãy để trống (null). Không được tự bịa đặt dữ liệu.
        TUYỆT ĐỐI không chèn thêm bất kỳ hội thoại, giải thích hay lời chào nào (phải trả về đúng chuẩn parser JSON)."""),
            ("user", "Nội dung OCR của bản án:\n\n{ocr_text}")
        ])
        chain  = prompt | self.structured_llm
        result = chain.invoke({"ocr_text": ocr_text})

        return result.model_dump() if hasattr(result, "model_dump") else result

    def extract_from_file(self, ocr_json_path: str) -> dict:
        if not os.path.isfile(ocr_json_path):
            raise FileNotFoundError(f"OCR result file not found: {ocr_json_path}")
        
        """
        Chuyển file JSON OCR thành plain text để đưa vào LLM.
        """
        with open(ocr_json_path, encoding="utf-8") as f:
            data = json.load(f)

        parts = []
        for page in data.get("pages", []):
            lines = [b["text"] for b in page.get("blocks", []) if b.get("text", "").strip()]
            if lines:
                parts.append("\n".join(lines))

        ocr_text = "\n".join(parts)

        if not ocr_text.strip():
            raise ValueError(f"No text extracted from: {ocr_json_path}")

        return self.extract(ocr_text)
    
def main():
    llm_service = LLMService()
    ocr_json_path = "./data/result_ocr/BA_05.2021.DS-ST_qwen36.json"

    result = llm_service.extract_from_file(ocr_json_path)

    output_path = "./data/result_extract/BA_05.2021.DS-ST_qwen36.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
        
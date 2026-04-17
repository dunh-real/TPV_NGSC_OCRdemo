from fpdf import FPDF

def txt_to_pdf(input_txt, output_pdf):
    pdf = FPDF()
    pdf.add_page()

    # NOTE: Thêm font chữ hỗ trợ tiếng Việt
    # NOTE: Cần đường dẫn đến file .ttf trên máy tính của con vợ
    # Windows thường để ở: C:\Windows\Fonts\arial.ttf
    try:
        pdf.add_font('Arial', '', r"C:\Windows\Fonts\arial.ttf")
        pdf.set_font('Arial', size=12)
    except:
        print("Không tìm thấy font Arial, đang sử dụng font mặc định (có thể lỗi dấu)")
        pdf.set_font("Helvetica", size=12)

    try:
        with open(input_txt, "r", encoding="utf-8") as f:
            for line in f:
                # multi_cell tự động xuống dòng nếu văn bản quá dài
                pdf.multi_cell(0, 5, txt=line)

        pdf.output(output_pdf)
        print(f"Thành công! File PDF đã được lưu tại: {output_pdf}")
        
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file đầu vào.")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

input_file = "./BA_05.2021.DS-ST.pdf_0.jpg.txt"
output_file = "tai_lieu_output(2).pdf"
txt_to_pdf(input_file, output_file)
import logging
import os
from fpdf import FPDF

FONT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "fonts", "TimesNewRoman.ttf"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "result_pdf"))


class TxtToPdfService:
    def __init__(self):
        if not os.path.isfile(FONT_PATH):
            raise FileNotFoundError(f"Font not found: {FONT_PATH}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logging.info(f"[TxtToPdfService] Ready. Font: Times New Roman")

    def convert(self, md_path: str) -> str:
        if not os.path.isfile(md_path):
            raise FileNotFoundError(f"Input file not found: {md_path}")

        file_name = os.path.splitext(os.path.basename(md_path))[0]
        pdf_path = os.path.join(OUTPUT_DIR, f"{file_name}.pdf")

        with open(md_path, encoding="utf-8") as f:
            lines = f.readlines()

        pdf = FPDF()
        pdf.add_font("TNR", "", FONT_PATH)
        pdf.set_margins(left=15, top=15, right=15)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("TNR", size=13)

        for line in lines:
            line = line.rstrip("\n")
            if line.startswith("---"):
                pdf.add_page()
                pdf.set_font("TNR", size=13)
            else:
                pdf.multi_cell(0, 7, text=line)
                pdf.set_x(pdf.l_margin)

        pdf.output(pdf_path)
        logging.info(f"[TxtToPdfService] Saved → {pdf_path}")
        return pdf_path

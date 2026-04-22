import logging
import os
from pdf2image import convert_from_path
from PIL import Image

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "result_tiff"))

logger = logging.getLogger(__name__)


class PdfToTiffService:
    """Convert PDF sang TIFF multi-page."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def convert(self, pdf_path: str, output_path: str = None) -> str:
        """
        Convert PDF → TIFF multi-page.

        Args:
            pdf_path:    Đường dẫn file PDF đầu vào.
            output_path: Đường dẫn file TIFF output. 

        Returns:
            Đường dẫn file TIFF đã tạo.
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if output_path is None:
            pdf_name    = os.path.splitext(os.path.basename(pdf_path))[0]
            output_path = os.path.join(OUTPUT_DIR, f"{pdf_name}.tiff")

        logger.info(f"[PdfToTiff] Converting: {pdf_path} @ {self.dpi} DPI")

        pages = convert_from_path(pdf_path, dpi=self.dpi)

        if not pages:
            raise ValueError(f"No pages found in: {pdf_path}")

        if len(pages) == 1:
            pages[0].save(output_path, format="TIFF", compression="tiff_lzw")
        else:
            pages[0].save(
                output_path,
                format="TIFF",
                compression="tiff_lzw",
                save_all=True,
                append_images=pages[1:],
            )

        logger.info(f"[PdfToTiff] Saved → {output_path} ({len(pages)} pages)")
        return output_path

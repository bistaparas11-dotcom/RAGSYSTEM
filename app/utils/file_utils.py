import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error
from io import BytesIO
from app.config import settings


def process_pdf(pdf_path, output_md_path, prompt="ocr", dpi=300):
    """
    Processes a PDF file by converting pages to images and calling the OCR API.
    """
    try:
        import fitz
    except ImportError:
        print("Error: PyMuPDF (fitz) is not installed.")
        sys.exit(1)

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        sys.exit(1)

    print(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    print(f"Total pages: {num_pages}, Using DPI: {dpi}, Prompt: '{prompt}'")

    full_markdown = []

    # Header for the markdown file
    full_markdown.append(f"# OCR Result for: {os.path.basename(pdf_path)}")
    full_markdown.append("\n---\n")

    for page_num in range(num_pages):
        print(f"Processing page {page_num + 1}/{num_pages}...")

        # Render page to an image
        page = doc.load_page(page_num)
        zoom = dpi / 72  # 72 is the default DPI in PyMuPDF
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert pixmap to image bytes
        img_bytes = pix.tobytes("png")
        image_b64 = base64.b64encode(img_bytes).decode('utf-8')

        payload = {
            "image_base64": image_b64,
            "prompt": prompt,
            "use_layout_detection": True,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "layout_merge_bboxes_mode": "small"
        }

        try:
            json_data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                settings.OCR_API_URL,
                data=json_data,
                headers={'Content-Type': 'application/json'}
            )

            start_time = time.time()
            with urllib.request.urlopen(req) as response:
                resp_body = response.read()
                latency = (time.time() - start_time) * 1000
                result = json.loads(resp_body)

                if result.get("success"):
                    page_md = result.get("markdown", "").strip()
                    if page_md:
                        full_markdown.append(f"## Page {page_num + 1}\n\n")
                        full_markdown.append(page_md)
                        full_markdown.append("\n\n---\n")
                    else:
                        full_markdown.append(
                            f"## Page {page_num+1}\n\n_No text detected on this page._\n\n---\n"
                        )
                else:
                    print(
                        f"Page {page_num+1} Failed: {result.get('error')}"
                    )
                    full_markdown.append(
                        f"## Page {page_num+1}\n\n**Error processing page:** {result.get('error')}\n\n---\n"
                    )

        except urllib.error.HTTPError as e:
            err_msg = e.read().decode('utf-8') if e.code != 503 else "Service unavailable"
            print(f"Page {page_num + 1} HTTP Error {e.code}: {err_msg}")
            full_markdown.append(
                f"## Page {page_num + 1}\n\n**HTTP Error {e.code}**\n\n---\n"
            )
        except Exception as e:
            print(f"Page {page_num + 1} Unexpected error: {e}")
            full_markdown.append(
                f"## Page {page_num + 1}\n\n**Unexpected error:** {str(e)}\n\n---\n")

    final_result = "\n".join(full_markdown)

    if output_md_path:
        with open(output_md_path, "w") as f:
            f.writelines(final_result)
        print(f"\nFinal result saved to: {output_md_path}")
    else:
        # Default output name if none provided
        out_name = os.path.splitext(os.path.basename(pdf_path))[
            0] + "_ocr_result.md"
        with open(out_name, "w") as f:
            f.writelines(final_result)
        print(f"\nFinal result saved to: {out_name}")

    return final_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a PDF file with PaddleOCR-VL API and get markdown output.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--output",
        "-o",
        help="Optional output markdown file path")
    parser.add_argument("--prompt", "-p", default="ocr",
                        help="OCR prompt (option: 'ocr' or 'table')")
    parser.add_argument(
        "--dpi",
        "-d",
        type=int,
        default=300,
        help="DPI for PDF conversion (default: 300)")
    args = parser.parse_args()

    process_pdf(args.pdf_path, args.output, args.prompt, args.dpi)

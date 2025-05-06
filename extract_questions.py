import os
import argparse
import json
from PyPDF2 import PdfReader
from tqdm import tqdm

def extract_qa_from_text(text):
    """Trích xuất các cặp câu hỏi và trả lời từ nội dung bài báo."""
    qas = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if any(keyword in line.lower() for keyword in ["objective", "aim", "goal", "purpose"]):
            question = f"What is the main objective of the study in: {line}"
            answer = lines[i + 1].strip() if i + 1 < len(lines) else "Not found"
            qas.append({"question": question, "answer": answer})
        elif any(phrase in line.lower() for phrase in ["the results show", "we found that", "this indicates"]):
            question = f"What are the main findings in: {line}"
            answer = lines[i + 1].strip() if i + 1 < len(lines) else "Not found"
            qas.append({"question": question, "answer": answer})
    return qas

def extract_from_pdf(pdf_path):
    """Đọc nội dung PDF và trích xuất câu hỏi"""
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    return extract_qa_from_text(text)

def main(input_dir, output_file):
    """Duyệt qua thư mục chứa PDF và trích xuất câu hỏi"""
    if not os.path.exists(input_dir):
        print(f"⚠️ Lỗi: Thư mục {input_dir} không tồn tại!")
        return

    qa_pairs = []
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️ Không có file PDF nào trong thư mục.")
        return

    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        filepath = os.path.join(input_dir, filename)
        try:
            print(f"📖 Đang xử lý: {filename}")
            qa_pairs.extend(extract_from_pdf(filepath))
        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý {filename}: {e}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"✅ Đã lưu {len(qa_pairs)} cặp Q&A vào {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất câu hỏi từ PDF về vật liệu bán dẫn.")
    parser.add_argument("--input_dir", required=True, help="Thư mục chứa các file PDF")
    parser.add_argument("--output_file", required=True, help="File JSON để lưu kết quả")
    args = parser.parse_args()

    main(args.input_dir, args.output_file)
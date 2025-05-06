import json
import pandas as pd
import re
from fuzzywuzzy import fuzz  # Dùng Levenshtein distance để so sánh

# Đọc kết quả từ file JSON
def load_results(file_path):
    """Đọc dữ liệu câu hỏi từ file JSON"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Chuẩn hóa câu (chuyển thành chữ thường và loại bỏ dấu câu, khoảng trắng thừa)
def clean_text(text):
    text = text.lower()  # Chuyển về chữ thường
    text = re.sub(r'\s+', ' ', text)  # Loại bỏ khoảng trắng dư thừa
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu
    return text.strip()

# Phân tích kết quả: tính độ chính xác (so sánh mềm)
def analyze_results(results):
    """Phân tích kết quả đánh giá"""
    correct_standard = 0
    correct_cot = 0
    total = len(results)

    for item in results:
        gt_answer = clean_text(item["ground_truth"])
        standard_answer = clean_text(item["standard_answer"])
        cot_answer = clean_text(item["cot_answer"])

        # Sử dụng Levenshtein distance để tính độ tương đồng
        standard_similarity = fuzz.ratio(gt_answer, standard_answer)
        cot_similarity = fuzz.ratio(gt_answer, cot_answer)

        # So sánh và đánh giá
        if standard_similarity > 80:  # Nếu tương đồng > 80%
            correct_standard += 1
        if cot_similarity > 80:  # Nếu tương đồng > 80%
            correct_cot += 1

        # Hiển thị ví dụ khi độ tương đồng thấp
        if standard_similarity < 80 or cot_similarity < 80:
            print(f"⚠️ Câu hỏi: {item['question']}")
            print(f"✅ Ground Truth: {item['ground_truth']}")
            print(f"🧠 Standard Answer: {item['standard_answer']} (Tương đồng: {standard_similarity}%)")
            print(f"🧠 CoT Answer: {item['cot_answer']} (Tương đồng: {cot_similarity}%)")
            print("-" * 100)

    accuracy_standard = correct_standard / total * 100
    accuracy_cot = correct_cot / total * 100

    print(f"✅ Độ chính xác của Standard Prompting: {accuracy_standard:.2f}%")
    print(f"✅ Độ chính xác của Chain of Thought (CoT): {accuracy_cot:.2f}%")

    return accuracy_standard, accuracy_cot

# Lưu kết quả phân tích
def save_analysis(accuracy_standard, accuracy_cot, output_file):
    """Lưu kết quả phân tích vào file CSV"""
    data = {
        "Method": ["Standard Prompting", "Chain of Thought (CoT)"],
        "Accuracy (%)": [accuracy_standard, accuracy_cot]
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"✅ Đã lưu kết quả phân tích vào {output_file}")

if __name__ == "__main__":
    input_file = "results/evaluated_results.json"
    output_file = "results/analysis_results.csv"

    print("🔍 Đang phân tích kết quả đánh giá...")
    results = load_results(input_file)
    accuracy_standard, accuracy_cot = analyze_results(results)
    save_analysis(accuracy_standard, accuracy_cot, output_file)
    print("✅ Hoàn thành phân tích!")

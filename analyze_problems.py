import json
import pandas as pd
from collections import Counter
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

STOPWORDS = set(stopwords.words('english'))

# Đọc dữ liệu từ file JSON đầu vào
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Kiểm tra dữ liệu có bị trùng lặp không
def check_duplicates(data):
    questions = [item["question"] for item in data]
    duplicate_counts = Counter(questions)
    duplicates = {q: c for q, c in duplicate_counts.items() if c > 1}
    return duplicates

# Kiểm tra câu hỏi/lời giải bị thiếu hoặc quá ngắn
def check_invalid_entries(data):
    invalid_entries = [item for item in data if len(item["question"]) < 10 or len(item["answer"]) < 5]
    return invalid_entries


# Phân tích số lượng câu hỏi và chủ đề phổ biến
def extract_keywords(text):
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in STOPWORDS]
    return words

def analyze_data(data):
    print(f"Tổng số câu hỏi: {len(data)}")
    
    # Lấy các từ khóa chính trong câu hỏi
    all_keywords = []
    for item in data:
        all_keywords.extend(extract_keywords(item["question"]))
    
    common_topics = Counter(all_keywords).most_common(10)
    
    print("Chủ đề phổ biến:")
    for topic, count in common_topics:
        print(f"- {topic}: {count} lần")

# Lọc dữ liệu để loại bỏ các câu không hợp lệ
def clean_data(data):
    clean_data = [item for item in data if len(item["question"]) >= 10 and len(item["answer"]) >= 5]
    return clean_data

# Lưu dữ liệu sạch vào file JSON
def save_clean_data(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Đã lưu dữ liệu sạch vào {output_file}")

if __name__ == "__main__":
    input_file = "db/questions/cot_questions.json"
    output_file = "db/questions/cot_questions_clean.json"
    
    print("🔍 Đang phân tích dữ liệu...")
    data = load_data(input_file)
    
    analyze_data(data)
    
    duplicates = check_duplicates(data)
    if duplicates:
        print(f"⚠️ Có {len(duplicates)} câu hỏi bị trùng lặp")
    
    invalid_entries = check_invalid_entries(data)
    if invalid_entries:
        print(f"⚠️ Có {len(invalid_entries)} câu hỏi hoặc câu trả lời không hợp lệ")
    
    print("🛠 Đang làm sạch dữ liệu...")
    cleaned_data = clean_data(data)
    
    save_clean_data(cleaned_data, output_file)
    print("✅ Hoàn thành!")

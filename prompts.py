"""
DeepSeek-r1-distill-llama-70b and Gemini 1.5 Flash, incorporating various prompting techniques.
Prompt templates are organized by category and key for easy access.
"""

# --- Prompt Prefixes (Có thể kết hợp) ---
COT_PREFIX = "Hãy suy nghĩ từng bước để phân tích vật liệu bán dẫn một cách toàn diện."
ROLE_PREFIXES = {
    "expert_researcher": "Bạn là một nhà nghiên cứu chuyên sâu về vật liệu bán dẫn với kiến thức sâu rộng về đặc tính vật lý, ứng dụng, và xu hướng công nghệ.",
    "helpful_assistant": "Bạn là một trợ lý AI hữu ích và nhiệt tình.",
    "concise_summarizer": "Bạn là một AI chuyên tóm tắt thông tin một cách ngắn gọn và chính xác.",
}

# --- Định nghĩa Cấu Trúc Prompt ---
"""
Dưới đây là 4 loại cấu trúc prompt chính được hỗ trợ:

1. BASIC_PROMPT_STRUCTURE:
   - Chỉ có template cơ bản, không có thêm bất kỳ hướng dẫn nào
   - Thích hợp cho những câu hỏi đơn giản
   
2. COT_PROMPT_STRUCTURE:
   - Thêm hướng dẫn "Suy nghĩ từng bước" trước template cơ bản
   - Giúp AI làm việc theo quy trình có cấu trúc và cẩn thận hơn
   
3. ROLE_PROMPT_STRUCTURE:
   - Thêm vai trò chuyên gia bán dẫn trước template cơ bản
   - Giúp AI trả lời với kiến thức chuyên sâu và chuyên nghiệp hơn
   
4. ROLE_COT_PROMPT_STRUCTURE:
   - Kết hợp cả vai trò chuyên gia và suy nghĩ từng bước
   - Là loại prompt toàn diện nhất, phù hợp với vấn đề phức tạp
"""

# Cấu trúc cụ thể của 4 loại prompt
PROMPT_STRUCTURES = {
    "basic": {
        "description": "Prompt cơ bản không có kỹ thuật bổ sung nào",
        "components": ["template"],
        "prefix": []
    },
    
    "cot": {
        "description": "Prompt có hướng dẫn suy nghĩ từng bước (Chain of Thought)",
        "components": ["cot_prefix", "template"],
        "prefix": [COT_PREFIX]
    },
    
    "role": {
        "description": "Prompt có xác định vai trò chuyên gia",
        "components": ["role_prefix", "template"],
        "prefix": [ROLE_PREFIXES["expert_researcher"]]
    },
    
    "role_cot": {
        "description": "Prompt kết hợp vai trò chuyên gia và suy nghĩ từng bước",
        "components": ["role_prefix", "cot_prefix", "template"],
        "prefix": [ROLE_PREFIXES["expert_researcher"], COT_PREFIX]
    }
}

# --- Base Templates ---
# Các template cơ sở cho phân tích vật liệu - không bao gồm các prefix kỹ thuật

# Template phân tích văn bản thông thường
TEXT_ANALYSIS_TEMPLATE = """Phân tích chi tiết vật liệu bán dẫn {material_name} với các đặc tính được cung cấp.

**Dữ liệu gốc:**
{details_string}

**Thông tin tham khảo từ tài liệu (nếu có):**
{retrieved_context}

**PHÂN TÍCH CHI TIẾT**

## 1. DỮ LIỆU VẬT LIỆU ĐẦU VÀO
- Tổng hợp các thông tin cơ bản về vật liệu {material_name}
- Phân tích cấu trúc tinh thể ({crystal_structure}) và tỷ lệ thành phần
- Xem xét các thuộc tính cung cấp: bandgap ({bandgap_energy} eV), độ dẫn điện, cấu trúc bề mặt

## 2. PHÂN TÍCH THUỘC TÍNH VẬT LÝ
- Đánh giá chi tiết cấu trúc tinh thể và xác định nhóm đối xứng
- Phân loại vật liệu dựa trên bandgap (kim loại/bán dẫn/cách điện)
- Phân tích độ dẫn điện và dự đoán cơ chế dẫn điện
- Đánh giá các thuộc tính nhiệt, khả năng bền nhiệt

## 3. ĐÁNH GIÁ TÍNH ỨNG DỤNG
- Xác định ứng dụng tiềm năng dựa trên bandgap và các thuộc tính vật lý
- Đánh giá mức độ phù hợp với ứng dụng được đề xuất: {target_application_potential}
- Dự đoán hiệu suất của vật liệu trong các ứng dụng cụ thể
- Phân tích ưu và nhược điểm trong từng lĩnh vực ứng dụng

## 4. DỰ ĐOÁN THUỘC TÍNH NÂNG CAO
- Dự đoán độ ổn định nhiệt và cơ học của vật liệu
- Đề xuất chiến lược pha tạp phù hợp (n-type/p-type), nồng độ và tác động
- Ước tính chiết suất (refractive index) dựa trên cấu trúc và bandgap
- Dự đoán ảnh hưởng của nhiệt độ đến độ dẫn điện và các thuộc tính khác

## 5. SO SÁNH VỚI VẬT LIỆU CHUẨN
- So sánh {material_name} với các vật liệu bán dẫn tương tự và tiêu chuẩn (Si, Ge, GaAs...)
- Phân tích ưu và nhược điểm so với các vật liệu hiện có
- Đánh giá vị thế cạnh tranh và tính độc đáo

## 6. KẾT QUẢ PHÂN TÍCH TOÀN DIỆN
- Tổng hợp điểm mạnh và điểm yếu chính của vật liệu
- Đánh giá tiềm năng tổng thể (thang điểm 1-10)
- Đề xuất ứng dụng tối ưu nhất 
- Đề xuất hướng nghiên cứu và cải tiến tiếp theo
"""

# Template phân tích có cấu trúc JSON
JSON_ANALYSIS_TEMPLATE = """Phân tích chi tiết vật liệu bán dẫn {material_name} và trả về kết quả được cấu trúc theo định dạng JSON.

**Dữ liệu gốc:**
{details_string}

**Thông tin tham khảo từ tài liệu (nếu có):**
{retrieved_context}

Hãy phân tích vật liệu theo 6 bước chính và trả về kết quả theo định dạng JSON với cấu trúc sau:

```json
{{
  "reasoning_steps": [
    {{
      "step": 1,
      "title": "Dữ liệu vật liệu đầu vào",
      "content": "Nội dung phân tích chi tiết...",
      "key_findings": ["Phát hiện chính 1", "Phát hiện chính 2", ...]
    }},
    {{
      "step": 2,
      "title": "Phân tích thuộc tính vật lý",
      "content": "Nội dung phân tích chi tiết...",
      "key_findings": ["Phát hiện chính 1", "Phát hiện chính 2", ...]
    }},
    ...
  ],
  "final_analysis": {{
    "overall_quality": 7.5,
    "bandgap_classification": "Semiconductor",
    "key_strengths": ["Điểm mạnh 1", "Điểm mạnh 2",...],
    "key_weaknesses": ["Điểm yếu 1", "Điểm yếu 2", ...]
  }},
  "recommendations": {{
    "applications": ["Ứng dụng 1", "Ứng dụng 2", ...],
    "improvements": [
      {{ "aspect": "Độ bền nhiệt", "recommendation": "Cải thiện...", "priority": "High" }},
      ...
    ]
  }},
  "basic_properties": {{
    "bandgap_type": "Trực tiếp",
    "crystal_system": "Lập phương",
    "carrier_type": "n-type",
    "thermal_stability": "Cao"
  }},
  "predictions": {{
    "bandgap_prediction": {{ "value": "Semiconductor", "confidence": 0.95 }},
    "conductivity_prediction": {{ "value": "Medium", "confidence": 0.8 }},
    "stability_prediction": {{ "value": "High", "confidence": 0.75 }}
  }}
}}
```

## Phân tích theo 6 bước:

### BƯỚC 1: DỮ LIỆU VẬT LIỆU ĐẦU VÀO
- Tổng hợp các thông tin cơ bản về {material_name}
- Phân tích cấu trúc tinh thể ({crystal_structure}) và thành phần
- Tổng hợp tất cả các thuộc tính được cung cấp

### BƯỚC 2: PHÂN TÍCH THUỘC TÍNH VẬT LÝ
- Phân loại bandgap: nhỏ/trung bình/lớn, trực tiếp/gián tiếp
- Phân tích độ dẫn điện và cơ chế dẫn điện
- Dự đoán các tính chất quang học và điện tử

### BƯỚC 3: ĐÁNH GIÁ TÍNH ỨNG DỤNG
- Xác định ứng dụng tiềm năng dựa trên bandgap và các thuộc tính
- Đánh giá mức độ phù hợp với {target_application_potential}
- Xếp hạng mức độ phù hợp với các lĩnh vực ứng dụng

### BƯỚC 4: DỰ ĐOÁN THUỘC TÍNH NÂNG CAO
- Dự đoán độ ổn định nhiệt và cơ học
- Đề xuất chiến lược pha tạp (n-type/p-type)
- Ước tính chiết suất và các thuộc tính quang học
- Phân tích ảnh hưởng của nhiệt độ

### BƯỚC 5: SO SÁNH VỚI VẬT LIỆU CHUẨN
- So sánh với các vật liệu tiêu chuẩn (Si, Ge, GaAs...)
- Phân tích điểm mạnh và điểm yếu so với vật liệu hiện có
- Đánh giá khả năng cạnh tranh

### BƯỚC 6: KẾT QUẢ PHÂN TÍCH TOÀN DIỆN
- Đánh giá chất lượng tổng thể (thang điểm 1-10)
- Liệt kê điểm mạnh và điểm yếu chính
- Đề xuất các ứng dụng tối ưu
- Đề xuất hướng nghiên cứu và cải tiến
"""

# --- Other Prompts (Simplified) ---
GENERAL_PROMPTS = {
    "introduction": "Viết lời giới thiệu về tác động của AI đối với nghiên cứu bán dẫn.",
    "summary": "Tóm tắt văn bản sau:\n{input_text}",
    "question": "Trả lời câu hỏi sau:\n{question}",
}

TASK_PROMPTS = {
    "generate_ideas": "Đề xuất một danh sách các ý tưởng sáng tạo để cải thiện quy trình sản xuất bán dẫn.",
    "explain_concept": "Giải thích khái niệm về '{concept}' bằng thuật ngữ đơn giản.",
    "compare_technologies": "So sánh ưu và nhược điểm của công nghệ '{technology1}' và '{technology2}'.",
    "analyze_material": """Phân tích chi tiết vật liệu bán dẫn {material_name} với các đặc tính được cung cấp.

**Dữ liệu gốc:**
{details_string}

**Thông tin tham khảo từ tài liệu (nếu có):**
{retrieved_context}

**PHÂN TÍCH CHI TIẾT**

## 1. DỮ LIỆU VẬT LIỆU ĐẦU VÀO
- Tổng hợp các thông tin cơ bản về vật liệu {material_name}
- Phân tích cấu trúc tinh thể ({crystal_structure}) và tỷ lệ thành phần
- Xem xét các thuộc tính cung cấp: bandgap ({bandgap_energy} eV), độ dẫn điện, cấu trúc bề mặt

## 2. PHÂN TÍCH THUỘC TÍNH VẬT LÝ
- Đánh giá chi tiết cấu trúc tinh thể và xác định nhóm đối xứng
- Phân loại vật liệu dựa trên bandgap (kim loại/bán dẫn/cách điện)
- Phân tích độ dẫn điện và dự đoán cơ chế dẫn điện
- Đánh giá các thuộc tính nhiệt, khả năng bền nhiệt

## 3. ĐÁNH GIÁ TÍNH ỨNG DỤNG
- Xác định ứng dụng tiềm năng dựa trên bandgap và các thuộc tính vật lý
- Đánh giá mức độ phù hợp với ứng dụng được đề xuất: {target_application_potential}
- Dự đoán hiệu suất của vật liệu trong các ứng dụng cụ thể
- Phân tích ưu và nhược điểm trong từng lĩnh vực ứng dụng

## 4. DỰ ĐOÁN THUỘC TÍNH NÂNG CAO
- Dự đoán độ ổn định nhiệt và cơ học của vật liệu
- Đề xuất chiến lược pha tạp phù hợp (n-type/p-type), nồng độ và tác động
- Ước tính chiết suất (refractive index) dựa trên cấu trúc và bandgap
- Dự đoán ảnh hưởng của nhiệt độ đến độ dẫn điện và các thuộc tính khác

## 5. SO SÁNH VỚI VẬT LIỆU CHUẨN
- So sánh {material_name} với các vật liệu bán dẫn tương tự và tiêu chuẩn (Si, Ge, GaAs...)
- Phân tích ưu và nhược điểm so với các vật liệu hiện có
- Đánh giá vị thế cạnh tranh và tính độc đáo

## 6. KẾT QUẢ PHÂN TÍCH TOÀN DIỆN
- Tổng hợp điểm mạnh và điểm yếu chính của vật liệu
- Đánh giá tiềm năng tổng thể (thang điểm 1-10)
- Đề xuất ứng dụng tối ưu nhất 
- Đề xuất hướng nghiên cứu và cải tiến tiếp theo
""",
    "analyze_material_cot": """Phân tích chi tiết vật liệu bán dẫn {material_name} và trả về kết quả được cấu trúc theo định dạng JSON.

**Dữ liệu gốc:**
{details_string}

**Thông tin tham khảo từ tài liệu (nếu có):**
{retrieved_context}

Hãy phân tích vật liệu theo 6 bước chính và trả về kết quả theo định dạng JSON với cấu trúc sau:

```json
{{
  "reasoning_steps": [
    {{
      "step": 1,
      "title": "Dữ liệu vật liệu đầu vào",
      "content": "Nội dung phân tích chi tiết...",
      "key_findings": ["Phát hiện chính 1", "Phát hiện chính 2", ...]
    }},
    {{
      "step": 2,
      "title": "Phân tích thuộc tính vật lý",
      "content": "Nội dung phân tích chi tiết...",
      "key_findings": ["Phát hiện chính 1", "Phát hiện chính 2", ...]
    }},
    ...
  ],
  "final_analysis": {{
    "overall_quality": 7.5,
    "bandgap_classification": "Semiconductor",
    "key_strengths": ["Điểm mạnh 1", "Điểm mạnh 2",...],
    "key_weaknesses": ["Điểm yếu 1", "Điểm yếu 2", ...]
  }},
  "recommendations": {{
    "applications": ["Ứng dụng 1", "Ứng dụng 2", ...],
    "improvements": [
      {{ "aspect": "Độ bền nhiệt", "recommendation": "Cải thiện...", "priority": "High" }},
      ...
    ]
  }},
  "basic_properties": {{
    "bandgap_type": "Trực tiếp",
    "crystal_system": "Lập phương",
    "carrier_type": "n-type",
    "thermal_stability": "Cao"
  }},
  "predictions": {{
    "bandgap_prediction": {{ "value": "Semiconductor", "confidence": 0.95 }},
    "conductivity_prediction": {{ "value": "Medium", "confidence": 0.8 }},
    "stability_prediction": {{ "value": "High", "confidence": 0.75 }}
  }}
}}
```

## Phân tích theo 6 bước:

### BƯỚC 1: DỮ LIỆU VẬT LIỆU ĐẦU VÀO
- Tổng hợp các thông tin cơ bản về {material_name}
- Phân tích cấu trúc tinh thể ({crystal_structure}) và thành phần
- Tổng hợp tất cả các thuộc tính được cung cấp

### BƯỚC 2: PHÂN TÍCH THUỘC TÍNH VẬT LÝ
- Phân loại bandgap: nhỏ/trung bình/lớn, trực tiếp/gián tiếp
- Phân tích độ dẫn điện và cơ chế dẫn điện
- Dự đoán các tính chất quang học và điện tử

### BƯỚC 3: ĐÁNH GIÁ TÍNH ỨNG DỤNG
- Xác định ứng dụng tiềm năng dựa trên bandgap và các thuộc tính
- Đánh giá mức độ phù hợp với {target_application_potential}
- Xếp hạng mức độ phù hợp với các lĩnh vực ứng dụng

### BƯỚC 4: DỰ ĐOÁN THUỘC TÍNH NÂNG CAO
- Dự đoán độ ổn định nhiệt và cơ học
- Đề xuất chiến lược pha tạp (n-type/p-type)
- Ước tính chiết suất và các thuộc tính quang học
- Phân tích ảnh hưởng của nhiệt độ

### BƯỚC 5: SO SÁNH VỚI VẬT LIỆU CHUẨN
- So sánh với các vật liệu tiêu chuẩn (Si, Ge, GaAs...)
- Phân tích điểm mạnh và điểm yếu so với vật liệu hiện có
- Đánh giá khả năng cạnh tranh

### BƯỚC 6: KẾT QUẢ PHÂN TÍCH TOÀN DIỆN
- Đánh giá chất lượng tổng thể (thang điểm 1-10)
- Liệt kê điểm mạnh và điểm yếu chính
- Đề xuất các ứng dụng tối ưu
- Đề xuất hướng nghiên cứu và cải tiến
""",
}

RESEARCH_PROMPTS = {
    "literature_review": "Cung cấp tổng quan tài liệu về chủ đề '{topic}'.",
    "future_trends": "Xu hướng tương lai trong lĩnh vực '{field}' là gì?",
    "research_question": "Những câu hỏi nghiên cứu chính trong lĩnh vực '{field}' là gì?",
}

STRUCTURED_OUTPUT_PROMPTS = {
    "ideas_json": "Đề xuất danh sách các ý tưởng sáng tạo để cải thiện quy trình sản xuất bán dẫn. Trả lời dưới dạng một danh sách JSON, mỗi đối tượng có key 'idea' và 'description'.",
    "comparison_table": "So sánh ưu và nhược điểm của công nghệ '{technology1}' và '{technology2}'. Trình bày kết quả dưới dạng bảng Markdown với cột 'Ưu điểm' và 'Nhược điểm'.",
    "key_points_bullets": "Liệt kê những điểm chính từ văn bản sau dưới dạng gạch đầu dòng:\n{input_text}",
}

# --- Few-Shot Examples (Ví dụ cách tích hợp) ---
# Kỹ thuật này thường hiệu quả hơn khi ví dụ được xây dựng động dựa trên ngữ cảnh,
# nhưng đây là cách lưu trữ một prompt few-shot cố định.
FEW_SHOT_PROMPTS = {
    "classify_material_property": """Phân loại thuộc tính chính được mô tả trong đoạn văn sau.
Ví dụ 1:
Đoạn văn: "Điện trở suất của màng mỏng giảm đáng kể khi nhiệt độ ủ tăng lên."
Phân loại: Thuộc tính điện (Electrical Property)

Ví dụ 2:
Đoạn văn: "Phân tích XRD cho thấy cấu trúc tinh thể lập phương tâm diện."
Phân loại: Thuộc tính cấu trúc (Structural Property)

Đoạn văn: "{input_text}"
Phân loại:""",
}

# --- Function to generate prompts ---
def get_prompt(prompt_type="basic", output_format="text", **kwargs):
    """
    Tạo prompt phân tích vật liệu bán dẫn theo loại và định dạng đầu ra.
    
    Args:
        prompt_type: Kiểu prompt ("basic", "cot", "role", "role_cot")
        output_format: Định dạng đầu ra ("text" hoặc "json")
        **kwargs: Các tham số khác để định dạng prompt
    
    Returns:
        Chuỗi prompt đã được định dạng
    """
    # Kiểm tra loại prompt có hợp lệ không
    if prompt_type not in PROMPT_STRUCTURES:
        raise ValueError(f"Loại prompt không hợp lệ: {prompt_type}. Các loại hợp lệ: {list(PROMPT_STRUCTURES.keys())}")
    
    # Kiểm tra định dạng đầu ra có hợp lệ không
    if output_format not in ["text", "json"]:
        raise ValueError(f"Định dạng đầu ra không hợp lệ: {output_format}. Định dạng hợp lệ: 'text' hoặc 'json'")
    
    # Kiểm tra các tham số bắt buộc
    required_params = [
        'material_name', 
        'details_string', 
        'retrieved_context', 
        'crystal_structure', 
        'bandgap_energy', 
        'target_application_potential'
    ]
    
    # Báo lỗi nếu thiếu tham số
    for param in required_params:
        if param not in kwargs:
            raise ValueError(f"Thiếu tham số: {param}")
    
    # Chọn template dựa trên định dạng đầu ra
    template = TEXT_ANALYSIS_TEMPLATE if output_format == "text" else JSON_ANALYSIS_TEMPLATE
    
    # Lấy cấu trúc prompt theo loại đã chọn
    structure = PROMPT_STRUCTURES[prompt_type]
    
    # Xây dựng prompt từ các thành phần theo cấu trúc
    prompt_parts = []
    
    # Thêm các prefix theo thứ tự trong cấu trúc
    prompt_parts.extend(structure["prefix"])
    
    # Định dạng template với các tham số
    try:
        formatted_template = template.format(**kwargs)
        prompt_parts.append(formatted_template)
    except KeyError as e:
        raise ValueError(f"Lỗi khi định dạng prompt: thiếu tham số {str(e)}")
    
    # Kết hợp các phần prompt với dấu xuống dòng kép
    return "\n\n".join(prompt_parts)

# --- For backwards compatibility ---
def legacy_get_prompt(category, key, add_cot=False, role=None, **kwargs):
    """
    Hàm cũ để tương thích ngược với mã hiện có.
    
    Ánh xạ sang hàm get_prompt mới, sử dụng các cấu hình đã xác định.
    """
    if category == "task" and key == "analyze_material":
        # Xác định loại prompt
        prompt_type = "basic"
        if add_cot and role:
            prompt_type = "role_cot"
        elif add_cot:
            prompt_type = "cot"
        elif role:
            prompt_type = "role"
            
        # Xác định định dạng đầu ra
        output_format = "text"
        
        # Gọi hàm mới
        return get_prompt(prompt_type, output_format, **kwargs)
    
    elif category == "task" and key == "analyze_material_cot":
        # Xác định loại prompt
        prompt_type = "basic"
        if add_cot and role:
            prompt_type = "role_cot"
        elif add_cot:
            prompt_type = "cot"
        elif role:
            prompt_type = "role"
            
        # Sử dụng định dạng JSON
        output_format = "json"
        
        # Gọi hàm mới
        return get_prompt(prompt_type, output_format, **kwargs)
    
    else:
        # Thông báo lỗi nếu không hỗ trợ
        raise ValueError(f"Không hỗ trợ category={category}, key={key}")

# Để tương thích ngược với mã hiện tại
get_prompt_legacy = legacy_get_prompt

# Phơi bày hàm get_prompt mới và các hàm tương thích ngược
__all__ = ['get_prompt', 'get_prompt_legacy', 'legacy_get_prompt']

# --- Example Usage ---
if __name__ == "__main__":
    print("\n===== ĐỊNH NGHĨA CẤU TRÚC PROMPT PHÂN TÍCH VẬT LIỆU =====")
    
    # In cấu trúc các loại prompt
    print("\n--- Cấu trúc 4 Loại Prompt ---")
    for prompt_type, structure in PROMPT_STRUCTURES.items():
        print(f"\n{prompt_type.upper()}:")
        print(f"  • Mô tả: {structure['description']}")
        print(f"  • Thành phần: {' + '.join(structure['components'])}")
        print(f"  • Số lượng prefix: {len(structure['prefix'])}")
    
    # Tạo dữ liệu mẫu
    sample_data = {
        "material_name": "Silicon",
        "details_string": "- Material Name: Silicon\n- Crystal Structure: Cubic Diamond\n- Bandgap: 1.12 eV\n- Conductivity: 1000 S/cm",
        "retrieved_context": "Silicon is a classic semiconductor material used in various electronic applications.",
        "crystal_structure": "Cubic Diamond",
        "bandgap_energy": 1.12,
        "target_application_potential": "Solar Cells, Transistors"
    }
    
    # Tạo các loại prompt
    print("\n\n===== MINH HỌA CÁC LOẠI PROMPT =====")
    
    prompt_examples = {}
    
    # Tạo ví dụ cho mỗi loại prompt
    for prompt_type in PROMPT_STRUCTURES.keys():
        # Text format (văn bản thông thường)
        prompt_text = get_prompt(prompt_type, "text", **sample_data)
        prompt_examples[f"{prompt_type}_text"] = prompt_text
        
        # JSON format (cấu trúc)
        prompt_json = get_prompt(prompt_type, "json", **sample_data)
        prompt_examples[f"{prompt_type}_json"] = prompt_json
    
    # In ví dụ minh họa so sánh (chỉ hiển thị 300 ký tự đầu tiên + độ dài)
    print("\n--- So sánh Các Loại Prompt (Text Format) ---")
    for prompt_type in PROMPT_STRUCTURES.keys():
        prompt_key = f"{prompt_type}_text"
        prompt = prompt_examples[prompt_key]
        print(f"\n{prompt_type.upper()} (độ dài: {len(prompt)} ký tự):")
        print(f"{prompt[:300]}...\n[còn tiếp]")
    
    # So sánh các Prefixes
    print("\n\n--- Minh Họa Prefixes ---")
    for prompt_type, structure in PROMPT_STRUCTURES.items():
        print(f"\nPrefixes cho {prompt_type.upper()}:")
        if structure["prefix"]:
            for i, prefix in enumerate(structure["prefix"], 1):
                print(f"{i}. {prefix}")
        else:
            print("  Không có prefix")
            
    # In minh họa đầy đủ cho ROLE_COT (loại đầy đủ nhất)
    print("\n\n===== MINH HỌA ĐẦY ĐỦ CHO ROLE_COT (TEXT FORMAT) =====")
    full_example = prompt_examples["role_cot_text"]
    print(full_example)
    
    # Kiểm tra tương thích ngược
    print("\n\n===== TƯƠNG THÍCH NGƯỢC VỚI MÃ CŨ =====")
    legacy_prompt = legacy_get_prompt("task", "analyze_material", add_cot=True, role="expert_researcher", **sample_data)
    print(f"Legacy prompt tạo ra (độ dài: {len(legacy_prompt)} ký tự)")
    print(f"So sánh với role_cot mới: {len(legacy_prompt) == len(prompt_examples['role_cot_text'])}")
    print("Lưu ý: Hai phương pháp cần tạo ra cùng một prompt để đảm bảo tương thích ngược.")
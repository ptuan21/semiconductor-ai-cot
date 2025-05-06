"""
This module contains predefined prompts for use with LLMs like
LLaMA 3 - 70B and Gemini 1.5 Flash, incorporating various prompting techniques.
"""

# --- Prompt Prefixes (Có thể kết hợp) ---
COT_PREFIX = "Hãy suy nghĩ từng bước."
ROLE_PREFIXES = {
    "expert_researcher": "Bạn là một nhà nghiên cứu chuyên sâu về vật liệu bán dẫn.",
    "helpful_assistant": "Bạn là một trợ lý AI hữu ích và nhiệt tình.",
    "concise_summarizer": "Bạn là một AI chuyên tóm tắt thông tin một cách ngắn gọn và chính xác.",
}

# --- Base Prompts (Không bao gồm các prefix kỹ thuật) ---

GENERAL_PROMPTS = {
    "introduction": "Viết lời giới thiệu về tác động của AI đối với nghiên cứu bán dẫn.",
    "summary": "Tóm tắt văn bản sau:\n{input_text}",
    "question": "Trả lời câu hỏi sau:\n{question}",
}

TASK_PROMPTS = {
    "generate_ideas": "Đề xuất một danh sách các ý tưởng sáng tạo để cải thiện quy trình sản xuất bán dẫn.",
    "explain_concept": "Giải thích khái niệm về '{concept}' bằng thuật ngữ đơn giản.",
    "compare_technologies": "So sánh ưu và nhược điểm của công nghệ '{technology1}' và '{technology2}'.",
    "analyze_material": """Phân tích chi tiết vật liệu bán dẫn với các đặc tính sau. Bình luận về tiềm năng ứng dụng được gợi ý trong cột target. Sử dụng thông tin tham khảo được cung cấp nếu có liên quan.

**Dữ liệu gốc:**
{details_string}

**Thông tin tham khảo từ tài liệu (nếu có):**
{retrieved_context}

**Phân tích (bao gồm nhận xét về tiềm năng ứng dụng được gợi ý, đề xuất pha tạp và chiết suất):**
Hãy suy nghĩ từng bước một cách có hệ thống:
1.  **Phân tích đặc tính cơ bản:** Đánh giá cấu trúc tinh thể, hình thái bề mặt, khe băng, độ dẫn điện, hệ số hấp thụ được cung cấp. Chỉ ra những điểm đáng chú ý hoặc bất thường.
2.  **Đánh giá tiềm năng ứng dụng:** Liên hệ các đặc tính đã phân tích với tiềm năng ứng dụng được gợi ý trong cột 'target_application_potential'. So sánh với thông tin tham khảo nếu có.
3.  **Đề xuất Chiến lược Pha tạp (Doping Strategy):**
    *   Dựa trên vật liệu cơ bản ({material_name}), hãy đề xuất các nguyên tố pha tạp (dopants) phù hợp để tạo ra vật liệu loại n (n-type) hoặc loại p (p-type).
    *   Nêu rõ loại pha tạp (n hay p) mà bạn đề xuất cho các ứng dụng tiềm năng.
    *   Ước tính khoảng nồng độ pha tạp thường được sử dụng (ví dụ: 10^17 - 10^20 cm^-3, hoặc mô tả định tính: thấp, trung bình, cao).
    *   Dự đoán ảnh hưởng chính của việc pha tạp được đề xuất lên độ dẫn điện và/hoặc khe băng của vật liệu.
4.  **Ước tính Chiết suất (Refractive Index Estimation):**
    *   Dựa trên khe băng ({bandgap_energy:.2f} eV) và cấu trúc tinh thể ({crystal_structure}), hãy đưa ra một khoảng giá trị ước tính hợp lý cho chiết suất (n) của vật liệu này trong vùng ánh sáng nhìn thấy (ví dụ: tại bước sóng 500-600 nm).
    *   Giải thích ngắn gọn các yếu tố chính ảnh hưởng đến chiết suất của loại vật liệu này.
5.  **Tổng hợp:** Tóm tắt lại các phân tích, tiềm năng và đề xuất chính.
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


# --- Function to retrieve and combine prompts ---
def get_prompt(category, key, add_cot=False, role=None, **kwargs):
    """
    Lấy và định dạng prompt, tùy chọn thêm các kỹ thuật như CoT và Role Prompting.

    :param category: Danh mục prompt (ví dụ: 'general', 'task', 'research', 'structured', 'few_shot').
    :param key: Khóa của prompt cụ thể trong danh mục.
    :param add_cot: True nếu muốn thêm prefix "Hãy suy nghĩ từng bước.".
    :param role: Key của vai trò trong ROLE_PREFIXES (ví dụ: 'expert_researcher') hoặc None.
    :param kwargs: Các đối số bổ sung để định dạng prompt cơ sở.
    :return: Chuỗi prompt đã được định dạng và kết hợp.
    :raises ValueError: Nếu category, key, hoặc role không hợp lệ.
    """
    categories = {
        "general": GENERAL_PROMPTS,
        "task": TASK_PROMPTS,
        "research": RESEARCH_PROMPTS,
        "structured": STRUCTURED_OUTPUT_PROMPTS,
        "few_shot": FEW_SHOT_PROMPTS,
    }

    if category not in categories:
        raise ValueError(f"Category không hợp lệ: {category}. Các category hợp lệ: {list(categories.keys())}")

    prompts = categories[category]
    if key not in prompts:
        raise ValueError(f"Key không hợp lệ: {key}. Các key hợp lệ cho category '{category}': {list(prompts.keys())}")

    # Lấy prompt cơ sở
    base_prompt = prompts[key]

    # --- THÊM DEBUG Ở ĐÂY ---
    print(f"DEBUG [get_prompt]: Formatting '{category}/{key}'. Received kwargs keys: {list(kwargs.keys())}")
    # ------------------------

    try:
        formatted_base_prompt = base_prompt.format(**kwargs)
    except KeyError as e:
        # --- THÊM DEBUG KHI LỖI ---
        print(f"ERROR [get_prompt]: KeyError formatting '{category}/{key}'. Missing key: {e}. Available kwargs: {list(kwargs.keys())}")
        # -------------------------
        raise ValueError(f"Thiếu đối số định dạng '{e}' cho prompt '{category}/{key}'")

    # Xây dựng prompt cuối cùng
    final_prompt_parts = []

    # Thêm vai trò (nếu có)
    if role:
        if role not in ROLE_PREFIXES:
            raise ValueError(f"Role không hợp lệ: {role}. Các role hợp lệ: {list(ROLE_PREFIXES.keys())}")
        final_prompt_parts.append(ROLE_PREFIXES[role])

    # Thêm CoT (nếu có)
    if add_cot:
        final_prompt_parts.append(COT_PREFIX)

    # Thêm prompt cơ sở đã định dạng
    final_prompt_parts.append(formatted_base_prompt)

    # Kết hợp các phần bằng xuống dòng kép để rõ ràng hơn
    return "\n\n".join(final_prompt_parts)


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Ví dụ Sử dụng get_prompt ---")

    # Prompt cơ bản
    basic_prompt = get_prompt("task", "explain_concept", concept="Band Gap")
    print("\n[Prompt Cơ bản]:\n", basic_prompt)

    # Prompt với Vai trò
    role_prompt = get_prompt("research", "future_trends", role="expert_researcher", field="Photovoltaics")
    print("\n[Prompt với Vai trò]:\n", role_prompt)

    # Prompt với CoT
    cot_prompt = get_prompt("task", "generate_ideas", add_cot=True)
    print("\n[Prompt với CoT]:\n", cot_prompt)

    # Prompt với Vai trò và CoT
    role_cot_prompt = get_prompt("general", "summary", role="concise_summarizer", add_cot=True, input_text="[Văn bản dài cần tóm tắt ở đây...]")
    print("\n[Prompt với Vai trò và CoT]:\n", role_cot_prompt)

    # Prompt yêu cầu Output có cấu trúc
    structured_prompt = get_prompt("structured", "ideas_json")
    print("\n[Prompt Output có Cấu trúc (JSON)]:\n", structured_prompt)

    # Prompt Few-Shot
    few_shot_prompt = get_prompt("few_shot", "classify_material_property", input_text="Nghiên cứu quang phổ hấp thụ UV-Vis cho thấy sự dịch chuyển cạnh hấp thụ về phía năng lượng thấp hơn.")
    print("\n[Prompt Few-Shot]:\n", few_shot_prompt)

    # Lỗi Category không hợp lệ
    try:
        get_prompt("invalid_category", "key")
    except ValueError as e:
        print("\n[Lỗi mong đợi (Category không hợp lệ)]:", e)

    # Lỗi Key không hợp lệ
    try:
        get_prompt("general", "invalid_key")
    except ValueError as e:
        print("\n[Lỗi mong đợi (Key không hợp lệ)]:", e)

    # Lỗi thiếu tham số định dạng
    try:
        get_prompt("task", "explain_concept") # Thiếu 'concept'
    except ValueError as e:
        print("\n[Lỗi mong đợi (Thiếu tham số định dạng)]:", e)

    # Lỗi Role không hợp lệ
    try:
        get_prompt("general", "introduction", role="invalid_role")
    except ValueError as e:
        print("\n[Lỗi mong đợi (Role không hợp lệ)]:", e)

    # Ví dụ cho prompt analyze_material mới
    print("\n--- Ví dụ Prompt Phân tích Vật liệu (với Pha tạp & Chiết suất) ---")
    # Dữ liệu mẫu (lấy từ CSV hoặc tạo giả)
    sample_material_data = {
        'material_name': 'ZnO',
        'crystal_structure': 'Hexagonal',
        'surface_morphology': 'Nanoparticles',
        'bandgap_energy (eV)': 3.3,
        'conductivity (S/cm)': 2.5,
        'absorption_coefficient (cm^-1)': 8e4,
        'target_application_potential': 'UV Application Potential'
    }
    details_str_sample = "\n".join([f"- {k.replace('_', ' ').capitalize()}: {v}" for k, v in sample_material_data.items() if k and v])
    retrieved_context_sample = "- Trích từ 'Paper A': 'Doping ZnO with Al increases conductivity significantly...'\n- Trích từ 'Paper B': 'Refractive index of ZnO thin films is typically around 2.0 in the visible range...'"

    analyze_prompt_full = get_prompt(
        category='task',
        key='analyze_material',
        add_cot=False, # Prompt đã có cấu trúc CoT bên trong
        role='expert_researcher',
        # Truyền các giá trị cần thiết để format prompt
        details_string=details_str_sample,
        retrieved_context=retrieved_context_sample,
        # Truyền thêm các giá trị đơn lẻ nếu prompt cần format trực tiếp
        material_name=sample_material_data['material_name'],
        bandgap_energy=sample_material_data['bandgap_energy (eV)'],
        crystal_structure=sample_material_data['crystal_structure']
        # ... các giá trị khác nếu cần
    )
    print(analyze_prompt_full)
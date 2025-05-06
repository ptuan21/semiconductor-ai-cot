import json
import time
import csv
import statistics # Thêm import
import collections # Thêm import
import os             # Thêm import os để làm việc với đường dẫn file
import re             # Thêm import re để làm sạch text đơn giản
from PyPDF2 import PdfReader # Thêm import từ PyPDF2
from model_manager import init_chat_engine, analyze_prompt_concurrently
from prompts import get_prompt # Import hàm get_prompt đã sửa đổi

# --- Hàm đọc và xử lý PDF ---
def load_and_process_pdfs(pdf_directory):
    """
    Đọc tất cả file PDF trong thư mục, trích xuất text, làm sạch và phân đoạn.
    Returns:
        list: Danh sách các dict [{'filename': str, 'chunk_index': int, 'text': str}]
    """
    all_chunks = []
    print(f"\n📄 Bắt đầu đọc và xử lý PDF từ thư mục: {pdf_directory}")
    try:
        for filename in os.listdir(pdf_directory):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(pdf_directory, filename)
                print(f"   - Đang xử lý: {filename}")
                try:
                    reader = PdfReader(file_path)
                    full_text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text: # Chỉ thêm nếu trích xuất được text
                            full_text += page_text + "\n" # Thêm dấu xuống dòng giữa các trang

                    # Làm sạch text cơ bản (xóa dòng trống, khoảng trắng thừa)
                    cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
                    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text) # Xóa dòng trống

                    # Phân đoạn đơn giản theo đoạn văn (dấu xuống dòng kép) hoặc độ dài cố định
                    # Ở đây dùng dấu xuống dòng kép làm dấu hiệu phân đoạn
                    chunks = cleaned_text.split('\n') # Hoặc dùng kỹ thuật khác
                    chunk_index = 0
                    for chunk in chunks:
                        trimmed_chunk = chunk.strip()
                        if len(trimmed_chunk) > 50: # Chỉ giữ lại chunk có độ dài tối thiểu
                            all_chunks.append({
                                "filename": filename,
                                "chunk_index": chunk_index,
                                "text": trimmed_chunk
                            })
                            chunk_index += 1
                    print(f"     + Trích xuất và phân thành {chunk_index} đoạn.")

                except Exception as e_pdf:
                    print(f"     ⚠️ Lỗi khi đọc file PDF '{filename}': {e_pdf}")
                    # Có thể thêm ghi log lỗi chi tiết hơn ở đây nếu cần
                    pass # Bỏ qua file lỗi và tiếp tục

        print(f"✅ Hoàn thành xử lý PDF. Tổng số đoạn văn bản: {len(all_chunks)}")
        return all_chunks

    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy thư mục PDF: {pdf_directory}")
        return []
    except Exception as e_dir:
        print(f"❌ Lỗi khi truy cập thư mục PDF: {e_dir}")
        return []

# --- Hàm tìm kiếm Chunk liên quan ---
def find_relevant_chunks(material_name, all_pdf_chunks, num_chunks=3):
    """
    Tìm kiếm các đoạn văn bản chứa tên vật liệu (không phân biệt hoa thường).
    Returns:
        list: Danh sách các chuỗi văn bản liên quan (tối đa num_chunks).
    """
    if not material_name or not all_pdf_chunks:
        return []

    relevant_texts = []
    material_name_lower = material_name.lower() # Tìm không phân biệt hoa thường
    print(f"      🔍 Tìm kiếm context cho '{material_name}'...")

    found_count = 0
    for chunk_data in all_pdf_chunks:
        chunk_text = chunk_data.get("text", "")
        if material_name_lower in chunk_text.lower():
            # Tìm thấy, thêm vào kết quả và giới hạn số lượng
            formatted_chunk = f"- Trích từ '{chunk_data.get('filename')}': \"{chunk_text[:250]}...\"" # Trích 1 phần chunk
            relevant_texts.append(formatted_chunk)
            found_count +=1
            if len(relevant_texts) >= num_chunks:
                break # Đủ số lượng yêu cầu

    print(f"      ➡️ Tìm thấy {found_count} đoạn liên quan (lấy tối đa {num_chunks}).")
    return relevant_texts

# --- Các hàm xử lý dữ liệu vật liệu và tạo prompt ---
def load_material_data(csv_path):
    """Đọc dữ liệu vật liệu từ file CSV"""
    materials = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            # Quan trọng: Kiểm tra xem file CSV có header không
            # Nếu dòng đầu tiên là header, dùng DictReader
            # Nếu không, dùng reader thường và truy cập bằng index
            # Giả sử có header:
            reader = csv.DictReader(file)
            for row in reader:
                materials.append(row)
        print(f"📄 Đã đọc {len(materials)} bản ghi từ {csv_path}")
        return materials
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file {csv_path}")
        return []
    except Exception as e:
        print(f"❌ Lỗi khi đọc file CSV: {e}")
        return []

def format_material_prompt_cot(material_data):
    """Định dạng dữ liệu vật liệu thành prompt CoT"""
    prompt_parts = ["Hãy suy nghĩ từng bước và phân tích vật liệu bán dẫn với các đặc tính sau:"]
    for key, value in material_data.items():
        # Xử lý trường hợp key rỗng hoặc value rỗng nếu cần
        if key and value:
             prompt_parts.append(f"- {key.replace('_', ' ').capitalize()}: {value}")
        elif key:
             prompt_parts.append(f"- {key.replace('_', ' ').capitalize()}: [không có giá trị]")
    prompt_parts.append("\nPhân tích:")
    return "\n".join(prompt_parts)

# --- Hàm chính để phân tích (Cập nhật xử lý kiểu bandgap) ---
def analyze_materials_with_prompt_comparison(
    csv_file,
    output_file_base,
    active_engines,
    pdf_chunks_data,
    limit_records=None,
    sleep_between=0):
    """
    Phân tích dữ liệu vật liệu với RAG và nhiều loại prompt, lưu kết quả và phân tích hiệu suất.
    """
    materials = load_material_data(csv_file)
    if not materials: return

    if limit_records is not None and limit_records > 0:
        materials_to_process = materials[:limit_records]
        print(f"ℹ️ Chỉ xử lý {len(materials_to_process)} bản ghi đầu tiên.")
    else:
        materials_to_process = materials
        limit_records = len(materials)

    prompt_variants_config = {
        "basic":    {"category": "task", "key": "analyze_material"},
        "cot":      {"category": "task", "key": "analyze_material", "add_cot": True},
        "role":     {"category": "task", "key": "analyze_material", "role": "expert_researcher"},
        "role_cot": {"category": "task", "key": "analyze_material", "role": "expert_researcher", "add_cot": True},
    }

    all_run_results = []
    total_start_time = time.time()

    for idx, material_data in enumerate(materials_to_process):
        material_id = material_data.get('material_id', f'Bản ghi {idx+1}')

        # --- Xử lý kiểu dữ liệu chặt chẽ hơn cho bandgap ---
        bandgap_val = 0.0 # Giá trị mặc định là float
        raw_bandgap_str = material_data.get('bandgap_energy (eV)')
        if raw_bandgap_str is not None:
            try:
                # Cố gắng chuyển đổi giá trị đọc được thành float
                bandgap_val = float(raw_bandgap_str)
            except (ValueError, TypeError):
                # Nếu chuyển đổi thất bại, ghi cảnh báo và giữ giá trị mặc định 0.0
                print(f"      WARNING: Giá trị bandgap không hợp lệ ('{raw_bandgap_str}') cho vật liệu {material_id}. Sử dụng giá trị mặc định 0.0.")
                # bandgap_val đã là 0.0 rồi
        # Tại thời điểm này, bandgap_val chắc chắn là float
        # ---------------------------------------------------

        crystal_struct_val = material_data.get('crystal_structure', 'Unknown')
        material_name_val = material_data.get('material_name', 'Unknown')

        print(f"\n🔬===========================================")
        print(f"🔬 Bắt đầu phân tích vật liệu: {material_id} ({material_name_val}) (Bản ghi {idx+1}/{len(materials_to_process)})")
        print(f"🔬===========================================")

        retrieved_chunks = find_relevant_chunks(material_name_val, pdf_chunks_data, num_chunks=3)
        retrieved_context_str = "\n".join(retrieved_chunks) if retrieved_chunks else "Không tìm thấy thông tin tham khảo."

        details_str = "\n".join([f"- {k.replace('_', ' ').capitalize()}: {v}" for k, v in material_data.items() if k and v])

        # Tạo kwargs động, đảm bảo bandgap_energy là float
        prompt_kwargs_combined = {
            "details_string": details_str,
            "retrieved_context": retrieved_context_str,
            "material_name": material_name_val,
            "bandgap_energy": bandgap_val, # Truyền giá trị float đã được đảm bảo
            "crystal_structure": crystal_struct_val
        }

        # Lặp qua từng biến thể prompt
        for prompt_name, config in prompt_variants_config.items():
            print(f"\n   --- Thử nghiệm Prompt Type: '{prompt_name}' ---")
            try:
                # Tạo prompt
                current_prompt = get_prompt(
                    category=config["category"],
                    key=config["key"],
                    add_cot=config.get("add_cot", False),
                    role=config.get("role", None),
                    **prompt_kwargs_combined
                )

                # Gọi hàm đồng thời từ model_manager
                engine_run_results = analyze_prompt_concurrently(current_prompt, active_engines)

                # Lưu kết quả
                for run_result in engine_run_results:
                    run_result["prompt_type"] = prompt_name
                    run_result["material_id"] = material_id
                    run_result["retrieved_context_summary"] = f"{len(retrieved_chunks)} chunks found" # Ghi lại số chunk tìm được
                    all_run_results.append(run_result)

            except ValueError as e:
                 print(f"      ❌ Lỗi tạo prompt '{prompt_name}': {e}")
                 all_run_results.append({ "prompt_type": prompt_name, "material_id": material_id, "engine": "N/A", "status": "error", "response": f"Lỗi tạo prompt: {e}", "execution_time": 0, "retrieved_context_summary": f"{len(retrieved_chunks)} chunks found" })
            except Exception as e_call:
                  print(f"      ❌ Lỗi không xác định khi chạy prompt '{prompt_name}': {e_call}")
                  all_run_results.append({ "prompt_type": prompt_name, "material_id": material_id, "engine": "N/A", "status": "error", "response": f"Lỗi thực thi không xác định: {e_call}", "execution_time": 0, "retrieved_context_summary": f"{len(retrieved_chunks)} chunks found"})

        # Chờ giữa các bản ghi nếu cần
        if sleep_between > 0 and idx < len(materials_to_process) - 1:
            print(f"\n   ⏳ Chờ {sleep_between} giây...")
            time.sleep(sleep_between)

    total_end_time = time.time()
    print(f"\n🏁 Hoàn thành thử nghiệm {len(materials_to_process)} bản ghi sau {total_end_time - total_start_time:.2f} giây.")
    output_detailed_file = output_file_base + "_detailed_rag.json" # Đổi tên file output
    try:
        sorted_results = sorted(all_run_results, key=lambda x: (x.get('material_id', ''), x.get('prompt_type', '')))
        with open(output_detailed_file, "w", encoding="utf-8") as f:
            json.dump(sorted_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Đã lưu kết quả chi tiết thử nghiệm RAG vào {output_detailed_file}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu kết quả chi tiết JSON: {e}")
    analyze_performance(all_run_results) # Phân tích hiệu suất như cũ


# --- Hàm phân tích hiệu suất ---
def analyze_performance(all_run_results):
    """Phân tích kết quả chạy và in ra so sánh hiệu suất prompt."""
    if not all_run_results:
        print("\nKhông có kết quả để phân tích.")
        return

    print("\n--- Phân tích Hiệu suất Prompt ---")

    # Nhóm kết quả theo (prompt_type, engine)
    grouped_results = collections.defaultdict(lambda: {"runs": [], "successful_times": [], "errors": 0})
    valid_engines = set()
    prompt_types = set()

    for run in all_run_results:
        pt = run.get("prompt_type", "unknown_prompt")
        eng = run.get("engine", "unknown_engine")
        # Chỉ phân tích các engine và prompt hợp lệ đã chạy
        if eng == "N/A" or pt == "unknown_prompt": continue

        key = (pt, eng)
        grouped_results[key]["runs"].append(run)
        valid_engines.add(eng)
        prompt_types.add(pt)

        if run.get("status") == "success":
            exec_time = run.get("execution_time")
            if exec_time is not None:
                grouped_results[key]["successful_times"].append(exec_time)
        else:
            grouped_results[key]["errors"] += 1

    # In bảng kết quả
    engines_list = sorted(list(valid_engines))
    prompt_types_list = sorted(list(prompt_types))

    header = f"{'Prompt Type':<20}" + "".join([f"{engine.upper():>18}" for engine in engines_list])
    print("\n" + header)
    print("=" * len(header))
    print("METRIC: Avg Time (s) / Errors")
    print("-" * len(header))

    for pt in prompt_types_list:
        row = f"{pt:<20}"
        for eng in engines_list:
            key = (pt, eng)
            data = grouped_results[key]
            total_runs = len(data["runs"])
            if total_runs == 0:
                row += f"{'N/A':>18}"
                continue

            successful_times = data["successful_times"]
            errors = data["errors"]
            avg_time = statistics.mean(successful_times) if successful_times else 0
            # Hiển thị AvgTime / Errors
            cell_value = f"{avg_time:.2f} / {errors}"
            row += f"{cell_value:>18}"
        print(row)

    print("-" * len(header))


if __name__ == "__main__":
    # --- Cấu hình ---
    engines_to_use = ["gemini", "groq"]
    pdf_document_directory = "data/raw/documents" # Đường dẫn đến thư mục chứa PDF
    material_csv_file = os.path.join(pdf_document_directory, "fake_materials_dataset.csv") # Đường dẫn CSV
    output_base_name = "results/rag_prompt_comparison_gemini_groq_flash" # Đổi tên cơ sở

    # --- Tùy chọn Giới hạn và Làm chậm ---
    MAX_RECORDS_TO_PROCESS = 100 # **GIỮ SỐ NHỎ ĐỂ THỬ NGHIỆM RAG**
    SLEEP_BETWEEN_RECORDS = 1

    # --- Bước 1: Tải và xử lý dữ liệu PDF (Chỉ chạy một lần) ---
    pdf_chunks = load_and_process_pdfs(pdf_document_directory)
    if not pdf_chunks:
        print("⚠️ Không có dữ liệu PDF để thực hiện RAG. Script sẽ chạy mà không có ngữ cảnh.")
        # Script vẫn sẽ chạy, nhưng retrieved_context sẽ luôn là "Không tìm thấy..."

    # --- Bước 2: Khởi tạo các engine ---
    print("\n🚀 Khởi tạo các chat engines (Gemini & Groq)...")
    active_chats = [init_chat_engine(name) for name in engines_to_use]
    valid_engines = [
        engine for engine in active_chats
        if engine[0] is not None and (engine[0] != 'gemini' or engine[1] is not None)
    ]

    if not valid_engines:
        print("❌ Không có engine nào được khởi tạo/hợp lệ. Thoát.")
    else:
        print(f"✅ Các engine sẵn sàng: {[e[0].upper() for e in valid_engines]}")
        # --- Bước 3: Chạy thử nghiệm RAG và so sánh prompt ---
        print(f"\n🚀 Bắt đầu thử nghiệm RAG và so sánh prompt (Tối đa {MAX_RECORDS_TO_PROCESS or 'tất cả'} bản ghi)...")
        analyze_materials_with_prompt_comparison(
            material_csv_file,
            output_base_name,
            valid_engines,
            pdf_chunks, # Truyền dữ liệu PDF đã xử lý vào hàm
            limit_records=MAX_RECORDS_TO_PROCESS,
            sleep_between=SLEEP_BETWEEN_RECORDS
        )

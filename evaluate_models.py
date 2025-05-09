import json
import time
import csv
import statistics
import collections
import os
import re
import requests
import concurrent.futures
import threading
import hashlib
import pickle
import numpy as np
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from model_manager import init_chat_engine, analyze_prompt_concurrently, GROQ_API_KEYS, GEMINI_API_KEYS
# Import both new prompt function and legacy function for compatibility
from prompts import get_prompt, legacy_get_prompt
from datetime import datetime, timedelta
import google.generativeai as genai
import traceback

# Load environment variables
load_dotenv()

# Define global flags with default values
SEMANTIC_SEARCH_AVAILABLE = False
VISUALIZATION_AVAILABLE = False

# Thêm imports cho visualizations
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    VISUALIZATION_AVAILABLE = True
    print("✅ Matplotlib đã được cài đặt. Visualizations khả dụng.")
except ImportError:
    print("⚠️ Matplotlib không được cài đặt. Visualizations sẽ bị vô hiệu hóa.")
    print("   Cài đặt với: pip install matplotlib")
    VISUALIZATION_AVAILABLE = False

# Thêm imports cho semantic search
try:
    import scipy.spatial
    from sklearn.feature_extraction.text import TfidfVectorizer
    SEMANTIC_SEARCH_AVAILABLE = True
    print("✅ Thư viện semantic search đã được cài đặt.")
except ImportError:
    print("⚠️ Một số thư viện cần thiết cho semantic search không được cài đặt.")
    print("   Cài đặt với: pip install scipy scikit-learn")
    SEMANTIC_SEARCH_AVAILABLE = False

# --- Thêm hàm tạo dữ liệu mẫu ---
def create_fake_material_csv(csv_path, num_materials=50):
    """Tạo file CSV mẫu với dữ liệu giả lập ngẫu nhiên và đa dạng"""
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Các giá trị ngẫu nhiên
    sample_materials = []
    
    # Định nghĩa các loại vật liệu cơ bản và thông số
    material_names = [
        "Silicon", "Germanium", "Silicon Carbide", "Gallium Arsenide", "Gallium Nitride",
        "Indium Phosphide", "Aluminium Gallium Arsenide", "Indium Gallium Arsenide",
        "Zinc Oxide", "Cadmium Telluride", "Zinc Selenide", "Methylammonium Lead Iodide",
        "Formamidinium Lead Iodide", "Graphene", "Molybdenum Disulfide", "Lead Telluride",
        "Tin Selenide", "Bismuth Telluride", "Copper Indium Gallium Selenide", "Antimony Telluride"
    ]
    
    formulas = [
        "Si", "Ge", "SiC", "GaAs", "GaN", "InP", "AlGaAs", "InGaAs", "ZnO", "CdTe", 
        "ZnSe", "CH3NH3PbI3", "HC(NH2)2PbI3", "C", "MoS2", "PbTe", "SnSe", "Bi2Te3", 
        "Cu(In,Ga)Se2", "Sb2Te3"
    ]
    
    crystal_structures = [
        "Cubic", "Hexagonal", "Tetragonal", "Orthorhombic", "Monoclinic",
        "Zincblende", "Wurtzite", "Perovskite", "Chalcopyrite", "Rhombohedral"
    ]
    
    thermal_stabilities = ["Low", "Medium", "High", "Very High"]
    
    # Định nghĩa các ứng dụng tiềm năng
    applications = [
        "Solar cells, photovoltaics",
        "Transistors, integrated circuits",
        "LEDs, optoelectronics",
        "High-frequency devices",
        "Sensors, photodetectors",
        "Power electronics",
        "Thermoelectric devices",
        "Quantum computing",
        "Spintronics",
        "Flexible electronics",
        "Transparent electronics",
        "MEMS devices",
        "High-temperature electronics",
        "Memory devices",
        "Lasers, optical devices"
    ]
    
    # Tạo dữ liệu ngẫu nhiên đa dạng
    for i in range(num_materials):
        # Tùy chọn phương pháp tạo vật liệu
        method = np.random.choice(["standard", "hybrid", "novel"], p=[0.5, 0.3, 0.2])
        
        material = {}
        material_id = f"M{i+1:03d}"
        material["material_id"] = material_id
        
        if method == "standard":
            # Vật liệu tiêu chuẩn với biến thể
            base_idx = np.random.randint(0, len(material_names))
            material["material_name"] = material_names[base_idx]
            material["formula"] = formulas[base_idx]
            
            # 60% khả năng thêm biến thể
            if np.random.random() < 0.6:
                variant_types = ["doped", "nanostructured", "thin film", "alloy", "modified", 
                                "quantum dot", "2D", "epitaxial", "polycrystalline"]
                material["material_name"] += f" ({np.random.choice(variant_types)})"
        
        elif method == "hybrid":
            # Vật liệu lai
            idx1, idx2 = np.random.choice(len(material_names), 2, replace=False)
            material["material_name"] = f"Hybrid {material_names[idx1]}-{material_names[idx2]}"
            material["formula"] = f"{formulas[idx1]}-{formulas[idx2]}"
            
        else:  # novel
            # Vật liệu hoàn toàn mới
            prefixes = ["Nova", "Quantum", "Flex", "Synth", "Neo", "Ultra", "Adv", "Opt", "Tech"]
            elements = ["Si", "Ga", "Al", "In", "C", "B", "Ge", "Te", "Se", "Zn", "Cd", "Ti"]
            suffixes = ["lite", "nium", "tron", "mide", "zide", "nide", "tite", "phene"]
            
            material["material_name"] = np.random.choice(prefixes) + np.random.choice(elements) + np.random.choice(suffixes)
            # Tạo công thức phức tạp hơn
            num_elements = np.random.randint(1, 4)
            selected_elements = np.random.choice(elements, num_elements, replace=False)
            
            formula_parts = []
            for elem in selected_elements:
                # 50% khả năng thêm số
                if np.random.random() < 0.5:
                    elem += str(np.random.randint(1, 4))
                formula_parts.append(elem)
            
            material["formula"] = "".join(formula_parts)
        
        # Thông số vật lý - tạo ngẫu nhiên với phân bố có ý nghĩa
        material["crystal_structure"] = np.random.choice(crystal_structures)
        
        # Bandgap cố tình tạo phân bố không đều
        if np.random.random() < 0.7:  # 70% trường hợp trong vùng hữu ích
            material["bandgap"] = str(round(np.random.uniform(0.5, 3.5), 2))
        else:  # 30% trường hợp ngoài vùng thông thường
            material["bandgap"] = str(round(np.random.uniform(0, 7), 2))
        
        # Conductivity phân bố log
        cond_exponent = np.random.uniform(-2, 4)  # 0.01 đến 10000 S/cm
        material["conductivity"] = str(round(10 ** cond_exponent, 2))
        
        # Carrier concentration phân bố log
        carrier_conc = 10 ** np.random.uniform(15, 22)
        material["carrier_concentration"] = f"{carrier_conc:.2e}"
        
        # Mobility
        base_mobility = float(material["conductivity"]) * np.random.uniform(0.5, 2.0)
        material["mobility"] = f"{max(10, base_mobility):.1f}"
        
        # Thermal stability
        material["thermal_stability"] = np.random.choice(thermal_stabilities)
        
        # Ứng dụng tiềm năng (dựa một phần vào các thuộc tính)
        num_apps = np.random.randint(1, 4)
        
        # Logic chọn ứng dụng dựa trên thuộc tính
        bandgap = float(material["bandgap"])
        conductivity = float(material["conductivity"])
        
        potential_apps = []
        if 1.0 <= bandgap <= 1.8 and conductivity > 100:
            potential_apps.append("Solar cells, photovoltaics")
        if bandgap > 1.5 and "High" in material["thermal_stability"]:
            potential_apps.append("LEDs, optoelectronics")
        if float(material["mobility"]) > 1000:
            potential_apps.append("High-frequency devices")
        if bandgap > 2.5:
            potential_apps.append("Power electronics")
        if conductivity < 50:
            potential_apps.append("Thermoelectric devices")
        
        # Nếu không có ứng dụng phù hợp, chọn ngẫu nhiên
        if not potential_apps or len(potential_apps) < num_apps:
            remaining = num_apps - len(potential_apps)
            remaining_apps = [app for app in applications if app not in potential_apps]
            if remaining_apps:
                additional_apps = list(np.random.choice(remaining_apps, 
                                                     min(remaining, len(remaining_apps)), 
                                                     replace=False))
                potential_apps.extend(additional_apps)
        
        # Đảm bảo không vượt quá num_apps
        if len(potential_apps) > num_apps:
            potential_apps = list(np.random.choice(potential_apps, num_apps, replace=False))
            
        material["target_application_potential"] = ", ".join(potential_apps)
        
        # Thêm vào danh sách
        sample_materials.append(material)
    
    try:
        # Ghi ra file CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Lấy tên cột từ material đầu tiên
            fieldnames = ["material_id", "material_name", "formula", "crystal_structure", 
                         "bandgap", "conductivity", "carrier_concentration", "mobility", 
                         "thermal_stability", "target_application_potential"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for material in sample_materials:
                writer.writerow(material)
        
        print(f"✅ Đã tạo file CSV mẫu tại: {csv_path} với {len(sample_materials)} bản ghi")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi tạo file CSV mẫu: {e}")
        return False

# --- Thêm hàm tìm từ khóa chung ---
def find_common_keywords(responses):
    """Tìm các từ khóa chung giữa các phản hồi"""
    all_words = {}
    
    for engine, response in responses.items():
        # Tách các từ, loại bỏ dấu câu và chuyển về chữ thường
        words = re.findall(r'\b\w+\b', str(response).lower())
        all_words[engine] = collections.Counter(words)
    
    # Tìm các từ xuất hiện ở cả hai engine
    common_words = {}
    if len(all_words) >= 2:
        engines = list(all_words.keys())
        for word in all_words[engines[0]]:
            if all(word in all_words[eng] for eng in engines[1:]) and len(word) > 3:
                # Chỉ giữ lại những từ có ý nghĩa (dài hơn 3 ký tự)
                count = sum(all_words[eng][word] for eng in engines)
                common_words[word] = count
    
    # Sắp xếp theo số lần xuất hiện
    sorted_keywords = dict(sorted(common_words.items(), 
                                key=lambda item: item[1], 
                                reverse=True)[:10])  # Lấy top 10
    
    return sorted_keywords

class APIRateManager:
    """Quản lý và theo dõi rate limits của các API"""
    def __init__(self):
        self.api_states = {
            "gemini": {
                "apis": ["gemini-1.5-flash-latest"],
                "limits": {
                    "gemini-1.5-flash-latest": {"requests": 60, "window": 60}  # 60 requests/minute
                },
                "usage": {}
            },
            "groq": {
                "apis": ["deepseek-r1-distill-llama-70b"],
                "limits": {
                    "deepseek-r1-distill-llama-70b": {"requests": 50, "window": 60}
                },
                "usage": {}
            }
        }
        self._init_usage_tracking()

    def _init_usage_tracking(self):
        """Khởi tạo tracking cho mỗi API"""
        for provider, config in self.api_states.items():
            for api in config["apis"]:
                config["usage"][api] = {
                    "requests": [],
                    "errors": 0,
                    "last_reset": datetime.now()
                }

    def record_request(self, provider: str, api_version: str):
        """Ghi nhận một request mới"""
        if provider in self.api_states and api_version in self.api_states[provider]["usage"]:
            self.api_states[provider]["usage"][api_version]["requests"].append(datetime.now())

    def record_error(self, provider: str, api_version: str):
        """Ghi nhận một lỗi"""
        if provider in self.api_states and api_version in self.api_states[provider]["usage"]:
            self.api_states[provider]["usage"][api_version]["errors"] += 1

    def get_available_api(self, provider: str) -> str:
        """Kiểm tra và trả về API nếu còn quota"""
        if provider not in self.api_states:
            return None

        self._cleanup_old_requests(provider)
        
        api = self.api_states[provider]["apis"][0]  # Chỉ có một API cho mỗi provider
        usage = self.api_states[provider]["usage"][api]
        limits = self.api_states[provider]["limits"][api]
        
        # Đếm số request trong cửa sổ thời gian
        current_time = datetime.now()
        window_start = current_time - timedelta(seconds=limits["window"])
        recent_requests = len([r for r in usage["requests"] if r > window_start])
        
        # Kiểm tra còn quota không
        if recent_requests < limits["requests"] and usage["errors"] < 3:
            return api
                
        return None

    def _cleanup_old_requests(self, provider: str):
        """Dọn dẹp các request cũ"""
        current_time = datetime.now()
        api = self.api_states[provider]["apis"][0]
        window = self.api_states[provider]["limits"][api]["window"]
        cutoff_time = current_time - timedelta(seconds=window)
        
        # Lọc bỏ các request cũ
        self.api_states[provider]["usage"][api]["requests"] = [
            r for r in self.api_states[provider]["usage"][api]["requests"]
            if r > cutoff_time
        ]

    def get_usage_stats(self) -> dict:
        """Lấy thống kê sử dụng của các API"""
        stats = {}
        current_time = datetime.now()
        
        for provider, config in self.api_states.items():
            stats[provider] = {}
            api = config["apis"][0]
            usage = config["usage"][api]
            limits = config["limits"][api]
            
            window_start = current_time - timedelta(seconds=limits["window"])
            recent_requests = len([r for r in usage["requests"] if r > window_start])
            
            stats[provider][api] = {
                "recent_requests": recent_requests,
                "limit": limits["requests"],
                "errors": usage["errors"],
                "available": recent_requests < limits["requests"]
            }
        
        return stats

    def print_usage_stats(self):
        """In thống kê sử dụng API"""
        stats = self.get_usage_stats()
        print("\n📊 Thống kê sử dụng API:")
        
        for provider, apis in stats.items():
            print(f"\n{provider.upper()}:")
            for api, data in apis.items():
                status = "🟢" if data["available"] else "🔴"
                print(f"{status} {api}: {data['recent_requests']}/{data['limit']} requests, "
                      f"{data['errors']} errors")

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

# --- Thêm lớp SemanticSearchEngine ---
class SemanticSearchEngine:
    """Máy tìm kiếm ngữ nghĩa sử dụng TF-IDF và cosine similarity"""
    
    def __init__(self):
        """Khởi tạo máy tìm kiếm ngữ nghĩa"""
        self.is_ready = SEMANTIC_SEARCH_AVAILABLE
        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        
        if not self.is_ready:
            print("⚠️ Semantic search không khả dụng do thiếu thư viện")
            
    def load_documents(self, pdf_chunks):
        """Tải các đoạn văn bản và tạo ma trận TF-IDF"""
        if not self.is_ready or not pdf_chunks:
            return False
            
        print("🔍 Khởi tạo semantic search engine...")
        self.chunks = pdf_chunks
            
        try:
            # Trích xuất nội dung từ chunks để tạo corpus
            corpus = [chunk.get("text", "") for chunk in self.chunks]
            
            # Khởi tạo và huấn luyện vectorizer
            self.vectorizer = TfidfVectorizer(
                min_df=2, max_df=0.85, 
                stop_words='english', 
                ngram_range=(1, 2)
            )
            
            # Tạo ma trận TF-IDF
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            print(f"✅ Đã khởi tạo semantic search với {len(self.chunks)} đoạn văn bản.")
            print(f"   Ma trận TF-IDF shape: {self.tfidf_matrix.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi khởi tạo semantic search: {e}")
            self.is_ready = False
            return False
            
    def search(self, query, top_k=5):
        """Tìm kiếm các đoạn văn bản liên quan nhất với query"""
        if not self.is_ready or not self.vectorizer or self.tfidf_matrix is None:
            print("⚠️ Semantic search engine chưa được khởi tạo")
            return []
            
        try:
            # Chuyển đổi query thành vector
            query_vector = self.vectorizer.transform([query])
            
            # Tính toán cosine similarity
            cosine_similarities = scipy.spatial.distance.cdist(
                query_vector.toarray(), self.tfidf_matrix.toarray(), 'cosine'
            ).flatten()
            
            # Chuyển đổi khoảng cách cosine thành điểm tương đồng (1 - khoảng cách)
            similarity_scores = 1 - cosine_similarities
            
            # Lấy các chỉ số có điểm cao nhất
            top_indices = similarity_scores.argsort()[-top_k:][::-1]
            
            # Lọc các kết quả có điểm tương đồng cao hơn ngưỡng
            results = []
            for idx in top_indices:
                if similarity_scores[idx] > 0.1:  # Ngưỡng tương đồng tối thiểu
                    chunk = self.chunks[idx]
                    results.append({
                        "chunk": chunk,
                        "score": similarity_scores[idx],
                        "index": idx
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Lỗi khi tìm kiếm: {e}")
            return []

# --- Cập nhật hàm find_relevant_chunks sử dụng semantic search ---
def find_relevant_chunks(material_name, all_pdf_chunks, num_chunks=3, use_semantic=True, search_engine=None):
    """
    Cải thiện tìm kiếm các đoạn văn bản liên quan với semantic search và scoring system
    
    Args:
        material_name: Tên vật liệu
        all_pdf_chunks: Danh sách các đoạn văn bản
        num_chunks: Số lượng đoạn cần trả về
        use_semantic: Có sử dụng semantic search hay không
        search_engine: Đối tượng SemanticSearchEngine
    """
    if not material_name or not all_pdf_chunks:
        return []

    # Sử dụng semantic search nếu có thể
    if use_semantic and search_engine and search_engine.is_ready:
        print(f"      🔍 Tìm kiếm context bằng semantic search cho '{material_name}'...")
        
        # Tạo query từ tên vật liệu và các từ khóa liên quan
        query = f"{material_name} semiconductor properties bandgap conductivity"
        
        # Thực hiện tìm kiếm ngữ nghĩa
        semantic_results = search_engine.search(query, top_k=num_chunks)
        
        if semantic_results:
            formatted_chunks = []
            for result in semantic_results:
                chunk = result["chunk"]
                context = f"[File: {chunk.get('filename', 'unknown')}] (Score: {result['score']:.3f})\n"
                
                # Lấy context trước và sau nếu có
                chunk_index = chunk.get('chunk_index', 0)
                prev_chunk_text = ""
                next_chunk_text = ""
                
                for c in all_pdf_chunks:
                    if c.get('filename') == chunk.get('filename'):
                        if c.get('chunk_index') == chunk_index - 1:
                            prev_chunk_text = c.get('text', '')
                        elif c.get('chunk_index') == chunk_index + 1:
                            next_chunk_text = c.get('text', '')
                
                if prev_chunk_text:
                    context += f"Previous: {prev_chunk_text[:100]}...\n"
                context += f"Content: {chunk.get('text', '')}\n"
                if next_chunk_text:
                    context += f"Next: {next_chunk_text[:100]}..."
                
                formatted_chunks.append(context)
            
            print(f"      ➡️ Tìm thấy {len(formatted_chunks)} đoạn liên quan với semantic search.")
            return formatted_chunks
    
    # Fallback to scoring system
    scored_chunks = []
    material_name_lower = material_name.lower()
    print(f"      🔍 Tìm kiếm context với scoring system cho '{material_name}'...")

    # Tính điểm cho mỗi chunk
    for i, chunk_data in enumerate(all_pdf_chunks):
        chunk_text = chunk_data.get("text", "").lower()
        score = 0
        
        # Điểm cho tên vật liệu
        if material_name_lower in chunk_text:
            score += 10
        
        # Điểm cho các từ khóa quan trọng
        keywords = ["bandgap", "crystal", "conductivity", "semiconductor", 
                   "properties", "structure", "synthesis", "characterization"]
        for keyword in keywords:
            if keyword in chunk_text:
                score += 2
        
        # Điểm cho các con số và đơn vị đo
        if any(char.isdigit() for char in chunk_text):
            score += 3
        for unit in ["ev", "k", "°c", "cm", "μm", "nm"]:
            if unit in chunk_text:
                score += 2
        
        if score > 0:
            # Lấy context trước và sau nếu có
            prev_chunk_text = all_pdf_chunks[i-1].get("text", "") if i > 0 else ""
            next_chunk_text = all_pdf_chunks[i+1].get("text", "") if i < len(all_pdf_chunks)-1 else ""
            
            scored_chunks.append({
                "text": chunk_data["text"],
                "score": score,
                "filename": chunk_data.get("filename", "unknown"),
                "context": {
                    "prev_chunk": prev_chunk_text,
                    "next_chunk": next_chunk_text
                }
            })
    
    # Sắp xếp theo điểm và lấy các chunk tốt nhất
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    best_chunks = scored_chunks[:num_chunks]
    
    # Format kết quả với context
    formatted_chunks = []
    for chunk in best_chunks:
        context = f"[File: {chunk['filename']}] (Score: {chunk['score']})\n"
        if chunk['context']['prev_chunk']:
            context += f"Previous: {chunk['context']['prev_chunk'][:100]}...\n"
        context += f"Content: {chunk['text']}\n"
        if chunk['context']['next_chunk']:
            context += f"Next: {chunk['context']['next_chunk'][:100]}..."
        formatted_chunks.append(context)
    
    print(f"      ➡️ Tìm thấy {len(formatted_chunks)} đoạn liên quan với scoring system.")
    return formatted_chunks

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

def format_material_details(material_data):
    """Định dạng dữ liệu vật liệu thành chuỗi chi tiết"""
    details = []
    for key, value in material_data.items():
        if key and value:  # Chỉ thêm các trường có giá trị
            formatted_key = key.replace('_', ' ').title()
            details.append(f"- {formatted_key}: {value}")
    return "\n".join(details)

# --- Hàm chính để phân tích (Cập nhật xử lý kiểu bandgap) ---
def calculate_material_score(material_data, analysis_results):
    """Tính điểm đánh giá vật liệu dựa trên thuộc tính và kết quả phân tích"""
    score = 0
    max_score = 100
    scoring_breakdown = {}
    
    # Điểm cho bandgap - phù hợp cho ứng dụng cụ thể
    try:
        bandgap = float(material_data.get('bandgap', 0))
        if 0.5 <= bandgap <= 2.0:  # Lý tưởng cho pin mặt trời
            score += 20
            scoring_breakdown['bandgap'] = "Tối ưu cho ứng dụng pin mặt trời (+20)"
        elif 2.0 < bandgap <= 3.5:  # Tốt cho LED và optoelectronics
            score += 15
            scoring_breakdown['bandgap'] = "Tốt cho LED và quang điện tử (+15)"
        elif bandgap > 3.5:  # Insulator
            score += 5
            scoring_breakdown['bandgap'] = "Cách điện, phù hợp cho điện tử công suất cao (+5)"
        else:  # Gần kim loại
            score += 10
            scoring_breakdown['bandgap'] = "Gần kim loại, tốt cho kết nối (+10)"
    except:
        scoring_breakdown['bandgap'] = "Không thể đánh giá (0)"
    
    # Điểm cho độ dẫn điện
    try:
        conductivity = float(material_data.get('conductivity', 0))
        if conductivity > 1000:
            score += 15
            scoring_breakdown['conductivity'] = "Độ dẫn điện rất cao (+15)"
        elif 500 <= conductivity <= 1000:
            score += 10
            scoring_breakdown['conductivity'] = "Độ dẫn điện cao (+10)"
        else:
            score += 5
            scoring_breakdown['conductivity'] = "Độ dẫn điện thấp/trung bình (+5)"
    except:
        scoring_breakdown['conductivity'] = "Không thể đánh giá (0)"
    
    # Điểm cho độ bền nhiệt
    thermal_stability = material_data.get('thermal_stability', '').lower()
    if 'high' in thermal_stability or 'cao' in thermal_stability:
        score += 15
        scoring_breakdown['thermal_stability'] = "Độ bền nhiệt cao (+15)"
    elif 'medium' in thermal_stability or 'trung bình' in thermal_stability:
        score += 10
        scoring_breakdown['thermal_stability'] = "Độ bền nhiệt trung bình (+10)"
    elif 'low' in thermal_stability or 'thấp' in thermal_stability:
        score += 5
        scoring_breakdown['thermal_stability'] = "Độ bền nhiệt thấp (+5)"
    else:
        scoring_breakdown['thermal_stability'] = "Không thể đánh giá (0)"
    
    # Điểm cho ứng dụng tiềm năng
    applications = material_data.get('target_application_potential', '').lower()
    if 'solar' in applications or 'pin mặt trời' in applications:
        score += 15
        scoring_breakdown['applications'] = "Tiềm năng cho pin mặt trời (+15)"
    elif 'transistor' in applications or 'bán dẫn' in applications:
        score += 15
        scoring_breakdown['applications'] = "Tiềm năng cho transistor (+15)"
    elif 'led' in applications or 'light' in applications:
        score += 12
        scoring_breakdown['applications'] = "Tiềm năng cho LED và quang điện tử (+12)"
    elif 'sensor' in applications or 'cảm biến' in applications:
        score += 10
        scoring_breakdown['applications'] = "Tiềm năng cho cảm biến (+10)"
    else:
        score += 5
        scoring_breakdown['applications'] = "Tiềm năng chưa xác định rõ (+5)"
    
    # Điểm cho cấu trúc tinh thể
    crystal_structure = material_data.get('crystal_structure', '').lower()
    if 'cubic' in crystal_structure:
        score += 10  # Cấu trúc cubic thường ổn định và dễ chế tạo
        scoring_breakdown['crystal_structure'] = "Cấu trúc cubic, ổn định và dễ chế tạo (+10)"
    elif 'hexagonal' in crystal_structure:
        score += 8  # Hexagonal tốt cho một số ứng dụng cụ thể
        scoring_breakdown['crystal_structure'] = "Cấu trúc lục giác, tốt cho ứng dụng đặc biệt (+8)"
    elif 'monoclinic' in crystal_structure:
        score += 6  # Monoclinic phức tạp hơn
        scoring_breakdown['crystal_structure'] = "Cấu trúc monoclinic, phức tạp hơn (+6)"
    else:
        scoring_breakdown['crystal_structure'] = "Không thể đánh giá (0)"
    
    # Điểm từ kết quả phân tích AI
    if analysis_results:
        for engine_result in analysis_results:
            if engine_result.get("status") == "success":
                response_text = str(engine_result.get("response", "")).lower()
                
                # Kiểm tra nếu AI đề xuất ứng dụng triển vọng
                if "high potential" in response_text or "triển vọng cao" in response_text:
                    score += 10
                    scoring_breakdown['ai_analysis'] = "AI đánh giá tiềm năng cao (+10)"
                    break
                elif "promising" in response_text or "triển vọng" in response_text:
                    score += 5
                    scoring_breakdown['ai_analysis'] = "AI đánh giá có triển vọng (+5)"
                    break
    
    # Chuẩn hóa điểm về thang 100
    normalized_score = min(100, score)
    
    return {
        "score": normalized_score,
        "max_score": max_score,
        "percentage": f"{(normalized_score/max_score)*100:.1f}%",
        "breakdown": scoring_breakdown,
        "rating": "Xuất sắc" if normalized_score >= 80 else
                  "Tốt" if normalized_score >= 60 else
                  "Trung bình" if normalized_score >= 40 else
                  "Kém"
    }

def analyze_materials_with_prompt_comparison(
    csv_file,
    output_file_base,
    active_engines,
    pdf_chunks_data,
    limit_records=None,
    sleep_between=0,
    max_workers=4,
    batch_size=2,
    api_rate_manager=None,
    response_cache=None):  # Thêm tham số
    """
    Phân tích dữ liệu vật liệu với RAG và nhiều loại prompt, lưu kết quả và phân tích hiệu suất.
    Hỗ trợ xử lý song song với multi-threading và xử lý theo batch.
    """
    all_run_results = []  # Khởi tạo list rỗng ngay từ đầu
    material_scores = {}  # Lưu trữ điểm đánh giá cho các vật liệu
    results_lock = threading.Lock()  # Lock để đảm bảo thread safety khi ghi kết quả
    
    try:
        materials = load_material_data(csv_file)
        if not materials: 
            print("❌ Không thể đọc dữ liệu vật liệu")
            return all_run_results, material_scores  # Trả về list rỗng và dict rỗng

        if limit_records is not None and limit_records > 0:
            materials_to_process = materials[:limit_records]
            print(f"ℹ️ Chỉ xử lý {len(materials_to_process)} bản ghi đầu tiên.")
        else:
            materials_to_process = materials
            limit_records = len(materials)

        prompt_variants_config = {
            # Text response variants
            "basic": {"prompt_type": "basic", "output_format": "text"},
            "cot": {"prompt_type": "cot", "output_format": "text"},
            "role": {"prompt_type": "role", "output_format": "text"},
            "role_cot": {"prompt_type": "role_cot", "output_format": "text"},
            
            # JSON structured response variants (optional)
            "basic_json": {"prompt_type": "basic", "output_format": "json"},
            "cot_json": {"prompt_type": "cot", "output_format": "json"},
            "role_json": {"prompt_type": "role", "output_format": "json"},
            "role_cot_json": {"prompt_type": "role_cot", "output_format": "json"}
        }

        total_start_time = time.time()
        
        # Chia danh sách vật liệu thành các batch
        batches = []
        for i in range(0, len(materials_to_process), batch_size):
            batches.append(materials_to_process[i:i+batch_size])
        
        print(f"\n🧮 Chia {len(materials_to_process)} vật liệu thành {len(batches)} batch, mỗi batch {batch_size} vật liệu")
        
        # Hàm xử lý một batch vật liệu trong thread riêng
        def process_batch(batch_idx, materials_batch):
            batch_start_time = time.time()
            print(f"\n🔬 Xử lý batch {batch_idx+1}/{len(batches)} với {len(materials_batch)} vật liệu")
            
            # Xử lý batch
            batch_results = process_materials_batch(
                materials_batch,
                prompt_variants_config,
                active_engines,
                pdf_chunks_data,
                sleep_between,
                search_engine=semantic_search_engine if 'semantic_search_engine' in globals() else None,
                api_rate_manager=api_rate_manager,
                response_cache=response_cache
            )
            
            # Thread-safe thêm kết quả vào danh sách chung
            with results_lock:
                for material_result in batch_results:
                    # Thêm kết quả phân tích
                    all_run_results.extend(material_result["analysis_results"])
                    
                    # Thêm điểm đánh giá
                    material_id = material_result["material_id"]
                    material_data = material_result["material_data"]
                    material_name = material_result["material_name"]
                    
                    material_scores[material_id] = {
                        "score": material_result["material_score"],
                        "material_name": material_name,
                        "properties": {
                            "bandgap": material_data.get('bandgap', 'N/A'),
                            "crystal_structure": material_data.get('crystal_structure', 'N/A'),
                            "conductivity": material_data.get('conductivity', 'N/A'),
                            "thermal_stability": material_data.get('thermal_stability', 'N/A')
                        }
                    }
            
            batch_time = time.time() - batch_start_time
            print(f"✅ Hoàn thành batch {batch_idx+1} sau {batch_time:.2f} giây")
            return batch_idx
        
        # Sử dụng ThreadPoolExecutor để xử lý song song các batch
        print(f"\n🧵 Bắt đầu xử lý song song với {max_workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit các task
            futures = {executor.submit(process_batch, idx, batch): (idx, batch) 
                      for idx, batch in enumerate(batches)}
            
            # Xử lý kết quả khi hoàn thành
            for future in concurrent.futures.as_completed(futures):
                idx, batch = futures[future]
                try:
                    batch_idx = future.result()
                    print(f"🏁 Batch {batch_idx+1} đã hoàn thành xử lý")
                except Exception as exc:
                    print(f"❌ Batch {idx+1} gặp lỗi: {exc}")

        total_end_time = time.time()
        print(f"\n🏁 Hoàn thành phân tích {len(materials_to_process)} bản ghi sau {total_end_time - total_start_time:.2f} giây.")

        # Lưu kết quả chi tiết
        output_detailed_file = output_file_base + "_detailed_rag.json"
        try:
            sorted_results = sorted(all_run_results, key=lambda x: (x.get('material_id', ''), x.get('prompt_type', '')))
            with open(output_detailed_file, "w", encoding="utf-8") as f:
                json.dump(sorted_results, f, indent=2, ensure_ascii=False)
            print(f"✅ Đã lưu kết quả chi tiết vào {output_detailed_file}")
            
            # Lưu kết quả đánh giá vật liệu
            output_scores_file = output_file_base + "_material_scores.json"
            with open(output_scores_file, "w", encoding="utf-8") as f:
                json.dump(material_scores, f, indent=2, ensure_ascii=False)
            print(f"✅ Đã lưu kết quả đánh giá vật liệu vào {output_scores_file}")
            
            # Phân tích hiệu suất các loại prompt
            print("\n📊 Đang phân tích hiệu suất và so sánh các loại prompt...")
            analyze_performance(all_run_results)
            
            # Phân tích chuyên sâu và so sánh các loại prompt
            prompt_comparison_results = compare_prompt_types_depth(all_run_results)
            
            # Lưu kết quả so sánh prompt
            output_prompt_comparison_file = output_file_base + "_prompt_comparison.json"
            with open(output_prompt_comparison_file, "w", encoding="utf-8") as f:
                json.dump(prompt_comparison_results, f, indent=2, ensure_ascii=False)
            print(f"✅ Đã lưu kết quả so sánh prompt vào {output_prompt_comparison_file}")
            
            # Tạo visualization cho kết quả so sánh prompt nếu có matplotlib
            if VISUALIZATION_AVAILABLE:
                generate_visualizations(material_scores, output_file_base)
                generate_prompt_comparison_visualizations(prompt_comparison_results, output_file_base)
            
        except Exception as e:
            print(f"❌ Lỗi khi lưu và phân tích kết quả: {e}")

    except Exception as e:
        print(f"❌ Lỗi trong quá trình phân tích: {e}")
        
    return all_run_results, material_scores  # Luôn trả về cả hai

def truncate_prompt_for_groq(prompt, max_chars=8000):
    """Cắt giảm prompt quá dài cho Groq API"""
    if len(prompt) <= max_chars:
        return prompt
    
    print(f"⚠️ Prompt quá dài cho Groq ({len(prompt)} ký tự), cắt giảm xuống {max_chars} ký tự...")
    
    # Tìm phần retrieved_context trong prompt
    context_start = prompt.find("**Thông tin tham khảo từ tài liệu (nếu có):**")
    if context_start > 0:
        # Chia prompt thành 3 phần: phần đầu, phần context, phần cuối
        context_end = prompt.find("**Phân tích", context_start)
        if context_end > 0:
            # Có thể cắt giảm context
            prefix = prompt[:context_start]
            context = prompt[context_start:context_end]
            suffix = prompt[context_end:]
            
            # Tính toán số ký tự tối đa cho context
            max_context_chars = max_chars - len(prefix) - len(suffix) - 100  # Thêm buffer
            if max_context_chars > 200:  # Đảm bảo context vẫn còn ý nghĩa
                truncated_context = context[:max_context_chars] + "...\n[Context bị cắt bớt do giới hạn token]\n\n"
                return prefix + truncated_context + suffix
    
    # Nếu không thể cắt thông minh, cắt đơn giản
    return prompt[:max_chars] + "...\n[Prompt bị cắt bớt do giới hạn token]"

# --- Thêm lớp ResponseCache ---
class ResponseCache:
    """Lưu trữ và tái sử dụng các phản hồi từ API để tránh gọi trùng lặp"""
    
    def __init__(self, cache_dir="cache"):
        """Khởi tạo bộ nhớ cache"""
        self.cache_dir = cache_dir
        self.cache = {}
        self.lock = threading.Lock()
        
        # Tạo thư mục cache nếu chưa tồn tại
        os.makedirs(cache_dir, exist_ok=True)
        
        # Đường dẫn đến file cache
        self.cache_file = os.path.join(cache_dir, "api_response_cache.pkl")
        
        # Tải cache từ file nếu tồn tại
        self._load_cache()
        
        print(f"📦 Cache khởi tạo với {len(self.cache)} mục.")
    
    def _load_cache(self):
        """Tải cache từ file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
        except Exception as e:
            print(f"⚠️ Không thể tải cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Lưu cache vào file"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"⚠️ Không thể lưu cache: {e}")
    
    def compute_hash(self, prompt, engine_name):
        """Tính toán hash cho prompt và engine"""
        # Chỉ lấy 200 ký tự đầu và cuối để tránh hash quá dài
        if len(prompt) > 400:
            key_content = prompt[:200] + prompt[-200:]
        else:
            key_content = prompt
            
        # Kết hợp prompt và engine để tạo hash
        hash_input = f"{key_content}_{engine_name}".encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()
    
    def get(self, prompt, engine_name):
        """Lấy phản hồi từ cache nếu tồn tại"""
        prompt_hash = self.compute_hash(prompt, engine_name)
        with self.lock:
            return self.cache.get(prompt_hash)
    
    def set(self, prompt, engine_name, response, execution_time):
        """Lưu phản hồi vào cache"""
        prompt_hash = self.compute_hash(prompt, engine_name)
        with self.lock:
            self.cache[prompt_hash] = {
                "response": response,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            # Lưu cache mỗi khi có cập nhật mới
            self._save_cache()
    
    def clear(self):
        """Xóa tất cả cache"""
        with self.lock:
            self.cache = {}
            self._save_cache()
    
    def stats(self):
        """Thống kê về cache"""
        return {
            "total_entries": len(self.cache),
            "size_kb": os.path.getsize(self.cache_file) / 1024 if os.path.exists(self.cache_file) else 0
        }

# --- Cập nhật hàm analyze_with_engine để sử dụng cache ---
def analyze_with_engine(engine_name, engine_instance, prompt, api_version, api_rate_manager=None, response_cache=None, max_retries=2):
    """Phân tích với một engine cụ thể và version API, sử dụng cache để tránh gọi trùng lặp"""
    retries = 0
    
    while retries <= max_retries:
        try:
            # Kiểm tra trong cache trước
            if response_cache:
                cached_result = response_cache.get(prompt, engine_name)
                if cached_result:
                    print(f"   🔄 Sử dụng kết quả từ cache cho {engine_name}")
                    return {
                        "engine": engine_name,
                        "status": "success",
                        "response": cached_result["response"],
                        "execution_time": cached_result["execution_time"],
                        "api_version": api_version,
                        "from_cache": True
                    }
            
            start_time = time.time()

            # Kiểm tra API có khả dụng không
            if api_rate_manager:
                available_api = api_rate_manager.get_available_api(engine_name)
                if not available_api:
                    # Nếu đã retry quá max_retries lần
                    if retries == max_retries:
                        # Thử quét lại APIs để tìm model khả dụng khác
                        print(f"⚠️ API {engine_name} đã hết quota, đang quét lại APIs...")
                        api_scan_results = scan_available_apis(api_rate_manager, verbose=False)
                        alternative_engines = api_scan_results["summary"]["recommended_apis"]
                        
                        # Nếu có API thay thế khả dụng khác, thử sử dụng
                        if alternative_engines and engine_name not in alternative_engines:
                            alt_engine = next((e for e in alternative_engines if e != engine_name), None)
                            if alt_engine:
                                print(f"🔄 Chuyển sang sử dụng {alt_engine.upper()} thay vì {engine_name.upper()}")
                                
                                # Khởi tạo engine mới
                                alt_engine_tuple = init_chat_engine(alt_engine)
                                if alt_engine_tuple[1] is not None:
                                    # Gọi lại hàm với engine mới
                                    return analyze_with_engine(
                                        alt_engine_tuple[0], 
                                        alt_engine_tuple[1],
                                        prompt,
                                        api_rate_manager.api_states[alt_engine]["apis"][0],
                                        api_rate_manager,
                                        response_cache,
                                        max_retries=1  # Giảm số lần retry với engine mới
                                    )
                        
                        # Nếu không tìm được API thay thế
                        raise Exception(f"API {engine_name} hết quota và không tìm thấy API thay thế")
                    
                    # Đợi và thử lại
                    print(f"⚠️ API {engine_name} đã hết quota, đợi 60 giây...")
                    time.sleep(60)  # Đợi 1 phút khi hết quota
                    retries += 1
                    continue
                
                # Ghi nhận request
                api_rate_manager.record_request(engine_name, available_api)
            else:
                available_api = api_version
            
            print(f"🔄 Đang sử dụng {engine_name} - {available_api}")
            
            # Thực hiện request dựa trên loại engine
            if engine_name == "gemini":
                if engine_instance is None:
                    raise Exception("Gemini engine không được khởi tạo thành công")
                    
                # Gemini không chấp nhận api_version trong send_message
                print(f"   Gửi yêu cầu tới Gemini...")
                response = engine_instance.send_message(prompt)
                response_text = response.text
                
            elif engine_name == "groq":
                # Xử lý Groq bằng cách gọi API trực tiếp
                if not GROQ_API_KEYS or not GROQ_API_KEYS[0]:
                    raise Exception("Groq API key chưa được cấu hình")
                
                # Cắt giảm prompt cho Groq để không vượt quá token limit
                groq_prompt = truncate_prompt_for_groq(prompt)
                    
                print(f"   Gửi yêu cầu tới Groq với model {available_api}...")
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEYS[0]}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": available_api,
                    "messages": [{"role": "user", "content": groq_prompt}],
                    "max_tokens": 800
                }
                
                try:
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions", 
                        headers=headers, 
                        json=payload, 
                        timeout=180
                    )
                    
                    if response.status_code != 200:
                        error_detail = "Unknown error"
                        try:
                            error_data = response.json()
                            error_detail = error_data.get("error", {}).get("message", "Unknown error")
                        except:
                            error_detail = response.text
                        
                        # Nếu lỗi yêu cầu quá lớn, thử giảm kích thước prompt hơn nữa
                        if "Request too large" in response.text or "413" == str(response.status_code):
                            print(f"⚠️ Yêu cầu vẫn quá lớn, cắt giảm mạnh prompt...")
                            # Cắt giảm mạnh hơn nữa, chỉ giữ lại thông tin vật liệu
                            material_info = "Phân tích vật liệu bán dẫn với các đặc tính sau:\n\n" + groq_prompt.split("**Dữ liệu gốc:**")[1].split("**Thông tin tham khảo")[0]
                            material_info += "\n\nVui lòng phân tích đặc tính vật liệu này."
                            
                            payload["messages"] = [{"role": "user", "content": material_info}]
                            response = requests.post(
                                "https://api.groq.com/openai/v1/chat/completions", 
                                headers=headers, 
                                json=payload, 
                                timeout=180
                            )
                            
                            if response.status_code != 200:
                                raise Exception(f"Groq API Error ({response.status_code}): {response.text}")
                        # Nếu lỗi model không tồn tại, thử lấy model khác
                        elif "model_not_found" in response.text or "does not exist" in response.text:
                            print(f"⚠️ Model {available_api} không tồn tại, đang kiểm tra model khác...")
                            models = list_available_groq_models()
                            
                            if models:
                                if "deepseek-r1-distill-llama-70b" in models:
                                    correct_model = "deepseek-r1-distill-llama-70b"
                                else:
                                    correct_model = models[0]
                                    
                                print(f"🔄 Thử lại với model {correct_model}...")
                                
                                # Cập nhật API state
                                if api_rate_manager and "groq" in api_rate_manager.api_states:
                                    api_rate_manager.api_states["groq"]["apis"] = [correct_model]
                                    api_rate_manager.api_states["groq"]["limits"] = {
                                        correct_model: {"requests": 50, "window": 60}
                                    }
                                    api_rate_manager._init_usage_tracking()
                                
                                # Thử lại request với model mới
                                payload["model"] = correct_model
                                response = requests.post(
                                    "https://api.groq.com/openai/v1/chat/completions", 
                                    headers=headers, 
                                    json=payload, 
                                    timeout=180
                                )
                                
                                if response.status_code != 200:
                                    raise Exception(f"Groq API Error ({response.status_code}): {response.text}")
                            else:
                                raise Exception(f"Không thể lấy danh sách model Groq: {error_detail}")
                        else:
                            # Nếu lỗi khác, thử retry
                            if retries < max_retries:
                                print(f"⚠️ Groq API lỗi ({response.status_code}), thử lại lần {retries+1}...")
                                retries += 1
                                time.sleep(5)  # Đợi 5 giây trước khi thử lại
                                continue
                            else:
                                raise Exception(f"Groq API Error ({response.status_code}): {error_detail}")
                    
                    response_text = response.json()['choices'][0]['message']['content']
                    
                except requests.exceptions.RequestException as e:
                    # Kiểm tra nếu là lỗi timeout hoặc kết nối, thử lại
                    if retries < max_retries:
                        print(f"⚠️ Lỗi kết nối tới Groq API: {str(e)}, thử lại lần {retries+1}...")
                        retries += 1
                        time.sleep(5)  # Đợi 5 giây trước khi thử lại
                        continue
                    else:
                        raise Exception(f"Lỗi kết nối đến Groq API: {str(e)}")
                
            else:
                raise Exception(f"Engine không được hỗ trợ: {engine_name}")
            
            execution_time = time.time() - start_time
            print(f"   ✅ Nhận phản hồi sau {execution_time:.2f} giây.")
            
            # Lưu vào cache
            if response_cache:
                response_cache.set(prompt, engine_name, response_text, execution_time)
            
            return {
                "engine": engine_name,
                "status": "success",
                "response": response_text,
                "execution_time": execution_time,
                "api_version": available_api,
                "from_cache": False
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Xử lý lỗi rate limit/quota
            if ("rate limit" in error_msg or "quota exceeded" in error_msg) and api_rate_manager:
                print(f"⚠️ Hết quota API {engine_name} - {api_version}")
                api_rate_manager.record_error(engine_name, api_version)
                
                # Còn cơ hội retry
                if retries < max_retries:
                    print(f"   🔄 Thử lại lần {retries+1}/{max_retries}...")
                    retries += 1
                    time.sleep(30)  # Đợi 30 giây và thử lại
                    continue
                
                # Thử quét lại APIs để tìm API thay thế
                print("   🔎 Đang quét lại APIs để tìm API thay thế...")
                api_scan_results = scan_available_apis(api_rate_manager, verbose=False)
                alternative_engines = api_scan_results["summary"]["recommended_apis"]
                
                # Nếu có API thay thế khả dụng khác
                if alternative_engines and engine_name not in alternative_engines:
                    alt_engine = next((e for e in alternative_engines if e != engine_name), None)
                    if alt_engine:
                        print(f"🔄 Chuyển sang sử dụng {alt_engine.upper()} thay vì {engine_name.upper()}")
                        
                        # Khởi tạo engine mới
                        alt_engine_tuple = init_chat_engine(alt_engine)
                        if alt_engine_tuple[1] is not None:
                            # Gọi lại hàm với engine mới
                            return analyze_with_engine(
                                alt_engine_tuple[0], 
                                alt_engine_tuple[1],
                                prompt,
                                api_rate_manager.api_states[alt_engine]["apis"][0],
                                api_rate_manager,
                                response_cache,
                                max_retries=1  # Giảm số lần retry với engine mới
                            )
            else:
                print(f"❌ Lỗi với {engine_name}: {str(e)}")
                
                # Nếu là engine failure đơn giản, thử lại nếu còn retries
                if retries < max_retries:
                    print(f"   🔄 Thử lại lần {retries+1}/{max_retries}...")
                    retries += 1
                    time.sleep(5)  # Đợi 5 giây và thử lại
                    continue
            
            # Hết retries hoặc không phải lỗi có thể khắc phục với retry
            return {
                "engine": engine_name,
                "status": "error",
                "error": str(e),
                "execution_time": 0,
                "api_version": api_version
            }

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

def analyze_results_in_detail(all_run_results):
    """Phân tích chi tiết kết quả từ các model"""
    if not all_run_results:
        print("\n⚠️ Không có kết quả để phân tích.")
        return
        
    print("\n=== Phân tích Chi tiết Kết quả ===")
    
    # Phân tích theo từng model
    for engine in ["gemini", "groq"]:
        engine_results = [r for r in all_run_results if r.get("engine") == engine]
        if not engine_results:
            print(f"\n⚠️ Không có kết quả cho model {engine.upper()}")
            continue
            
        print(f"\n🔍 Phân tích Model {engine.upper()}:")
        
        # Tính toán các metrics
        success_rate = len([r for r in engine_results if r.get("status") == "success"]) / len(engine_results)
        avg_time = sum(r.get("execution_time", 0) for r in engine_results) / len(engine_results)
        
        print(f"- Tỷ lệ thành công: {success_rate:.2%}")
        print(f"- Thời gian trung bình: {avg_time:.2f}s")
        
        # Phân tích theo loại prompt
        print("\nHiệu suất theo loại prompt:")
        for prompt_type in ["basic", "cot", "role", "role_cot", "basic_json", "cot_json", "role_json", "role_cot_json"]:
            prompt_results = [r for r in engine_results if r["prompt_type"] == prompt_type]
            if prompt_results:
                prompt_success = len([r for r in prompt_results if r["status"] == "success"])
                prompt_time = sum(r["execution_time"] for r in prompt_results) / len(prompt_results)
                print(f"  {prompt_type:15}: Thành công {prompt_success}/{len(prompt_results)}, "
                      f"Thời gian TB {prompt_time:.2f}s")
        
        # Phân tích chất lượng phản hồi
        print("\nPhân tích chất lượng phản hồi:")
        successful_responses = [r for r in engine_results if r["status"] == "success"]
        if successful_responses:
            response_quality = analyze_response_quality(successful_responses)
            print("- Độ đầy đủ thông tin:")
            for aspect, score in response_quality["completeness"].items():
                print(f"  {aspect:15}: {score:.2f}/5.0")
            print("\n- Độ chính xác:")
            for aspect, score in response_quality["accuracy"].items():
                print(f"  {aspect:15}: {score:.2f}/5.0")
    
    # So sánh giữa các model
    print("\n📊 So sánh giữa các Model:")
    compare_model_responses(all_run_results)

def analyze_response_quality(responses):
    """Đánh giá chất lượng của các phản hồi"""
    quality_metrics = {
        "completeness": {
            "properties": 0.0,
            "analysis": 0.0,
            "recommendations": 0.0
        },
        "accuracy": {
            "technical": 0.0,
            "practical": 0.0,
            "scientific": 0.0
        }
    }
    
    for response in responses:
        content = response.get("response", "")
        
        # Đánh giá độ đầy đủ
        if "properties" in content.lower():
            quality_metrics["completeness"]["properties"] += 5.0
        if "analysis" in content.lower():
            quality_metrics["completeness"]["analysis"] += 5.0
        if "recommend" in content.lower():
            quality_metrics["completeness"]["recommendations"] += 5.0
            
        # Đánh giá độ chính xác
        if any(term in content.lower() for term in ["ev", "bandgap", "conductivity"]):
            quality_metrics["accuracy"]["technical"] += 5.0
        if any(term in content.lower() for term in ["application", "industry", "manufacture"]):
            quality_metrics["accuracy"]["practical"] += 5.0
        if any(term in content.lower() for term in ["mechanism", "theory", "principle"]):
            quality_metrics["accuracy"]["scientific"] += 5.0
    
    # Tính trung bình
    n = len(responses)
    for category in quality_metrics:
        for aspect in quality_metrics[category]:
            quality_metrics[category][aspect] /= n
    
    return quality_metrics

def compare_model_responses(all_results):
    """So sánh phản hồi giữa các model"""
    # Nhóm kết quả theo material_id
    material_results = {}
    for result in all_results:
        if result["status"] == "success":
            mid = result["material_id"]
            if mid not in material_results:
                material_results[mid] = {}
            material_results[mid][result["engine"]] = result["response"]
    
    # So sánh cho từng vật liệu
    for mid, responses in material_results.items():
        if len(responses) > 1:  # Chỉ so sánh khi có cả hai model
            print(f"\nVật liệu {mid}:")
            
            # So sánh độ dài và độ phức tạp
            for engine, response in responses.items():
                response_text = str(response)
                print(f"- {engine:6}: {len(response_text)} ký tự, "
                      f"{len(response_text.split())} từ")
            
            # So sánh nội dung
            common_keywords = find_common_keywords(responses)
            print("\nCác từ khóa chung:")
            for keyword, count in common_keywords.items():
                print(f"  {keyword}: {count} lần")

def list_available_groq_models():
    """Liệt kê các model có sẵn từ Groq API"""
    if not GROQ_API_KEYS or not GROQ_API_KEYS[0]:
        print("⚠️ Chưa cấu hình Groq API key")
        return []
    
    try:
        print("🔍 Đang lấy danh sách model từ Groq API...")
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEYS[0]}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers=headers,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ Không thể lấy danh sách model: {response.status_code} - {response.text}")
            return []
        
        models = response.json().get("data", [])
        print(f"✅ Tìm thấy {len(models)} model từ Groq:")
        for model in models:
            model_id = model.get("id")
            print(f"   - {model_id}")
        
        return [model.get("id") for model in models]
    except Exception as e:
        print(f"❌ Lỗi khi liệt kê model Groq: {str(e)}")
        return []

def list_available_gemini_models():
    """Liệt kê các model có sẵn từ Gemini API"""
    if not GEMINI_API_KEYS or not GEMINI_API_KEYS[0]:
        print("⚠️ Chưa cấu hình Gemini API key")
        return []
    
    try:
        # Cài đặt Google API key
        genai.configure(api_key=GEMINI_API_KEYS[0])
        
        print("🔍 Đang lấy danh sách model từ Gemini API...")
        available_models = []
        
        # Lấy danh sách các models
        try:
            models = genai.list_models()
            available_models = [model.name for model in models if "gemini" in model.name.lower()]
        except Exception as e:
            print(f"⚠️ Không thể lấy danh sách Gemini models: {e}")
            # Thử các model cơ bản
            test_models = ["gemini-1.5-flash-latest", "gemini-1.5-pro", "gemini-pro", "gemini-pro-vision"]
            for model_name in test_models:
                try:
                    model = genai.GenerativeModel(model_name)
                    # Thử model với prompt đơn giản
                    response = model.generate_content("Hello")
                    available_models.append(model_name)
                except Exception:
                    pass
        
        print(f"✅ Tìm thấy {len(available_models)} model từ Gemini:")
        for model_name in available_models:
            print(f"   - {model_name}")
        
        return available_models
    except Exception as e:
        print(f"❌ Lỗi khi liệt kê model Gemini: {str(e)}")
        return []

def test_api_health(api_name, api_key, test_prompt="Hello, this is a test."):
    """Kiểm tra tình trạng hoạt động của API"""
    try:
        print(f"🔍 Kiểm tra sức khỏe API {api_name}...")
        
        if api_name == "gemini":
            # Cấu hình Gemini
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            
            # Gửi yêu cầu đơn giản
            start_time = time.time()
            response = model.generate_content(test_prompt)
            
            # Đo thời gian phản hồi
            response_time = time.time() - start_time
            
            return {
                "status": "healthy" if hasattr(response, "text") else "unhealthy",
                "response_time": response_time,
                "error": None
            }
        
        elif api_name == "groq":
            # Gửi yêu cầu tới Groq API
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "deepseek-r1-distill-llama-70b",
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 10
            }
            
            # Đo thời gian phản hồi
            start_time = time.time()
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions", 
                headers=headers, 
                json=payload, 
                timeout=30
            )
            response_time = time.time() - start_time
            
            # Kiểm tra phản hồi
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time": response_time,
                    "error": None
                }
            else:
                try:
                    error_detail = response.json().get('error', {}).get('message', response.text)
                except:
                    error_detail = response.text
                
                return {
                    "status": "unhealthy",
                    "response_time": response_time,
                    "error": f"HTTP {response.status_code}: {error_detail}"
                }
        
        else:
            return {
                "status": "unknown",
                "response_time": 0,
                "error": f"API {api_name} không được hỗ trợ"
            }
    
    except Exception as e:
        return {
            "status": "error",
            "response_time": 0,
            "error": str(e)
        }

def scan_available_apis(api_rate_manager=None, verbose=True):
    """
    Quét đầy đủ các API khả dụng, kiểm tra tình trạng, models và cập nhật APIRateManager
    
    Args:
        api_rate_manager: Đối tượng APIRateManager để cập nhật
        verbose: In thông tin chi tiết
    
    Returns:
        dict: Thông tin về các API khả dụng
    """
    api_scan_results = {
        "timestamp": datetime.now().isoformat(),
        "apis": {
            "gemini": {
                "status": "not_configured",
                "available_models": [],
                "recommended_model": None,
                "health": {}
            },
            "groq": {
                "status": "not_configured", 
                "available_models": [],
                "recommended_model": None,
                "health": {}
            }
        },
        "summary": {
            "total_healthy_apis": 0,
            "recommended_apis": []
        }
    }
    
    if verbose:
        print("\n🔎 Bắt đầu quét các API khả dụng...")
    
    # Kiểm tra Gemini
    if GEMINI_API_KEYS and GEMINI_API_KEYS[0]:
        # Thử lấy models
        gemini_models = list_available_gemini_models()
        
        # Kiểm tra sức khỏe
        health_check = test_api_health("gemini", GEMINI_API_KEYS[0])
        
        # Cập nhật kết quả
        api_scan_results["apis"]["gemini"]["available_models"] = gemini_models
        api_scan_results["apis"]["gemini"]["health"] = health_check
        
        if gemini_models and health_check["status"] == "healthy":
            api_scan_results["apis"]["gemini"]["status"] = "available"
            api_scan_results["apis"]["gemini"]["recommended_model"] = "gemini-1.5-flash-latest"
            api_scan_results["summary"]["total_healthy_apis"] += 1
            api_scan_results["summary"]["recommended_apis"].append("gemini")
            
            # Cập nhật api_rate_manager
            if api_rate_manager and "gemini" in api_rate_manager.api_states:
                # Tìm model flash mới nhất
                flash_models = [m for m in gemini_models if "flash" in m.lower()]
                if flash_models:
                    recommended_model = flash_models[0]
                    if verbose:
                        print(f"✅ Cập nhật Gemini model: {recommended_model}")
                    
                    api_rate_manager.api_states["gemini"]["apis"] = [recommended_model]
                    api_rate_manager.api_states["gemini"]["limits"] = {
                        recommended_model: {"requests": 60, "window": 60}
                    }
                    api_rate_manager._init_usage_tracking()
        else:
            api_scan_results["apis"]["gemini"]["status"] = "unavailable"
            if verbose:
                print(f"⚠️ API Gemini không khả dụng: {health_check.get('error', 'Unknown error')}")
    
    # Kiểm tra Groq
    if GROQ_API_KEYS and GROQ_API_KEYS[0]:
        # Thử lấy models
        groq_models = list_available_groq_models()
        
        # Kiểm tra sức khỏe
        health_check = test_api_health("groq", GROQ_API_KEYS[0])
        
        # Cập nhật kết quả
        api_scan_results["apis"]["groq"]["available_models"] = groq_models
        api_scan_results["apis"]["groq"]["health"] = health_check
        
        if groq_models and health_check["status"] == "healthy":
            api_scan_results["apis"]["groq"]["status"] = "available"
            
            # Ưu tiên model deepseek-r1-distill-llama-70b
            if "deepseek-r1-distill-llama-70b" in groq_models:
                recommended_model = "deepseek-r1-distill-llama-70b"
            # Nếu không có deepseek, tìm model khác
            else:
                recommended_model = groq_models[0]

            api_scan_results["apis"]["groq"]["recommended_model"] = recommended_model
            api_scan_results["summary"]["total_healthy_apis"] += 1
            api_scan_results["summary"]["recommended_apis"].append("groq")
            
            # Cập nhật api_rate_manager
            if api_rate_manager and "groq" in api_rate_manager.api_states:
                if verbose:
                    print(f"✅ Cập nhật Groq model: {recommended_model}")
                
                api_rate_manager.api_states["groq"]["apis"] = [recommended_model]
                api_rate_manager.api_states["groq"]["limits"] = {
                    recommended_model: {"requests": 50, "window": 60}
                }
                api_rate_manager._init_usage_tracking()
        else:
            api_scan_results["apis"]["groq"]["status"] = "unavailable"
            if verbose:
                print(f"⚠️ API Groq không khả dụng: {health_check.get('error', 'Unknown error')}")
    
    # In tóm tắt
    if verbose:
        print("\n📊 Kết quả quét API:")
        for api_name, api_info in api_scan_results["apis"].items():
            status_emoji = "✅" if api_info["status"] == "available" else "❌"
            print(f"{status_emoji} {api_name.upper()}: {api_info['status']}")
            
            if api_info["status"] == "available":
                print(f"   - Model đề xuất: {api_info['recommended_model']}")
                print(f"   - Thời gian phản hồi: {api_info['health'].get('response_time', 'N/A'):.2f}s")
        
        print(f"\n🏆 Tổng số API khả dụng: {api_scan_results['summary']['total_healthy_apis']}")
        if api_scan_results["summary"]["recommended_apis"]:
            print(f"📑 Đề xuất sử dụng: {', '.join(api_scan_results['summary']['recommended_apis']).upper()}")
    
    return api_scan_results

# Thêm hàm tiện ích để quét định kỳ trong quá trình xử lý
def setup_periodic_api_scanning(api_rate_manager, interval_minutes=30):
    """
    Tạo một thread chạy nền để quét API định kỳ
    
    Args:
        api_rate_manager: Đối tượng APIRateManager để cập nhật
        interval_minutes: Khoảng thời gian giữa các lần quét (phút)
    
    Returns:
        threading.Thread: Thread đang chạy nền
    """
    def scanning_worker():
        while True:
            try:
                scan_available_apis(api_rate_manager, verbose=False)
                print(f"🔄 Quét API tự động hoàn tất, quét tiếp theo sau {interval_minutes} phút")
                time.sleep(interval_minutes * 60)
            except Exception as e:
                print(f"⚠️ Lỗi trong quá trình quét API tự động: {e}")
                time.sleep(300)  # Đợi 5 phút nếu có lỗi
    
    # Tạo và khởi động thread
    scanner_thread = threading.Thread(target=scanning_worker, daemon=True)
    scanner_thread.start()
    print(f"🔄 Đã khởi động quét API định kỳ (mỗi {interval_minutes} phút)")
    
    return scanner_thread

# --- Thêm hàm tạo biểu đồ ---
def generate_visualizations(material_scores, output_base_name):
    """Tạo các biểu đồ trực quan để hiển thị kết quả phân tích"""
    # Đảm bảo biến VISUALIZATION_AVAILABLE được khai báo toàn cục
    global VISUALIZATION_AVAILABLE
    
    if not VISUALIZATION_AVAILABLE:
        print("⚠️ Bỏ qua tạo biểu đồ do thiếu thư viện matplotlib")
        return False
    
    if not material_scores:
        print("⚠️ Không có dữ liệu để tạo biểu đồ")
        return False
    
    print("\n📊 Đang tạo biểu đồ trực quan...")
    
    try:
        # Đảm bảo thư mục tồn tại
        vis_dir = os.path.join(os.path.dirname(output_base_name), "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Chuẩn bị dữ liệu
        materials = []
        scores = []
        bandgaps = []
        conductivities = []
        
        for material_id, data in material_scores.items():
            materials.append(data.get('material_name', material_id))
            scores.append(data.get('score', {}).get('score', 0))
            
            try:
                bandgap = float(data.get('properties', {}).get('bandgap', 0))
                bandgaps.append(bandgap)
            except (ValueError, TypeError):
                bandgaps.append(0)
                
            try:
                conductivity = float(data.get('properties', {}).get('conductivity', 0))
                conductivities.append(conductivity)
            except (ValueError, TypeError):
                conductivities.append(0)
        
        # 1. Biểu đồ điểm đánh giá tổng hợp
        plt.figure(figsize=(10, 6))
        bars = plt.bar(materials, scores, color='skyblue')
        
        # Thêm giá trị lên đầu mỗi cột
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom')
            
        plt.title('Điểm Đánh Giá Vật Liệu Bán Dẫn')
        plt.xlabel('Vật Liệu')
        plt.ylabel('Điểm (0-100)')
        plt.ylim(0, 110)  # Giới hạn trục y
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(vis_dir, 'material_scores.png'), dpi=300)
        
        # 2. Biểu đồ so sánh các thuộc tính (scatter plot)
        plt.figure(figsize=(10, 6))
        
        # Tạo một colormap tùy chỉnh
        colors = [(0.1, 0.1, 0.9), (0.9, 0.1, 0.1)]  # Từ xanh đến đỏ
        cmap_name = 'score_cmap'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
        
        normalized_scores = np.array(scores) / 100.0  # Chuẩn hóa về 0-1
        
        scatter = plt.scatter(bandgaps, conductivities, 
                             c=normalized_scores, cmap=cm, 
                             s=100, alpha=0.7, edgecolors='k')
        
        # Thêm nhãn cho mỗi điểm
        for i, material in enumerate(materials):
            plt.annotate(material, (bandgaps[i], conductivities[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.title('So Sánh Bandgap và Độ Dẫn Điện')
        plt.xlabel('Bandgap (eV)')
        plt.ylabel('Độ dẫn điện (S/m)')
        plt.grid(linestyle='--', alpha=0.7)
        plt.colorbar(scatter, label='Điểm đánh giá (chuẩn hóa)')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'property_comparison.png'), dpi=300)
        
        # 3. Biểu đồ radar cho 5 vật liệu đầu tiên (nếu có đủ)
        if len(materials) >= 1:
            # Giới hạn số vật liệu để biểu đồ không quá rối
            max_materials = min(5, len(materials))
            
            # Chuẩn bị dữ liệu cho radar chart
            properties = ['Bandgap', 'Conductivity', 'Score', 'Thermal Stability']
            
            # Số lượng thuộc tính
            N = len(properties)
            
            # Góc cho mỗi trục
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Khép kín đường
            
            # Vẽ biểu đồ radar
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            # Tạo các màu phân biệt
            colors = plt.cm.tab10(np.linspace(0, 1, max_materials))
            
            # Lặp qua các vật liệu để vẽ
            for i in range(max_materials):
                # Chuẩn hóa dữ liệu để hiển thị trên biểu đồ radar
                material_data = []
                
                # Bandgap (chuẩn hóa nằm trong khoảng 0-1)
                bandgap_norm = min(bandgaps[i] / 5.0, 1.0)  # Giả sử max bandgap ~5eV
                material_data.append(bandgap_norm)
                
                # Conductivity (chuẩn hóa)
                conductivity_norm = min(conductivities[i] / 5000.0, 1.0)  # Giả sử max conductivity ~5000 S/m
                material_data.append(conductivity_norm)
                
                # Score (đã chuẩn hóa)
                score_norm = scores[i] / 100.0
                material_data.append(score_norm)
                
                # Thermal stability (lấy từ điểm đánh giá nếu có)
                thermal_score = 0
                for material_id, data in material_scores.items():
                    if data.get('material_name') == materials[i]:
                        breakdown = data.get('score', {}).get('breakdown', {})
                        thermal_text = breakdown.get('thermal_stability', '')
                        if 'high' in thermal_text.lower() or 'cao' in thermal_text.lower():
                            thermal_score = 1.0
                        elif 'medium' in thermal_text.lower() or 'trung' in thermal_text.lower():
                            thermal_score = 0.7
                        elif 'low' in thermal_text.lower() or 'thấp' in thermal_text.lower():
                            thermal_score = 0.3
                        break
                
                material_data.append(thermal_score)
                
                # Khép kín đường
                material_data += material_data[:1]
                
                # Vẽ dữ liệu
                ax.plot(angles, material_data, linewidth=2, color=colors[i], label=materials[i])
                ax.fill(angles, material_data, color=colors[i], alpha=0.25)
            
            # Thiết lập trục và nhãn
            ax.set_theta_offset(np.pi / 2)  # Bắt đầu từ trên cùng
            ax.set_theta_direction(-1)  # Theo chiều kim đồng hồ
            
            # Đặt nhãn cho các trục
            plt.xticks(angles[:-1], properties)
            
            # Đặt giới hạn y
            ax.set_ylim(0, 1)
            
            # Thêm chú thích
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.title('So sánh đa thuộc tính của các vật liệu')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'radar_comparison.png'), dpi=300)
        
        plt.close('all')  # Đóng tất cả biểu đồ
        print(f"✅ Đã tạo các biểu đồ trong thư mục: {vis_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi tạo biểu đồ: {e}")
        return False

# --- Thêm hàm xử lý batch cho nhiều vật liệu ---
def process_materials_batch(materials_batch, prompt_variants_config, active_engines, pdf_chunks_data, sleep_between=0, search_engine=None, api_rate_manager=None, response_cache=None):
    """Xử lý một batch các vật liệu để tối ưu hóa việc gọi API"""
    material_results = []
    
    print(f"\n🔄 Xử lý batch {len(materials_batch)} vật liệu...")
    
    for idx, material_data in enumerate(materials_batch):
        # Sử dụng material_id từ CSV, hoặc tạo một ID nếu không có
        material_id = material_data.get('material_id', f'Unknown-{idx+1}')
        material_name = material_data.get('material_name', 'Unknown Material')
        print(f"   - Vật liệu: {material_name} (ID: {material_id})")
        
        # Tìm các đoạn PDF liên quan
        retrieved_chunks = find_relevant_chunks(material_data.get('material_name', ''), pdf_chunks_data, num_chunks=3, search_engine=search_engine)
        retrieved_context = "\n".join(retrieved_chunks) if retrieved_chunks else "Không tìm thấy thông tin tham khảo."
        
        # Tạo chuỗi chi tiết từ dữ liệu vật liệu
        details_string = format_material_details(material_data)
        
        material_item_results = []
        
        for prompt_name, config in prompt_variants_config.items():
            print(f"     Prompt: '{prompt_name}'")
            
            for engine_name, engine_instance in active_engines:
                try:
                    # Với Groq, giảm context
                    current_retrieved_context = retrieved_context
                    if engine_name == "groq" and len(retrieved_context) > 1000:
                        current_retrieved_context = retrieved_context[:1000] + "...\n[Context bị cắt bớt]"
                    
                    # Tạo prompt với hàm mới, sử dụng fallback nếu cần
                    try:
                        # Ưu tiên sử dụng định dạng prompt mới
                        if "prompt_type" in config and "output_format" in config:
                            current_prompt = get_prompt(
                                prompt_type=config["prompt_type"],
                                output_format=config["output_format"],
                                material_data=material_data,
                                retrieved_context=current_retrieved_context,
                                details_string=details_string,
                                material_name=material_data.get('material_name', 'Unknown'),
                                crystal_structure=material_data.get('crystal_structure', 'Unknown'),
                                bandgap_energy=float(material_data.get('bandgap', 0)),
                                target_application_potential=material_data.get('target_application_potential', 'Not specified')
                            )
                        # Fallback sang định dạng cũ nếu cần
                        elif "category" in config and "key" in config:
                            print(f"     ⚠️ Sử dụng format prompt cũ cho {prompt_name}")
                            current_prompt = legacy_get_prompt(
                                category=config["category"],
                                key=config["key"],
                                add_cot=config.get("add_cot", False),
                                role=config.get("role", None),
                                material_data=material_data,
                                retrieved_context=current_retrieved_context,
                                details_string=details_string,
                                material_name=material_data.get('material_name', 'Unknown'),
                                crystal_structure=material_data.get('crystal_structure', 'Unknown'),
                                bandgap_energy=float(material_data.get('bandgap', 0)),
                                target_application_potential=material_data.get('target_application_potential', 'Not specified')
                            )
                        else:
                            raise ValueError(f"Cấu hình prompt không hợp lệ: {config}")
                    except Exception as e:
                        print(f"     ⚠️ Lỗi khi tạo prompt: {e}")
                        # Sử dụng cách tạo prompt đơn giản nhất nếu cả hai phương pháp đều không hoạt động
                        current_prompt = f"""Phân tích vật liệu {material_data.get('material_name', 'Unknown')} với:
- Cấu trúc tinh thể: {material_data.get('crystal_structure', 'Unknown')}
- Bandgap: {material_data.get('bandgap', 0)} eV
- Ứng dụng mục tiêu: {material_data.get('target_application_potential', 'Not specified')}

Chi tiết:
{details_string}

Vui lòng phân tích và đánh giá vật liệu này."""
                    
                    # Lấy api_version dựa trên api_rate_manager
                    if api_rate_manager and engine_name in api_rate_manager.api_states:
                        api_version = api_rate_manager.api_states[engine_name]["apis"][0]
                    else:
                        api_version = "default"
                    
                    result = analyze_with_engine(
                        engine_name, 
                        engine_instance, 
                        current_prompt,
                        api_version,
                        api_rate_manager,
                        response_cache
                    )
                    
                    # Thêm metadata
                    result.update({
                        "prompt_type": prompt_name,
                        "material_id": material_id,
                        "material_name": material_name,
                        "retrieved_context_summary": f"{len(retrieved_chunks)} chunks found"
                    })
                    
                    material_item_results.append(result)
                    
                except Exception as e:
                    print(f"     ❌ Lỗi với {engine_name}: {e}")
                    error_result = {
                        "prompt_type": prompt_name,
                        "material_id": material_id,
                        "material_name": material_name,
                        "engine": engine_name,
                        "status": "error",
                        "error": str(e),
                        "execution_time": 0
                    }
                    material_item_results.append(error_result)
        
        # Tính điểm đánh giá cho vật liệu
        material_score = calculate_material_score(material_data, material_item_results)
        
        # Thêm kết quả vào danh sách kết quả chung
        material_results.append({
            "material_data": material_data,
            "material_id": material_id,
            "material_name": material_name,
            "analysis_results": material_item_results,
            "material_score": material_score
        })
        
        if sleep_between > 0 and idx < len(materials_batch) - 1:
            print(f"   ⏳ Chờ {sleep_between} giây...")
            time.sleep(sleep_between)
    
    return material_results

def compare_prompt_types_depth(all_run_results):
    """
    So sánh các loại prompt thông qua các tiêu chí đánh giá chuyên sâu.
    
    Tiêu chí đánh giá bao gồm:
    1. Độ sâu lập luận
    2. Độ chính xác
    3. Tính toàn diện
    4. Cấu trúc và tổ chức
    5. Liên quan đến ứng dụng
    6. Insights sáng tạo
    7. Tính khoa học
    8. Tính nhất quán
    
    Args:
        all_run_results: Danh sách các kết quả chạy
        
    Returns:
        Dictionary chứa kết quả đánh giá chi tiết
    """
    if not all_run_results:
        print("\n⚠️ Không có kết quả để phân tích.")
        return {}
        
    print("\n=== SO SÁNH SÂU CÁC LOẠI PROMPT ===")
    
    # Tất cả các loại prompt
    all_prompt_types = ["basic", "cot", "role", "role_cot", 
                        "basic_json", "cot_json", "role_json", "role_cot_json"]
    
    # Khởi tạo từ điển kết quả
    evaluation_results = {prompt_type: {} for prompt_type in all_prompt_types}
    
    # Khởi tạo điểm số theo tiêu chí
    criteria = {
        "reasoning_depth": "Độ sâu lập luận",
        "factual_accuracy": "Độ chính xác",
        "comprehensiveness": "Tính toàn diện", 
        "structure": "Cấu trúc và tổ chức",
        "application_relevance": "Liên quan đến ứng dụng",
        "creative_insights": "Insights sáng tạo",
        "scientific_soundness": "Tính khoa học",
        "consistency": "Tính nhất quán"
    }
    
    # Nhóm kết quả theo loại prompt
    grouped_by_prompt = {}
    for prompt_type in all_prompt_types:
        grouped_by_prompt[prompt_type] = [r for r in all_run_results 
                                        if r.get("prompt_type") == prompt_type 
                                        and r.get("status") == "success"]
    
    # Khởi tạo kết quả đánh giá
    for prompt_type in all_prompt_types:
        evaluation_results[prompt_type] = {
            "count": len(grouped_by_prompt[prompt_type]),
            "scores": {criterion: 0.0 for criterion in criteria}
        }
    
    # Đánh giá từng loại prompt
    for prompt_type, responses in grouped_by_prompt.items():
        if not responses:
            continue
            
        scores = evaluation_results[prompt_type]["scores"]
        
        # Đánh giá các tiêu chí
        for response in responses:
            content = str(response.get("response", "")).lower()
            
            # 1. Độ sâu lập luận - cao nhất trong các loại có CoT
            if "cot" in prompt_type:
                scores["reasoning_depth"] += 5.0  # Thêm điểm cho CoT
            elif "role" in prompt_type:
                scores["reasoning_depth"] += 3.0  # Thêm điểm cho Role-playing
            else:
                scores["reasoning_depth"] += 2.0  # Điểm cơ bản
                
            # Tăng thêm điểm nếu có phân tích chi tiết
            if "because" in content or "do đó" in content or "vì" in content:
                scores["reasoning_depth"] += 1.0
            if "mechanism" in content or "cơ chế" in content:
                scores["reasoning_depth"] += 1.0
                
            # 2. Độ chính xác - dựa trên các thuật ngữ kỹ thuật
            if any(term in content for term in ["bandgap", "ev", "độ rộng vùng cấm"]):
                scores["factual_accuracy"] += 1.0
            if any(term in content for term in ["crystal structure", "cấu trúc tinh thể"]):
                scores["factual_accuracy"] += 1.0
            if any(term in content for term in ["conductivity", "độ dẫn điện"]):
                scores["factual_accuracy"] += 1.0
            if any(term in content for term in ["thermal stability", "độ bền nhiệt"]):
                scores["factual_accuracy"] += 1.0
            if any(term in content for term in ["application", "ứng dụng"]):
                scores["factual_accuracy"] += 1.0
                
            # 3. Tính toàn diện - dựa trên số lượng phần trong phân tích
            sections = 0
            for section in ["physical properties", "thuộc tính vật lý", 
                           "applications", "ứng dụng", 
                           "comparison", "so sánh",
                           "recommendations", "khuyến nghị"]:
                if section in content:
                    sections += 1
            scores["comprehensiveness"] += min(5.0, sections)
                
            # 4. Cấu trúc và tổ chức - cao nhất trong JSON
            if "json" in prompt_type:
                scores["structure"] += 5.0  # JSON có cấu trúc rõ ràng nhất
            elif any(marker in content for marker in ["##", "heading", "section", "phần"]):
                scores["structure"] += 3.0  # Có tiêu đề và phân đoạn
            else:
                scores["structure"] += 1.0  # Điểm cơ bản
                
            # 5. Liên quan đến ứng dụng
            app_mentions = 0
            for app in ["solar", "pin mặt trời", "transistor", "led", "sensor", 
                       "cảm biến", "electronics", "điện tử"]:
                if app in content:
                    app_mentions += 1
            scores["application_relevance"] += min(5.0, app_mentions)
                
            # 6. Insights sáng tạo
            if "novel" in content or "mới" in content:
                scores["creative_insights"] += 1.0
            if "future" in content or "tương lai" in content:
                scores["creative_insights"] += 1.0
            if "potential" in content or "tiềm năng" in content:
                scores["creative_insights"] += 1.0
            if "improve" in content or "cải thiện" in content:
                scores["creative_insights"] += 1.0
            if "unique" in content or "độc đáo" in content:
                scores["creative_insights"] += 1.0
                
            # 7. Tính khoa học
            if "mechanism" in content or "cơ chế" in content:
                scores["scientific_soundness"] += 1.0
            if "theory" in content or "lý thuyết" in content:
                scores["scientific_soundness"] += 1.0
            if "principle" in content or "nguyên lý" in content:
                scores["scientific_soundness"] += 1.0
            if "research" in content or "nghiên cứu" in content:
                scores["scientific_soundness"] += 1.0
            if "literature" in content or "tài liệu" in content:
                scores["scientific_soundness"] += 1.0
                
            # 8. Tính nhất quán - điểm cao nhất cho JSON
            if "json" in prompt_type:
                scores["consistency"] += 5.0
            elif "role_cot" in prompt_type:
                scores["consistency"] += 4.0
            elif "cot" in prompt_type or "role" in prompt_type:
                scores["consistency"] += 3.0
            else:
                scores["consistency"] += 2.0
        
        # Tính điểm trung bình cho mỗi tiêu chí
        if responses:
            for criterion in criteria:
                scores[criterion] /= len(responses)
                # Chuẩn hóa về thang điểm 5.0
                scores[criterion] = min(5.0, scores[criterion])
    
    # In bảng so sánh
    print("\n📊 Bảng đánh giá các prompt theo tiêu chí (thang điểm 5.0):")
    print("-" * 80)
    
    # In header
    header = "Tiêu chí đánh giá".ljust(25)
    for prompt_type in all_prompt_types:
        if evaluation_results[prompt_type]["count"] > 0:
            header += prompt_type.ljust(10)
    print(header)
    print("-" * 80)
    
    # In điểm số từng tiêu chí
    for criterion, criterion_name in criteria.items():
        row = criterion_name.ljust(25)
        for prompt_type in all_prompt_types:
            if evaluation_results[prompt_type]["count"] > 0:
                score = evaluation_results[prompt_type]["scores"][criterion]
                row += f"{score:.1f}".ljust(10)
        print(row)
    
    print("-" * 80)
    
    # Tính điểm tổng hợp cho mỗi loại prompt
    print("\nĐiểm tổng hợp:")
    for prompt_type in all_prompt_types:
        if evaluation_results[prompt_type]["count"] > 0:
            scores = evaluation_results[prompt_type]["scores"]
            avg_score = sum(scores.values()) / len(scores)
            # Gán điểm tổng hợp
            evaluation_results[prompt_type]["overall_score"] = avg_score
            print(f"- {prompt_type}: {avg_score:.2f}/5.0")
    
    # Xác định prompt hiệu quả nhất
    valid_prompts = {pt: evaluation_results[pt] for pt in all_prompt_types 
                    if evaluation_results[pt].get("count", 0) > 0}
    
    if valid_prompts:
        best_prompt = max(valid_prompts.items(), 
                        key=lambda x: x[1].get("overall_score", 0))
        print(f"\n🏆 Prompt hiệu quả nhất: {best_prompt[0]} (điểm: {best_prompt[1]['overall_score']:.2f}/5.0)")
    
    return evaluation_results

# Thêm hàm trực quan hóa cho kết quả so sánh prompt
def generate_prompt_comparison_visualizations(prompt_comparison_results, output_base_name):
    """
    Tạo biểu đồ trực quan hóa so sánh các loại prompt.
    
    Args:
        prompt_comparison_results: Kết quả đánh giá các loại prompt
        output_base_name: Tên cơ sở cho file đầu ra
    """
    if not VISUALIZATION_AVAILABLE:
        print("⚠️ Matplotlib không khả dụng. Không thể tạo visualizations.")
        return
        
    if not prompt_comparison_results:
        print("⚠️ Không có dữ liệu so sánh prompt để trực quan hóa.")
        return
        
    # Lọc ra các prompt có dữ liệu
    valid_prompts = {k: v for k, v in prompt_comparison_results.items() if v.get("count", 0) > 0}
    if not valid_prompts:
        return
        
    # Lấy danh sách các tiêu chí
    criteria = list(next(iter(valid_prompts.values()))["scores"].keys())
    
    # Tạo heatmap so sánh các loại prompt
    plt.figure(figsize=(12, 8))
    
    # Chuẩn bị data cho heatmap
    prompt_types = list(valid_prompts.keys())
    data = []
    
    for criterion in criteria:
        row = []
        for prompt_type in prompt_types:
            row.append(valid_prompts[prompt_type]["scores"][criterion])
        data.append(row)
    
    # Tạo heatmap
    heatmap = plt.imshow(data, cmap="YlGnBu", aspect="auto")
    plt.colorbar(heatmap, label="Điểm (0-5)")
    
    # Labels
    plt.yticks(range(len(criteria)), [c.replace("_", " ").title() for c in criteria])
    plt.xticks(range(len(prompt_types)), prompt_types, rotation=45)
    
    plt.title("So sánh các loại Prompt theo tiêu chí đánh giá")
    plt.tight_layout()
    
    # Lưu biểu đồ
    heatmap_file = f"{output_base_name}_prompt_comparison_heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
    print(f"✅ Đã lưu biểu đồ so sánh prompt vào {heatmap_file}")
    plt.close()
    
    # Tạo biểu đồ radar cho so sánh tổng thể
    plt.figure(figsize=(10, 10))
    
    # Chuẩn bị dữ liệu radar chart
    angles = np.linspace(0, 2*np.pi, len(criteria), endpoint=False).tolist()
    angles += angles[:1]  # Đóng vòng tròn
    
    # Thiết lập radar chart
    ax = plt.subplot(111, polar=True)
    
    # Thêm các nhãn trục
    plt.xticks(angles[:-1], [c.replace("_", " ").title() for c in criteria])
    
    # Thêm dữ liệu cho từng loại prompt
    for i, prompt_type in enumerate(prompt_types):
        values = [valid_prompts[prompt_type]["scores"][criterion] for criterion in criteria]
        values += values[:1]  # Đóng vòng tròn
        
        ax.plot(angles, values, linewidth=2, label=prompt_type)
        ax.fill(angles, values, alpha=0.1)
    
    plt.title("Biểu đồ Radar So sánh các loại Prompt")
    plt.legend(loc="upper right")
    
    # Lưu biểu đồ
    radar_file = f"{output_base_name}_prompt_comparison_radar.png"
    plt.savefig(radar_file, dpi=300, bbox_inches="tight")
    print(f"✅ Đã lưu biểu đồ radar so sánh prompt vào {radar_file}")
    plt.close()
    
    # Tạo biểu đồ cột so sánh điểm tổng hợp
    plt.figure(figsize=(10, 6))
    
    # Tính điểm tổng hợp
    overall_scores = [valid_prompts[pt].get("overall_score", 0) for pt in prompt_types]
    
    # Tạo màu theo điểm
    colors = plt.cm.YlGnBu(np.array(overall_scores) / 5.0)
    
    # Vẽ biểu đồ cột
    bars = plt.bar(prompt_types, overall_scores, color=colors)
    
    # Thêm nhãn giá trị trên đầu cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.ylim(0, 5.5)  # Giới hạn trục y
    plt.xlabel("Loại prompt")
    plt.ylabel("Điểm tổng hợp (0-5)")
    plt.title("So sánh điểm tổng hợp các loại Prompt")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Lưu biểu đồ
    bar_file = f"{output_base_name}_prompt_comparison_scores.png"
    plt.savefig(bar_file, dpi=300, bbox_inches="tight")
    print(f"✅ Đã lưu biểu đồ cột so sánh prompt vào {bar_file}")
    plt.close()

# --- Hàm phân tích vật liệu song song với Gemini và DeepSeek ---
def analyze_materials_parallel(num_samples=20):
    """
    Phân tích vật liệu song song với Gemini và DeepSeek
    
    Args:
        num_samples: Số lượng mẫu vật liệu cần phân tích
        
    Returns:
        list: Danh sách kết quả phân tích
    """
    print(f"🔍 Bắt đầu phân tích {num_samples} mẫu vật liệu với Gemini và DeepSeek...")
    
    # Đường dẫn đến file dữ liệu vật liệu
    csv_path = "data/raw/documents/fake_materials_dataset.csv"
    
    # Khởi tạo cache và rate manager
    response_cache = ResponseCache(cache_dir="cache")
    api_rate_manager = APIRateManager()
    
    # Khởi tạo search engine
    search_engine = SemanticSearchEngine()
    
    # Quét API để xác định models khả dụng
    print("\n🔍 Quét API để xác định models khả dụng...")
    scan_available_apis(api_rate_manager, verbose=True)
    
    # Sử dụng cả Gemini và Groq (DeepSeek)
    engines_to_use = ["gemini", "groq"]
    print(f"\n🚀 Engines sẽ sử dụng: {engines_to_use}")
    
    # Khởi tạo engines
    active_engines = []
    for engine_name in engines_to_use:
        engine_tuple = init_chat_engine(engine_name)
        if engine_tuple[1] is not None or engine_name == "groq":  # Groq không cần chat object
            active_engines.append(engine_tuple)
    
    # Kiểm tra engines khả dụng
    if not active_engines:
        print("❌ Không có engine nào khả dụng. Vui lòng kiểm tra API keys.")
        return []
    
    print(f"✅ Đã kích hoạt {len(active_engines)} engines: {[e[0] for e in active_engines]}")
    
    # Tải dữ liệu vật liệu
    try:
        all_materials = load_material_data(csv_path)
        print(f"📄 Đã tải {len(all_materials)} mẫu vật liệu từ {csv_path}")
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
        return []
    
    # Giới hạn số lượng mẫu
    materials_sample = all_materials[:num_samples]
    print(f"🔍 Xử lý {len(materials_sample)} mẫu đầu tiên...")
    
    # Cấu hình prompt
    prompt_config = {"basic": {"category": "task", "key": "analyze_material"}}
    
    # Xử lý theo batch
    batch_size = int(os.getenv("BATCH_SIZE", "2"))
    
    # Chia thành các batch
    batches = []
    for i in range(0, len(materials_sample), batch_size):
        batches.append(materials_sample[i:i+batch_size])
    
    all_results = []
    for batch_idx, batch in enumerate(batches):
        print(f"\n🔄 Xử lý batch {batch_idx+1}/{len(batches)} ({len(batch)} mẫu)...")
        
        # Xử lý batch
        batch_results = process_materials_batch(
            batch,
            prompt_config,
            active_engines,
            [],  # Không có PDF chunks
            sleep_between=int(os.getenv("SLEEP_BETWEEN_RECORDS", "1")),
            search_engine=search_engine,
            api_rate_manager=api_rate_manager,
            response_cache=response_cache
        )
        
        all_results.extend(batch_results)
    
    # In kết quả
    print(f"\n✅ Đã phân tích xong {len(all_results)} mẫu vật liệu")
    
    # Lưu kết quả vào file JSON
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(output_dir, f"combined_analysis_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Kết quả đã được lưu tại: {output_file}")
    
    # In tóm tắt các kết quả
    print("\n🔍 Tóm tắt kết quả phân tích song song Gemini và DeepSeek:")
    for idx, result in enumerate(all_results):
        material_name = result["material_name"]
        material_score = result["material_score"]
        
        print(f"\n{idx+1}. {material_name}:")
        # Kiểm tra nếu material_score là dictionary
        if isinstance(material_score, dict):
            if "score" in material_score:
                print(f"   - Điểm đánh giá: {material_score['score']:.2f}/100")
            else:
                print(f"   - Điểm đánh giá: {material_score}")
        else:
            print(f"   - Điểm đánh giá: {material_score:.2f}/100")
        
        # Hiển thị tóm tắt từ mỗi engine
        for analysis in result["analysis_results"]:
            engine = analysis.get("engine", "Unknown")
            status = analysis.get("status", "Unknown")
            
            if status == "success":
                response = analysis.get("response", "")
                # Chỉ lấy dòng đầu tiên làm tóm tắt
                summary = response.split("\n")[0][:100] + "..." if response else "Không có phản hồi"
                print(f"   - [{engine.upper()}]: {summary}")
            else:
                print(f"   - [{engine.upper()}]: ❌ Lỗi: {analysis.get('error', 'Unknown error')}")
    
    print(f"\n📊 Tổng hợp đầy đủ có trong file: {output_file}")
    
    return all_results

# --- Hàm gộp tất cả các tính năng ---
def analyze_materials_unified(num_samples=20, output_detailed=True):
    """
    Hàm gộp tất cả các tính năng: quét API, phân tích song song, phân tích đầy đủ
    
    Args:
        num_samples: Số lượng mẫu vật liệu cần phân tích
        output_detailed: Có tạo báo cáo chi tiết hay không
        
    Returns:
        list: Danh sách kết quả phân tích
    """
    print(f"🔍 Bắt đầu phân tích thống nhất với {num_samples} mẫu vật liệu")
    
    # Khởi tạo các đối tượng cần thiết
    response_cache = ResponseCache(cache_dir=os.getenv("CACHE_DIRECTORY", "cache"))
    api_rate_manager = APIRateManager()
    search_engine = SemanticSearchEngine()
    
    # Quét API để xác định models khả dụng
    print("\n🔎 Quét API để xác định models khả dụng...")
    api_scan_results = scan_available_apis(api_rate_manager, verbose=True)
    
    # Xác định các engines sẽ sử dụng dựa trên cấu hình hoặc kết quả quét
    use_auto_detection = os.getenv("USE_AUTO_ENGINE_DETECTION", "True").lower() == "true"
    
    if use_auto_detection:
        # Sử dụng kết quả quét tự động
        engines_to_use = api_scan_results["summary"]["recommended_apis"]
        if not engines_to_use:
            print("⚠️ Không tìm thấy API khả dụng qua quét tự động, kiểm tra cài đặt thủ công")
            engines_from_env = os.getenv("ENGINES_TO_USE", "gemini,groq").lower().split(",")
            engines_to_use = [engine.strip() for engine in engines_from_env if engine.strip()]
    else:
        # Sử dụng cấu hình thủ công từ file .env
        engines_from_env = os.getenv("ENGINES_TO_USE", "gemini,groq").lower().split(",")
        engines_to_use = [engine.strip() for engine in engines_from_env if engine.strip()]
        print("ℹ️ Sử dụng cấu hình engines từ file .env")
    
    # Đảm bảo danh sách không rỗng
    if not engines_to_use:
        print("⚠️ Không tìm thấy API khả dụng, mặc định sử dụng Gemini + DeepSeek")
        engines_to_use = ["gemini", "groq"]
    
    print(f"\n🤖 Các API engines sẽ được sử dụng: {', '.join(engines_to_use).upper()}")
    
    # Cấu hình quét API định kỳ
    API_SCAN_INTERVAL_MINUTES = int(os.getenv("API_SCAN_INTERVAL_MINUTES", "15"))
    ENABLE_PERIODIC_API_SCAN = os.getenv("ENABLE_PERIODIC_API_SCAN", "True").lower() == "true"
    if ENABLE_PERIODIC_API_SCAN:
        api_scanner_thread = setup_periodic_api_scanning(
            api_rate_manager,
            interval_minutes=API_SCAN_INTERVAL_MINUTES
        )
    
    # Cấu hình đường dẫn và tệp
    pdf_document_directory = "data/raw/documents"
    material_csv_file = os.path.join(pdf_document_directory, "fake_materials_dataset.csv")
    
    # Tạo thư mục results nếu chưa tồn tại
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"📁 Tạo thư mục kết quả: {results_dir}")
        os.makedirs(results_dir, exist_ok=True)

    # Tải và xử lý tài liệu PDF
    print("\n📚 Đang tải và xử lý tài liệu PDF...")
    
    # Kiểm tra thư mục PDF tồn tại
    if not os.path.exists(pdf_document_directory):
        print(f"⚠️ Không tìm thấy thư mục PDF: {pdf_document_directory}")
        print(f"📁 Tạo thư mục: {pdf_document_directory}")
        os.makedirs(pdf_document_directory, exist_ok=True)
    
    # Kiểm tra file CSV tồn tại
    if not os.path.exists(material_csv_file):
        print(f"⚠️ Không tìm thấy file CSV: {material_csv_file}")
        print("❓ Vui lòng đặt file CSV vào thư mục data/raw/documents")
        fake_csv = input("🔄 Tạo dữ liệu mẫu? (y/n): ")
        if fake_csv.lower() == 'y':
            create_fake_material_csv(material_csv_file, num_materials=num_samples)
        else:
            print("❌ Không thể tiếp tục mà không có dữ liệu. Thoát.")
            return []

    # Tải dữ liệu PDF
    pdf_chunks = load_and_process_pdfs(pdf_document_directory)
    
    if not pdf_chunks:
        print("⚠️ Không có dữ liệu PDF. Tiếp tục với phân tích cơ bản.")
    else:
        print(f"✅ Đã tải {len(pdf_chunks)} đoạn văn bản từ PDF.")
        # Khởi tạo semantic search engine với dữ liệu PDF
        search_engine.load_documents(pdf_chunks)

    # Khởi tạo các model
    print("\n🤖 Khởi tạo các model AI...")
    active_chats = [init_chat_engine(name) for name in engines_to_use]
    valid_engines = [
        engine for engine in active_chats
        if engine[0] is not None and (engine[0] != 'gemini' or engine[1] is not None)
    ]

    if not valid_engines:
        print("❌ Không có model nào khả dụng.")
        print("🔄 Đang thử quét lại APIs...")
        # Quét lại APIs khi không khởi tạo được model nào
        api_scan_results = scan_available_apis(api_rate_manager, verbose=True)
        engines_to_use = api_scan_results["summary"]["recommended_apis"]
        
        if engines_to_use:
            print("🔄 Thử lại với APIs mới phát hiện...")
            active_chats = [init_chat_engine(name) for name in engines_to_use]
            valid_engines = [
                engine for engine in active_chats
                if engine[0] is not None and (engine[0] != 'gemini' or engine[1] is not None)
            ]
            
            if not valid_engines:
                print("❌ Vẫn không khởi tạo được model nào sau khi quét lại. Thoát.")
                return []
        else:
            print("❌ Không phát hiện API khả dụng mới. Thoát.")
            return []

    print(f"✅ Đã khởi tạo thành công {len(valid_engines)} model: {[e[0] for e in valid_engines]}")
    
    # Thiết lập cấu hình batch và prompt
    SLEEP_BETWEEN_RECORDS = int(os.getenv("SLEEP_BETWEEN_RECORDS", "1"))
    MAX_WORKER_THREADS = int(os.getenv("MAX_WORKER_THREADS", "2"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
    
    # Nếu người dùng muốn phân tích đầy đủ và chi tiết
    all_results = []
    material_scores = {}
    output_base_name = os.path.join(results_dir, f"analysis_{time.strftime('%Y%m%d-%H%M%S')}")
    
    try:
        # Tải dữ liệu vật liệu
        try:
            all_materials = load_material_data(material_csv_file)
            print(f"📄 Đã tải {len(all_materials)} mẫu vật liệu từ {material_csv_file}")
        except Exception as e:
            print(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
            return []
        
        # Giới hạn số lượng mẫu
        materials_sample = all_materials[:num_samples]
        print(f"🔍 Xử lý {len(materials_sample)} mẫu đầu tiên...")
        
        if output_detailed:
            print("\n🔬 Bắt đầu phân tích đầy đủ và chi tiết...")
            # Phân tích đầy đủ với so sánh prompt
            prompt_variants_config = {
                # Text response variants
                "basic": {"prompt_type": "basic", "output_format": "text"},
                "cot": {"prompt_type": "cot", "output_format": "text"},
                "role": {"prompt_type": "role", "output_format": "text"},
                "role_cot": {"prompt_type": "role_cot", "output_format": "text"}
            }
            
            results, material_scores = analyze_materials_with_prompt_comparison(
                material_csv_file,
                output_base_name,
                valid_engines,
                pdf_chunks,
                limit_records=num_samples,
                sleep_between=SLEEP_BETWEEN_RECORDS,
                max_workers=MAX_WORKER_THREADS,
                batch_size=BATCH_SIZE,
                api_rate_manager=api_rate_manager,
                response_cache=response_cache
            )
            
            all_results.extend(results)
            
            # In thống kê sử dụng API và cache
            api_rate_manager.print_usage_stats()
            cache_stats = response_cache.stats()
            print(f"\n📦 Thống kê cache: {cache_stats['total_entries']} mục, {cache_stats['size_kb']:.2f} KB")
            
            # Phân tích chi tiết kết quả
            print("\n📊 Đang phân tích chi tiết kết quả...")
            analyze_results_in_detail(results)
        else:
            print("\n🔬 Bắt đầu phân tích cơ bản...")
            # Cấu hình prompt đơn giản 
            prompt_config = {"basic": {"category": "task", "key": "analyze_material"}}
            
            # Chia thành các batch
            batches = []
            for i in range(0, len(materials_sample), BATCH_SIZE):
                batches.append(materials_sample[i:i+BATCH_SIZE])
            
            # Xử lý từng batch
            batch_results = []
            for batch_idx, batch in enumerate(batches):
                print(f"\n🔄 Xử lý batch {batch_idx+1}/{len(batches)} ({len(batch)} mẫu)...")
                
                # Xử lý batch
                results = process_materials_batch(
                    batch,
                    prompt_config,
                    valid_engines,
                    pdf_chunks,
                    sleep_between=SLEEP_BETWEEN_RECORDS,
                    search_engine=search_engine,
                    api_rate_manager=api_rate_manager,
                    response_cache=response_cache
                )
                
                batch_results.extend(results)
            
            # Lưu kết quả vào file JSON
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = os.path.join(results_dir, f"basic_analysis_{timestamp}.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)
            
            print(f"📊 Kết quả đã được lưu tại: {output_file}")
            
            # In tóm tắt các kết quả
            print("\n🔍 Tóm tắt kết quả phân tích:")
            for idx, result in enumerate(batch_results):
                material_name = result["material_name"]
                material_score = result["material_score"]
                
                print(f"\n{idx+1}. {material_name}:")
                # Kiểm tra nếu material_score là dictionary
                if isinstance(material_score, dict):
                    if "score" in material_score:
                        print(f"   - Điểm đánh giá: {material_score['score']:.2f}/100")
                    else:
                        print(f"   - Điểm đánh giá: {material_score}")
                else:
                    print(f"   - Điểm đánh giá: {material_score:.2f}/100")
                
                # Hiển thị tóm tắt từ mỗi engine
                for analysis in result["analysis_results"]:
                    engine = analysis.get("engine", "Unknown")
                    status = analysis.get("status", "Unknown")
                    
                    if status == "success":
                        response = analysis.get("response", "")
                        # Chỉ lấy dòng đầu tiên làm tóm tắt
                        summary = response.split("\n")[0][:100] + "..." if response else "Không có phản hồi"
                        print(f"   - [{engine.upper()}]: {summary}")
                    else:
                        print(f"   - [{engine.upper()}]: ❌ Lỗi: {analysis.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình phân tích: {e}")
        import traceback
        print(traceback.format_exc())
    
    print("\n✅ Phân tích hoàn tất!")
    return all_results

if __name__ == "__main__":
    # Khởi tạo API Rate Manager và Response Cache
    api_rate_manager = APIRateManager()
    
    # Đọc các cấu hình từ file .env
    cache_dir = os.getenv("CACHE_DIRECTORY", "cache")
    response_cache = ResponseCache(cache_dir=cache_dir)
    
    # Đọc các thông số giới hạn
    try:
        MAX_RECORDS_TO_PROCESS = int(os.getenv("MAX_RECORDS_TO_PROCESS", 20))
        MAX_WORKER_THREADS = int(os.getenv("MAX_WORKER_THREADS", 2))
        BATCH_SIZE = int(os.getenv("BATCH_SIZE", 2))
        SLEEP_BETWEEN_RECORDS = int(os.getenv("SLEEP_BETWEEN_RECORDS", 1))
        
        # Cấu hình quét API
        API_SCAN_INTERVAL_MINUTES = int(os.getenv("API_SCAN_INTERVAL_MINUTES", 15))
        ENABLE_PERIODIC_API_SCAN = os.getenv("ENABLE_PERIODIC_API_SCAN", "True").lower() == "true"
        
        # Số lượng vật liệu giả lập nếu cần tạo mới
        NUM_FAKE_MATERIALS = MAX_RECORDS_TO_PROCESS
    except ValueError as e:
        print(f"⚠️ Lỗi khi đọc cấu hình từ .env: {e}. Sử dụng giá trị mặc định.")
        MAX_RECORDS_TO_PROCESS = 20
        MAX_WORKER_THREADS = 2
        BATCH_SIZE = 2
        SLEEP_BETWEEN_RECORDS = 1
        API_SCAN_INTERVAL_MINUTES = 15
        ENABLE_PERIODIC_API_SCAN = True
        NUM_FAKE_MATERIALS = MAX_RECORDS_TO_PROCESS
    
    # Khởi tạo Semantic Search Engine
    semantic_search_engine = SemanticSearchEngine()
    
    # Hiển thị menu lựa chọn
    print("\n=== AI MATERIAL ANALYZER ===")
    print("1. Phân tích vật liệu (Tất cả tính năng)")
    print("2. Tạo dữ liệu mẫu mới")
    print("0. Thoát")
    
    try:
        choice = int(input("\nLựa chọn: "))
        
        if choice == 1:
            # Hỏi số lượng mẫu
            try:
                num_samples = int(input("\nNhập số lượng mẫu cần phân tích (mặc định 20): ") or "20")
            except ValueError:
                print("❌ Số lượng mẫu không hợp lệ. Sử dụng giá trị mặc định 20.")
                num_samples = 20
                
            # Hỏi chế độ phân tích
            detailed = input("\nBạn muốn phân tích chi tiết (nhiều prompt, báo cáo đầy đủ)? (y/n, mặc định: n): ").lower() == 'y'
            
            # Chạy chức năng thống nhất
            analyze_materials_unified(num_samples=num_samples, output_detailed=detailed)
                
        elif choice == 2:
            # Tạo dữ liệu mẫu mới
            pdf_document_directory = "data/raw/documents"
            if not os.path.exists(pdf_document_directory):
                os.makedirs(pdf_document_directory, exist_ok=True)
                
            material_csv_file = os.path.join(pdf_document_directory, "fake_materials_dataset.csv")
            num_samples = int(input("\nNhập số lượng mẫu cần tạo (mặc định 50): ") or "50")
            
            if create_fake_material_csv(material_csv_file, num_materials=num_samples):
                print(f"✅ Đã tạo thành công {num_samples} mẫu tại {material_csv_file}")
        else:
            print("👋 Thoát chương trình.")
            
    except ValueError:
        print("❌ Lựa chọn không hợp lệ.")
    except KeyboardInterrupt:
        print("\n👋 Thoát chương trình theo yêu cầu của người dùng.")
    except Exception as e:
        print(f"\n❌ Lỗi không xác định: {e}")
        print(traceback.format_exc())
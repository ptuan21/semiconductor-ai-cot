import os
import google.generativeai as genai
import requests
from dotenv import load_dotenv
import concurrent.futures
import time
import traceback
import google.api_core.exceptions

# Load environment variables
load_dotenv()

# Configure API keys for Gemini and Groq ONLY
GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")  # Multiple keys separated by comma
GROQ_API_KEYS = os.getenv("GROQ_API_KEYS", "").split(",")      # Multiple keys separated by comma

# Current key indices
current_gemini_key_index = 0
current_groq_key_index = 0

def get_next_api_key(api_keys, current_index):
    """Get next available API key from the pool"""
    if not api_keys or not api_keys[0]:  # If no keys or empty string
        return None
    next_index = (current_index + 1) % len(api_keys)
    return api_keys[next_index], next_index

# Configure Gemini client with first key
if GEMINI_API_KEYS and GEMINI_API_KEYS[0]:
    try:
        genai.configure(api_key=GEMINI_API_KEYS[0])
        print("✅ Cấu hình Gemini thành công.")
    except Exception as e:
        print(f"⚠️ Lỗi cấu hình Gemini API Key: {e}")
else:
    print("⚠️ API Key của Gemini chưa được cấu hình trong file .env.")

# Kiểm tra Groq Keys
if not GROQ_API_KEYS or not GROQ_API_KEYS[0]:
    print("⚠️ API Key của Groq chưa được cấu hình trong file .env.")

def rotate_api_key(engine_name):
    """Rotate to next API key for the specified engine"""
    global current_gemini_key_index, current_groq_key_index
    
    if engine_name == "gemini":
        next_key, next_index = get_next_api_key(GEMINI_API_KEYS, current_gemini_key_index)
        if next_key:
            try:
                genai.configure(api_key=next_key)
                current_gemini_key_index = next_index
                print(f"🔄 Đã chuyển sang Gemini API key mới (index: {next_index})")
                return True
            except Exception as e:
                print(f"⚠️ Lỗi khi cấu hình Gemini key mới: {e}")
    
    elif engine_name == "groq":
        next_key, next_index = get_next_api_key(GROQ_API_KEYS, current_groq_key_index)
        if next_key:
            current_groq_key_index = next_index
            print(f"🔄 Đã chuyển sang Groq API key mới (index: {next_index})")
            return True
    
    return False

# --- Hàm khởi tạo và gọi API đơn lẻ (Giữ lại Gemini và Groq) ---
def init_chat_engine(engine_name):
    """Khởi tạo chat engine theo tên (Gemini hoặc Groq)."""
    if engine_name == "gemini":
        if not GEMINI_API_KEYS or not GEMINI_API_KEYS[0]:
            print("⚠️ API Key của Gemini chưa được cấu hình.")
            return (engine_name, None)
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            print("ℹ️ Sử dụng model: gemini-1.5-flash-latest")
            chat = model.start_chat(history=[])
            return (engine_name, chat)
        except AttributeError:
             print(f"❌ Lỗi khởi tạo Gemini: Phiên bản 'google.generativeai' có thể đã cũ hoặc cài đặt lỗi.")
             print(f"   Hãy thử: pip install --upgrade google-generativeai")
             return (engine_name, None)
        except Exception as e:
            print(f"❌ Lỗi khởi tạo Gemini: {e}")
            return (engine_name, None)
    elif engine_name == "groq":
        if not GROQ_API_KEYS or not GROQ_API_KEYS[0]:
            print("⚠️ API Key của Groq chưa được cấu hình.")
            return (engine_name, None)
        return (engine_name, None)  # Groq không cần đối tượng chat stateful
    else:
        print(f"⚠️ Engine '{engine_name}' không được hỗ trợ (chỉ Gemini hoặc Groq).")
        return (engine_name, None)

def chat_with_engine(engine_tuple, message, max_retries=2):
    """Gửi tin nhắn tới đúng engine (Gemini hoặc Groq), với retry và key rotation."""
    engine_name, chat = engine_tuple
    retries = 0
    key_rotations = 0
    max_key_rotations = len(GEMINI_API_KEYS) if engine_name == "gemini" else len(GROQ_API_KEYS)

    while retries <= max_retries and key_rotations < max_key_rotations:
        try:
            if engine_name == "gemini":
                if chat is None: return "❌ Gemini engine chưa được khởi tạo thành công."
                print(f"   Attempting Gemini call (Attempt {retries + 1}/{max_retries + 1}, Key {current_gemini_key_index + 1}/{len(GEMINI_API_KEYS)})...")
                response = chat.send_message(message)
                return response.text

            elif engine_name == "groq":
                if not GROQ_API_KEYS or not GROQ_API_KEYS[current_groq_key_index]: 
                    return "❌ API Key của Groq chưa được cấu hình."
                
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEYS[current_groq_key_index]}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "llama3-70b-8192",
                    "messages": [{"role": "user", "content": message}],
                    "max_tokens": 500
                }
                print(f"   Attempting Groq call (Attempt {retries + 1}/{max_retries + 1}, Key {current_groq_key_index + 1}/{len(GROQ_API_KEYS)})...")
                res = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                                  headers=headers, json=payload, timeout=180)

                if res.status_code == 429:  # Rate limit
                    if rotate_api_key(engine_name):
                        key_rotations += 1
                        continue
                    else:
                        return f"❌ Groq rate limit hit and no more API keys available"

                if res.status_code != 200:
                    try:
                        error_detail = res.json().get('error', {}).get('message', res.text)
                    except requests.exceptions.JSONDecodeError:
                        error_detail = res.text
                    return f"❌ Groq API Error ({res.status_code}): {error_detail}"

                try:
                    return res.json()['choices'][0]['message']['content']
                except (KeyError, IndexError, TypeError, requests.exceptions.JSONDecodeError) as json_err:
                    return f"❌ Groq JSON Error: {json_err} - Response: {res.text}"

            else:
                return f"❌ Engine '{engine_name}' không hợp lệ hoặc chưa được khởi tạo."

        except google.api_core.exceptions.ResourceExhausted as e:
            if engine_name == "gemini":
                # Try rotating API key first
                if rotate_api_key(engine_name):
                    key_rotations += 1
                    # Reinitialize chat with new key
                    try:
                        model = genai.GenerativeModel('gemini-1.5-flash-latest')
                        chat = model.start_chat(history=[])
                        continue
                    except Exception as reinit_err:
                        print(f"⚠️ Lỗi khởi tạo lại Gemini với key mới: {reinit_err}")
                
                # If key rotation failed or we still hit rate limit
                if retries < max_retries:
                    retry_delay = 30  # Default delay
                    try:
                        for detail in e.details:
                            if hasattr(detail, 'retry_delay') and detail.retry_delay.seconds > 0:
                                retry_delay = detail.retry_delay.seconds
                                break
                    except Exception:
                        pass
                    print(f"   ⚠️ Gemini rate limit (429) hit. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retries += 1
                    continue
            
            print(f"❌ Lỗi API (ResourceExhausted 429): {e}")
            tb_str = traceback.format_exc()
            return f"❌ Lỗi API (429): {e}\nTraceback:\n{tb_str}"

        except requests.exceptions.Timeout:
            if retries < max_retries:
                print(f"   ⚠️ Timeout, retrying... ({retries + 1}/{max_retries})")
                retries += 1
                continue
            return f"❌ Lỗi Timeout khi gọi API {engine_name.upper()}."

        except requests.exceptions.RequestException as req_err:
            if retries < max_retries:
                print(f"   ⚠️ Network error, retrying... ({retries + 1}/{max_retries})")
                retries += 1
                continue
            return f"❌ Lỗi kết nối mạng ({engine_name}): {str(req_err)}"

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"❌ Lỗi không xác định khi chat với {engine_name}: {str(e)}")
            if retries < max_retries:
                retries += 1
                continue
            return f"❌ Lỗi không xác định khi chat với {engine_name}: {str(e)}\nTraceback:\n{tb_str}"

    return f"❌ Không thể nhận phản hồi từ {engine_name} sau {retries} lần thử và {key_rotations} lần đổi key."

# --- Hàm bao bọc cho Executor (_run_chat_task_internal) ---
# Sửa lại để LUÔN trả về dictionary
def _run_chat_task_internal(engine_tuple, prompt):
    """Hàm nội bộ để gọi chat_with_engine, trả về trạng thái và thời gian dưới dạng dict."""
    engine_name, _ = engine_tuple
    start_time = time.time()
    # Gọi hàm chat_with_engine với retry logic
    result_text = chat_with_engine(engine_tuple, prompt, max_retries=2)
    end_time = time.time()
    exec_time = end_time - start_time
    # Xác định trạng thái dựa trên tiền tố lỗi
    status = "success" if not result_text.startswith("❌") else "error"

    # In log lỗi hoặc thành công (Có thể giữ lại hoặc xóa nếu chat_with_engine đã in đủ)
    # if status == "error":
    #     print(f"   ⚠️ Hoàn thành ({engine_name.upper()}) với lỗi sau {exec_time:.2f} giây.")
    # else:
    #     print(f"   ✅ Hoàn thành ({engine_name.upper()}) sau {exec_time:.2f} giây")

    # ---> Luôn trả về dictionary <---
    return {
        "engine": engine_name,
        "status": status,
        "response": result_text, # Trả về cả lỗi hoặc kết quả thành công
        "execution_time": exec_time
    }

# --- Hàm chạy đồng thời (analyze_prompt_concurrently) ---
# Bây giờ không cần kiểm tra isinstance nữa vì _run_chat_task_internal đã đảm bảo trả về dict
def analyze_prompt_concurrently(prompt, engine_tuples, max_workers=None):
    """
    Gửi prompt đến nhiều engine đồng thời và trả về danh sách kết quả chi tiết.
    Returns:
        list: Danh sách các dictionary.
    """
    if max_workers is None:
        max_workers = len(engine_tuples)

    results_list = [] # Sẽ chứa các dict kết quả
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_engine = {
            executor.submit(_run_chat_task_internal, engine_tuple, prompt): engine_tuple[0]
            for engine_tuple in engine_tuples if engine_tuple[0] is not None
        }

        # Thu thập kết quả khi hoàn thành
        for future in concurrent.futures.as_completed(future_to_engine):
            engine_name = future_to_engine[future]
            try:
                # Lấy dict kết quả từ future (bây giờ chắc chắn là dict)
                run_result_dict = future.result()
                results_list.append(run_result_dict)

            except Exception as exc:
                # Xử lý lỗi nếu bản thân future gặp sự cố
                print(f"   ‼️ Lỗi Executor nghiêm trọng cho {engine_name}: {exc}")
                results_list.append({
                    "engine": engine_name,
                    "status": "error",
                    "response": f"❌ Lỗi Executor: {exc}",
                    "execution_time": 0
                })
    return results_list # Đảm bảo trả về list chỉ chứa các dict


# --- Hàm Main cũ (cập nhật để chỉ dùng Gemini, Groq) ---
def main_interactive_gemini_groq():
    print("🤖 Khởi động hệ thống Chat Gemini & Groq (Tương tác)...")
    # Chỉ định rõ engines cần dùng
    engines_to_use = ["gemini", "groq"]
    print("🚀 Khởi tạo các chat engines...")
    active_chats = [init_chat_engine(name) for name in engines_to_use]
    # Lọc engine hợp lệ
    valid_engines = [
        engine for engine in active_chats
        if engine[0] is not None and (engine[0] != 'gemini' or engine[1] is not None)
    ]

    if not valid_engines:
         print("❌ Không có engine nào hợp lệ. Thoát.")
         return

    print(f"✅ Các engine sẵn sàng: {[e[0].upper() for e in valid_engines]} (Gõ 'quit' để thoát)")

    while True:
        user_input = input("\nBạn: ")
        if user_input.lower() == 'quit':
            print("👋 Tạm biệt!")
            break

        print("\n🔄 Đang gửi yêu cầu đồng thời...")
        all_results = analyze_prompt_concurrently(user_input, valid_engines)

        print("\n--- Kết quả ---")
        for result in all_results:
            print(f"[{result['engine'].upper()}]: {result['response']}\n")


if __name__ == "__main__":
     # Chạy main tương tác Gemini & Groq nếu gọi trực tiếp model_manager.py
     main_interactive_gemini_groq()

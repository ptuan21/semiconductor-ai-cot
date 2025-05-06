import os
import google.generativeai as genai
import openai
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure clients
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def init_chat_engine(engine_name="gemini"):
    """Khởi tạo chat engine theo tên."""
    if engine_name == "gemini":
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            chat = model.start_chat(history=[])
            return (engine_name, chat)
        except Exception as e:
            print(f"Lỗi khởi tạo Gemini: {e}")
            return (engine_name, None)
    elif engine_name == "openrouter":
        return (engine_name, None)  # OpenRouter không cần init
    elif engine_name == "groq":
        return (engine_name, None)  # Groq cũng không cần init
    else:
        return (engine_name, None)

def chat_with_engine(engine_tuple, message):
    """Gửi tin nhắn tới đúng engine."""
    engine_name, chat = engine_tuple
    try:
        if engine_name == "gemini":
            response = chat.send_message(message)
            return response.text
        elif engine_name == "openrouter":
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://yourdomain.com",
                "X-Title": "Material Analysis"
            }
            payload = {
                "model": "mistralai/mixtral-8x7b-instruct",  # hoặc llama3-70b nếu thích
                "messages": [{"role": "user", "content": message}]
            }
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            return res.json()['choices'][0]['message']['content']
        elif engine_name == "groq":
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "mixtral-8x7b-32768",  # Hoặc "llama3-70b-8192"
                "messages": [{"role": "user", "content": message}]
            }
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            return res.json()['choices'][0]['message']['content']
            if res.status_code != 200:
                return f"❌ Groq API Error: {res.text}"  # Bắt lỗi server trả về rõ ràng
            return res.json()['choices'][0]['message']['content']
        else:
            return "❌ Không tìm thấy engine phù hợp."
    except Exception as e:
        return f"❌ Lỗi khi chat với {engine_name}: {str(e)}"

def main():
    print("🤖 Khởi động hệ thống Chat đa mô hình...")
    engines = ["gemini", "openrouter", "groq"]
    active_chats = [init_chat_engine(name) for name in engines]

    print("✅ Sẵn sàng! (Gõ 'quit' để thoát)")

    while True:
        user_input = input("\nBạn: ")
        if user_input.lower() == 'quit':
            print("👋 Tạm biệt!")
            break

        for engine_tuple in active_chats:
            response = chat_with_engine(engine_tuple, user_input)
            print(f"\n[{engine_tuple[0].upper()}]: {response}")

if __name__ == "__main__":
    main()

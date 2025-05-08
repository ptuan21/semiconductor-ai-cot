"""
Script chạy phân tích song song với Gemini và DeepSeek
"""

from evaluate_models import analyze_materials_parallel

if __name__ == "__main__":
    print("=== PHÂN TÍCH VẬT LIỆU SONG SONG GEMINI + DEEPSEEK ===")
    
    try:
        num_samples = int(input("Nhập số lượng mẫu cần phân tích (mặc định 20): ") or "20")
        analyze_materials_parallel(num_samples)
    except ValueError:
        print("❌ Số lượng mẫu không hợp lệ. Sử dụng giá trị mặc định 20.")
        analyze_materials_parallel(20)
    except KeyboardInterrupt:
        print("\n👋 Thoát chương trình theo yêu cầu của người dùng.")
    except Exception as e:
        print(f"\n❌ Lỗi không xác định: {e}") 
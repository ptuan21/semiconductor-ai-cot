"""
Script phân tích vật liệu bán dẫn thống nhất
Kết hợp tất cả chức năng: quét API, phân tích song song Gemini và DeepSeek
"""

from evaluate_models import analyze_materials_unified
import sys

if __name__ == "__main__":
    print("=== PHÂN TÍCH VẬT LIỆU BÁN DẪN THỐNG NHẤT ===")
    
    # Kiểm tra tham số dòng lệnh
    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
            print(f"Số lượng mẫu từ tham số dòng lệnh: {num_samples}")
        except ValueError:
            print("❌ Tham số không hợp lệ. Sử dụng số lượng mẫu mặc định (20).")
            num_samples = 20
    else:
        # Hỏi số lượng mẫu
        try:
            num_samples = int(input("Nhập số lượng mẫu cần phân tích (mặc định 20): ") or "20")
        except ValueError:
            print("❌ Số lượng mẫu không hợp lệ. Sử dụng giá trị mặc định 20.")
            num_samples = 20
    
    # Hỏi chế độ phân tích
    detailed_input = input("Bạn muốn phân tích chi tiết (nhiều prompt, báo cáo đầy đủ)? (y/n, mặc định: n): ")
    detailed = detailed_input.lower() == 'y'
    
    try:
        # Chạy phân tích thống nhất
        analyze_materials_unified(num_samples=num_samples, output_detailed=detailed)
        
        print("\n✅ Phân tích hoàn tất!")
    except KeyboardInterrupt:
        print("\n👋 Thoát chương trình theo yêu cầu của người dùng.")
    except Exception as e:
        import traceback
        print(f"\n❌ Lỗi không xác định: {e}")
        traceback.print_exc() 
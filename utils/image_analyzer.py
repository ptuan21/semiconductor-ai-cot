import cv2
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import logging
from model_manager import init_chat_engine, analyze_prompt_concurrently
import base64

class MaterialImageAnalyzer:
    """Phân tích hình ảnh vật liệu và phát hiện lỗi"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Khởi tạo các model AI
        self.engines = [
            init_chat_engine("gemini"),
            init_chat_engine("groq")
        ]
        self.valid_engines = [
            engine for engine in self.engines
            if engine[0] is not None
        ]
        
    def analyze_image(self, image_path: str) -> Dict:
        """
        Phân tích hình ảnh và phát hiện các lỗi
        Args:
            image_path: Đường dẫn đến file hình ảnh
        Returns:
            Dict chứa kết quả phân tích
        """
        try:
            # Đọc hình ảnh
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Không thể đọc hình ảnh từ {image_path}")
            
            # Chuyển sang grayscale để xử lý
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Phát hiện các loại lỗi khác nhau
            surface_defects = self._detect_surface_defects(gray)
            crystal_defects = self._detect_crystal_defects(gray)
            structural_defects = self._detect_structural_defects(gray)
            
            # Tổng hợp kết quả
            results = {
                "surface_defects": surface_defects,
                "crystal_defects": crystal_defects,
                "structural_defects": structural_defects,
                "total_defects": len(surface_defects) + len(crystal_defects) + len(structural_defects),
                "image_quality": self._assess_image_quality(gray)
            }
            
            # Tạo hình ảnh với các lỗi được đánh dấu
            marked_image = self._mark_defects(img.copy(), results)
            output_path = Path(image_path).parent / f"analyzed_{Path(image_path).name}"
            cv2.imwrite(str(output_path), marked_image)
            results["marked_image_path"] = str(output_path)
            
            # Thêm phân tích từ các model AI
            ai_analysis = self._get_ai_analysis(img, results)
            results["ai_analysis"] = ai_analysis
            
            return results
            
        except Exception as e:
            self.logger.error(f"Lỗi khi phân tích hình ảnh: {e}")
            return {"error": str(e)}

    def _get_ai_analysis(self, img: np.ndarray, defect_results: Dict) -> Dict:
        """Phân tích hình ảnh bằng các model AI"""
        try:
            # Chuyển hình ảnh thành base64 để đưa vào prompt
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Tạo prompt cho các model
            prompt = self._create_analysis_prompt(defect_results, img_base64)
            
            # Phân tích với các model
            ai_results = analyze_prompt_concurrently(prompt, self.valid_engines)
            
            # Tổng hợp kết quả từ các model
            analysis_results = {
                "gemini": None,
                "groq": None,
                "combined_insights": []
            }
            
            all_insights = set()
            for result in ai_results:
                model_name = result.get("engine", "unknown")
                response = result.get("response", "")
                
                # Lưu phân tích riêng của từng model
                analysis_results[model_name] = response
                
                # Trích xuất insights từ response
                insights = self._extract_insights(response)
                all_insights.update(insights)
            
            # Thêm các insights đã được tổng hợp
            analysis_results["combined_insights"] = sorted(list(all_insights))
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Lỗi khi phân tích AI: {e}")
            return {
                "error": str(e),
                "gemini": None,
                "groq": None,
                "combined_insights": []
            }

    def _create_analysis_prompt(self, defect_results: Dict, img_base64: str) -> str:
        """Tạo prompt cho phân tích AI"""
        prompt = f"""Analyze this semiconductor material image and provide insights about:

1. Material Properties:
   - Crystal structure and quality
   - Surface characteristics
   - Potential material composition
   - Phase uniformity

2. Defect Analysis:
   The following defects were detected:
   - {len(defect_results['surface_defects'])} surface defects
   - {len(defect_results['crystal_defects'])} crystal defects
   - {len(defect_results['structural_defects'])} structural defects
   
   Image quality score: {defect_results['image_quality']['quality_score']:.1f}

3. Please provide:
   - Detailed analysis of the material's characteristics
   - Potential causes of the detected defects
   - Impact on material performance
   - Recommendations for improvement
   - Possible applications based on observed properties

Base your analysis on the image and the defect detection results provided.
Format your response in a clear, structured manner with specific sections for each aspect.

[Image data available but not shown in prompt]"""
        return prompt

    def _extract_insights(self, response: str) -> List[str]:
        """Trích xuất các insights chính từ response của model"""
        insights = []
        
        # Tách response thành các dòng
        lines = response.split('\n')
        
        for line in lines:
            # Lọc các dòng có chứa thông tin quan trọng
            line = line.strip()
            if line and len(line) > 10:  # Bỏ qua các dòng quá ngắn
                # Loại bỏ các ký tự đánh dấu thông thường
                line = line.lstrip('•-*').strip()
                if line:
                    insights.append(line)
        
        return insights

    def _detect_surface_defects(self, gray_img: np.ndarray) -> List[Dict]:
        """Phát hiện các lỗi bề mặt"""
        defects = []
        
        # Áp dụng bộ lọc Gaussian để giảm nhiễu
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        
        # Phát hiện cạnh bằng Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Tìm contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 100:  # Lọc các vùng quá nhỏ
                x, y, w, h = cv2.boundingRect(contour)
                defects.append({
                    "id": f"surface_{i}",
                    "type": "surface",
                    "location": (x, y),
                    "size": (w, h),
                    "area": area,
                    "severity": "high" if area > 500 else "medium" if area > 200 else "low"
                })
        
        return defects

    def _detect_crystal_defects(self, gray_img: np.ndarray) -> List[Dict]:
        """Phát hiện các lỗi tinh thể"""
        defects = []
        
        # Áp dụng ngưỡng thích ứng
        thresh = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Tìm các vùng không đồng nhất
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Tìm contours của các vùng không đồng nhất
        contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if 50 < area < 1000:  # Lọc theo kích thước phù hợp
                x, y, w, h = cv2.boundingRect(contour)
                # Tính độ tròn để phân biệt lỗi tinh thể
                perimeter = cv2.arcLength(contour, True)
                circularity = 4*np.pi*area/(perimeter*perimeter) if perimeter > 0 else 0
                
                if circularity > 0.7:  # Lỗi tinh thể thường có dạng tròn
                    defects.append({
                        "id": f"crystal_{i}",
                        "type": "crystal",
                        "location": (x, y),
                        "size": (w, h),
                        "area": area,
                        "circularity": circularity,
                        "severity": "high" if circularity > 0.9 else "medium"
                    })
        
        return defects

    def _detect_structural_defects(self, gray_img: np.ndarray) -> List[Dict]:
        """Phát hiện các lỗi cấu trúc"""
        defects = []
        
        # Tính gradient
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Chuyển về uint8 để xử lý tiếp
        gradient_magnitude = np.uint8(gradient_magnitude * 255 / gradient_magnitude.max())
        
        # Áp dụng ngưỡng để tìm các vùng có gradient cao
        _, thresh = cv2.threshold(gradient_magnitude, 127, 255, cv2.THRESH_BINARY)
        
        # Tìm các đường thẳng bằng Hough Transform
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                defects.append({
                    "id": f"structural_{i}",
                    "type": "structural",
                    "start_point": (x1, y1),
                    "end_point": (x2, y2),
                    "length": length,
                    "severity": "high" if length > 200 else "medium" if length > 100 else "low"
                })
        
        return defects

    def _assess_image_quality(self, gray_img: np.ndarray) -> Dict:
        """Đánh giá chất lượng hình ảnh"""
        # Tính histogram
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        
        # Tính các metrics
        mean_intensity = np.mean(gray_img)
        std_intensity = np.std(gray_img)
        
        # Tính độ tương phản
        contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
        
        # Tính độ nhiễu
        noise_metric = cv2.Laplacian(gray_img, cv2.CV_64F).var()
        
        return {
            "mean_intensity": float(mean_intensity),
            "contrast": float(contrast),
            "noise_level": float(noise_metric),
            "quality_score": float(100 - (noise_metric/100 + (1-contrast)*50))
        }

    def _mark_defects(self, img: np.ndarray, results: Dict) -> np.ndarray:
        """Đánh dấu các lỗi trên hình ảnh"""
        # Màu cho từng loại lỗi
        colors = {
            "surface": (0, 0, 255),      # Đỏ
            "crystal": (0, 255, 0),      # Xanh lá
            "structural": (255, 0, 0)     # Xanh dương
        }
        
        # Đánh dấu lỗi bề mặt và tinh thể
        for defect_type in ["surface_defects", "crystal_defects"]:
            for defect in results.get(defect_type, []):
                color = colors[defect["type"]]
                x, y = defect["location"]
                w, h = defect["size"]
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                # Thêm nhãn với mức độ nghiêm trọng
                label = f"{defect['type']}:{defect['severity']}"
                cv2.putText(img, label, (x, y-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Đánh dấu lỗi cấu trúc
        for defect in results.get("structural_defects", []):
            color = colors["structural"]
            start_point = tuple(map(int, defect["start_point"]))
            end_point = tuple(map(int, defect["end_point"]))
            cv2.line(img, start_point, end_point, color, 2)
        
        # Thêm thông tin tổng quan
        quality_score = results.get("image_quality", {}).get("quality_score", 0)
        total_defects = results.get("total_defects", 0)
        
        info_text = f"Quality Score: {quality_score:.1f} | Total Defects: {total_defects}"
        cv2.putText(img, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return img 
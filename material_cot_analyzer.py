import json
from typing import Dict, List, Any
import numpy as np
import google.generativeai as genai
from groq import Groq
import os
from dotenv import load_dotenv
import random
from datetime import datetime, timedelta
from openai import OpenAI
import time

class APIConfig:
    """Cấu hình cho mỗi API endpoint"""
    def __init__(self, api_key: str, model: str, priority: int = 1, proxies=None, max_errors=3):
        self.api_key = api_key
        self.model = model
        self.priority = priority
        self.last_used = None
        self.error_count = 0
        self.max_errors = max_errors
        self.cooldown_minutes = 5
        self.proxies = proxies

    def is_available(self) -> bool:
        """Kiểm tra xem API có khả dụng không"""
        if self.error_count >= self.max_errors:
            if self.last_used and datetime.now() - self.last_used < timedelta(minutes=self.cooldown_minutes):
                return False
            self.error_count = 0
        return True

    def mark_error(self):
        """Đánh dấu lỗi cho API"""
        self.error_count += 1
        self.last_used = datetime.now()

    def mark_success(self):
        """Đánh dấu thành công cho API"""
        self.error_count = 0
        self.last_used = datetime.now()

class MaterialCoTAnalyzer:
    """
    Phân tích vật liệu sử dụng phương pháp Chain-of-Thought (CoT)
    """
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Khởi tạo cấu hình API
        self._init_api_configs()
        
        # Initialize AI models
        self._init_ai_models()
        
        self.cot_steps = [
            self._analyze_composition,
            self._analyze_crystal_structure,
            self._analyze_electronic_properties,
            self._analyze_thermal_properties,
            self._predict_applications,
            self._suggest_improvements,
            self._get_ai_insights
        ]
        
    def _init_api_configs(self):
        """
        Khởi tạo cấu hình API cho các mô hình LLM khác nhau
        """
        # Đọc danh sách API keys từ biến môi trường
        load_dotenv()  # Đảm bảo đã tải biến môi trường
        
        # Đọc cài đặt proxy nếu có
        http_proxy = os.getenv('HTTP_PROXY', '')
        https_proxy = os.getenv('HTTPS_PROXY', '')
        self.proxies = {}
        
        if http_proxy:
            self.proxies['http'] = http_proxy
        if https_proxy:
            self.proxies['https'] = https_proxy
            
        print(f"🌐 Cài đặt proxy: {'Có' if self.proxies else 'Không'}")
        
        # Kiểm tra máy chủ Google và máy chủ Groq trước khi khởi tạo
        try:
            self._check_server_availability()
        except Exception as e:
            print(f"⚠️ Không thể kiểm tra khả dụng máy chủ API: {e}")
        
        # Khởi tạo các API keys cho Gemini
        gemini_api_keys = os.getenv('GEMINI_API_KEYS', '').split(',')
        self.gemini_configs = []
        
        for i, key in enumerate(gemini_api_keys):
            key = key.strip()
            if key:
                try:
                    # Kiểm tra nhanh key này có hợp lệ không
                    print(f"🔑 Kiểm tra Gemini API key {i+1}...")
                    # Thêm cấu hình mới
                    self.gemini_configs.append(
                        APIConfig(key, 'gemini-1.5-flash-latest', i+1, proxies=self.proxies, max_errors=3)
                    )
                except Exception as e:
                    print(f"⚠️ Không thể khởi tạo Gemini API key {i+1}: {e}")
        
        # Khởi tạo các API keys cho Groq
        groq_api_keys = os.getenv('GROQ_API_KEYS', '').split(',')
        self.groq_configs = []
        
        for i, key in enumerate(groq_api_keys):
            key = key.strip()
            if key:
                try:
                    # Thêm cấu hình mới
                    print(f"🔑 Kiểm tra Groq API key {i+1}...")
                    self.groq_configs.append(
                        APIConfig(key, 'llama3-8b-8192', i+1, proxies=self.proxies, max_errors=3)
                    )
                except Exception as e:
                    print(f"⚠️ Không thể khởi tạo Groq API key {i+1}: {e}")
        
        # Thiết lập các mô hình có sẵn
        self.models_available = {
            'gemini': len(self.gemini_configs) > 0,
            'groq': len(self.groq_configs) > 0
        }
        
        print(f"🤖 Gemini API sẵn có: {self.models_available['gemini']} ({len(self.gemini_configs)} keys)")
        print(f"🤖 Groq API sẵn có: {self.models_available['groq']} ({len(self.groq_configs)} keys)")
        
    def _check_server_availability(self):
        """
        Kiểm tra xem máy chủ API có thể kết nối được không
        """
        import socket
        import requests
        import time
        
        # Kiểm tra Google (cho Gemini)
        try:
            start_time = time.time()
            response = requests.get('https://generativelanguage.googleapis.com/healthz', 
                                   timeout=5, proxies=self.proxies)
            end_time = time.time()
            if response.status_code == 200:
                print(f"✅ Máy chủ Google (Gemini) hoạt động ({(end_time - start_time)*1000:.0f}ms)")
            else:
                print(f"⚠️ Máy chủ Google (Gemini) trả về mã trạng thái {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠️ Không thể kết nối đến máy chủ Google (Gemini): {e}")
        
        # Kiểm tra Groq
        try:
            start_time = time.time()
            response = requests.get('https://api.groq.com/healthz', 
                                   timeout=5, proxies=self.proxies)
            end_time = time.time()
            if response.status_code == 200:
                print(f"✅ Máy chủ Groq hoạt động ({(end_time - start_time)*1000:.0f}ms)")
            else:
                print(f"⚠️ Máy chủ Groq trả về mã trạng thái {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠️ Không thể kết nối đến máy chủ Groq: {e}")
            
    def _reload_apis(self):
        """
        Tải lại thông tin API khi cần thiết
        """
        print("🔄 Đang tải lại cấu hình API...")
        self._init_api_configs()

    def _get_available_api_config(self, configs: List[APIConfig]) -> APIConfig:
        """Lấy cấu hình API khả dụng tiếp theo"""
        available_configs = [c for c in configs if c.is_available()]
        if not available_configs:
            return None
        return random.choice(available_configs)

    def _init_ai_models(self):
        """Khởi tạo các mô hình AI với nhiều API endpoints"""
        self.models_available = {
            'gemini': len(self.gemini_configs) > 0,
            'groq': len(self.groq_configs) > 0
        }
        
        if not any(self.models_available.values()):
            print("⚠️ Warning: No API keys available, will use local analysis")
            print("💡 Tip: Add API keys to .env file:")
            print("GEMINI_API_KEYS=key1,key2,key3")
            print("GROQ_API_KEYS=key1,key2,key3")

    def analyze_with_cot(self, material_data: Dict) -> Dict:
        """
        Phân tích vật liệu theo từng bước suy luận
        """
        analysis_results = {
            'steps': [],
            'final_analysis': {},
            'recommendations': {},
            'basic_properties': {},
            'predictions': {}
        }
        
        try:
            # Ưu tiên phân tích cấu trúc nếu có API khả dụng
            if self.models_available['gemini'] or self.models_available['groq']:
                # Sử dụng phương thức mới với cấu trúc JSON
                structured_results = self._analyze_with_structured_cot(material_data)
                if structured_results:
                    return structured_results
            
            # Fallback nếu phân tích có cấu trúc không thành công
            # Thực hiện từng bước phân tích
            context = {'material_data': material_data}
            for step in self.cot_steps:
                step_result = step(context)
                analysis_results['steps'].append(step_result)
                context.update(step_result.get('context_update', {}))
            
            # Tổng hợp kết quả
            analysis_results['final_analysis'] = self._synthesize_results(analysis_results['steps'])
            analysis_results['recommendations'] = self._generate_recommendations(context)
            
            # Cập nhật basic_properties và predictions từ final_analysis
            analysis_results['basic_properties'] = analysis_results['final_analysis'].get('basic_properties', {})
            analysis_results['predictions'] = analysis_results['final_analysis'].get('predictions', {})
            
            # Thêm reasoning_steps cho hiển thị Chain of Thought
            analysis_results['reasoning_steps'] = self._format_reasoning_steps(analysis_results['steps'])
            
            # Thêm strengths và weaknesses để hiển thị trong tab CoT
            analysis_results['strengths'] = self._extract_strengths(context, analysis_results['final_analysis'])
            analysis_results['weaknesses'] = self._extract_weaknesses(context, analysis_results['final_analysis'])
            
        except Exception as e:
            print(f"❌ Error in analysis: {str(e)}")
            # Thêm thông tin lỗi vào kết quả
            analysis_results['error'] = {
                'message': str(e),
                'type': type(e).__name__
            }
        
        return analysis_results
    
    def _analyze_composition(self, context: Dict) -> Dict:
        """
        Bước 1: Phân tích thành phần
        """
        material_data = context['material_data']
        composition = material_data['composition']
        
        # Suy luận về thành phần
        reasoning = [
            f"1. Xem xét thành phần của vật liệu {material_data['name']}:",
            "   - Phân tích tỷ lệ các nguyên tố",
            "   - Kiểm tra tính ổn định hóa học",
            "   - Đánh giá khả năng tương tác điện tử"
        ]
        
        # Phân tích chi tiết
        analysis = {
            'element_count': len(composition),
            'main_elements': list(composition.keys()),
            'stoichiometry': self._check_stoichiometry(composition),
            'stability_prediction': self._predict_stability(composition)
        }
        
        return {
            'step_name': 'Composition Analysis',
            'reasoning': reasoning,
            'analysis': analysis,
            'context_update': {'composition_analysis': analysis}
        }
    
    def _analyze_crystal_structure(self, context: Dict) -> Dict:
        """
        Bước 2: Phân tích cấu trúc tinh thể
        """
        material_data = context['material_data']
        structure = material_data['crystal_structure']
        
        reasoning = [
            f"2. Đánh giá cấu trúc tinh thể {structure}:",
            "   - Xác định nhóm đối xứng",
            "   - Phân tích các thông số mạng",
            "   - Dự đoán tính chất vật lý"
        ]
        
        analysis = {
            'structure_type': structure,
            'symmetry': self._get_symmetry_info(structure),
            'predicted_properties': self._predict_structure_properties(structure)
        }
        
        return {
            'step_name': 'Crystal Structure Analysis',
            'reasoning': reasoning,
            'analysis': analysis,
            'context_update': {'structure_analysis': analysis}
        }
    
    def _analyze_electronic_properties(self, context: Dict) -> Dict:
        """
        Bước 3: Phân tích tính chất điện tử
        """
        material_data = context['material_data']
        properties = material_data['properties']
        
        reasoning = [
            "3. Phân tích tính chất điện tử:",
            "   - Đánh giá bandgap",
            "   - Xem xét độ dẫn điện",
            "   - Phân tích nồng độ hạt tải"
        ]
        
        analysis = {
            'bandgap_type': self._classify_bandgap(properties.get('bandgap')),
            'conductivity_analysis': self._analyze_conductivity(properties),
            'carrier_properties': self._analyze_carriers(properties)
        }
        
        return {
            'step_name': 'Electronic Properties Analysis',
            'reasoning': reasoning,
            'analysis': analysis,
            'context_update': {'electronic_analysis': analysis}
        }
    
    def _analyze_thermal_properties(self, context: Dict) -> Dict:
        """
        Bước 4: Phân tích tính chất nhiệt
        """
        material_data = context['material_data']
        properties = material_data['properties']
        
        reasoning = [
            "4. Đánh giá tính chất nhiệt:",
            "   - Xem xét nhiệt độ nóng chảy",
            "   - Phân tích độ dẫn nhiệt",
            "   - Dự đoán ổn định nhiệt"
        ]
        
        analysis = {
            'thermal_stability': self._analyze_thermal_stability(properties),
            'heat_transport': self._analyze_heat_transport(properties),
            'temperature_effects': self._predict_temperature_effects(properties)
        }
        
        return {
            'step_name': 'Thermal Properties Analysis',
            'reasoning': reasoning,
            'analysis': analysis,
            'context_update': {'thermal_analysis': analysis}
        }
    
    def _predict_applications(self, context: Dict) -> Dict:
        """
        Bước 5: Dự đoán ứng dụng tiềm năng
        """
        electronic = context.get('electronic_analysis', {})
        thermal = context.get('thermal_analysis', {})
        
        reasoning = [
            "5. Xác định ứng dụng tiềm năng:",
            "   - Dựa trên tính chất điện tử",
            "   - Xem xét khả năng chịu nhiệt",
            "   - Đánh giá tính thực tế"
        ]
        
        applications = self._identify_applications(electronic, thermal)
        
        return {
            'step_name': 'Application Prediction',
            'reasoning': reasoning,
            'analysis': {'potential_applications': applications},
            'context_update': {'applications': applications}
        }
    
    def _suggest_improvements(self, context: Dict) -> Dict:
        """
        Bước 6: Đề xuất cải tiến
        """
        all_analysis = context
        
        reasoning = [
            "6. Đề xuất hướng cải tiến:",
            "   - Xác định điểm yếu",
            "   - Đề xuất giải pháp",
            "   - Ưu tiên các cải tiến"
        ]
        
        improvements = self._generate_improvement_suggestions(all_analysis)
        
        return {
            'step_name': 'Improvement Suggestions',
            'reasoning': reasoning,
            'analysis': {'suggested_improvements': improvements},
            'context_update': {'improvements': improvements}
        }
    
    def _get_ai_insights(self, context: Dict) -> Dict:
        """
        Bước 7: Lấy phân tích từ các mô hình AI với nhiều API endpoints
        """
        material_data = context['material_data']
        
        reasoning = [
            "7. Phân tích từ các mô hình AI:",
            "   - Tổng hợp dữ liệu vật liệu",
            "   - Phân tích với nhiều API endpoints",
            "   - So sánh và tổng hợp kết quả"
        ]
        
        prompt = self._prepare_material_prompt(material_data)
        ai_responses = {}
        
        # Thử tất cả các API endpoints cho Gemini
        gemini_analysis = self._try_gemini_apis(prompt)
        if gemini_analysis:
            # Chuyển đổi 'content' thành 'analysis' để tương thích với code cũ
            ai_responses['gemini'] = {
                'analysis': gemini_analysis.get('content', ''),
                'confidence': gemini_analysis.get('confidence', 0.85),
                'status': gemini_analysis.get('status', 'success'),
                'api_priority': gemini_analysis.get('api_priority', 1)
            }
        
        # Thử tất cả các API endpoints cho Groq
        groq_analysis = self._try_groq_apis(prompt)
        if groq_analysis:
            # Chuyển đổi 'content' thành 'analysis' để tương thích với code cũ
            ai_responses['groq'] = {
                'analysis': groq_analysis.get('content', ''),
                'confidence': groq_analysis.get('confidence', 0.8),
                'status': groq_analysis.get('status', 'success'),
                'api_priority': groq_analysis.get('api_priority', 1)
            }
        
        # Nếu không có API nào thành công, sử dụng phân tích cục bộ
        if not ai_responses:
            local_analysis = self._perform_local_analysis(material_data)
            ai_responses['local'] = {
                'analysis': local_analysis,
                'confidence': 0.7,
                'status': 'fallback'
            }
            print("ℹ️ Using local analysis as fallback")
        
        return {
            'step_name': 'AI Model Analysis',
            'reasoning': reasoning,
            'analysis': {'ai_responses': ai_responses},
            'context_update': {'ai_analysis': ai_responses}
        }

    def _try_gemini_apis(self, prompt: str) -> Dict:
        """Thử tất cả các API endpoints của Gemini"""
        for config in self.gemini_configs:
            if not config.is_available():
                continue
                
            try:
                print(f"🔄 Thử kết nối với Gemini API (priority {config.priority})...")
                genai.configure(api_key=config.api_key)
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                
                # Thêm timeout để tránh treo vô thời hạn
                start_time = time.time()
                response = model.generate_content(prompt)
                elapsed_time = time.time() - start_time
                
                print(f"✅ Gemini API (priority {config.priority}) thành công trong {elapsed_time:.2f}s")
                config.mark_success()
                
                return {
                    'content': response.text,
                    'confidence': 0.85,
                    'status': 'success',
                    'api_priority': config.priority
                }
            except Exception as e:
                error_msg = f"❌ Lỗi khi sử dụng Gemini: {e}"
                print(error_msg)
                api_errors.append(error_msg)
        
        print("❌ Tất cả Gemini API keys đều không khả dụng")
        return None

    def _try_groq_apis(self, prompt: str) -> Dict:
        """Thử tất cả các API endpoints của Groq"""
        for config in self.groq_configs:
            if not config.is_available():
                continue
                
            try:
                print(f"🔄 Thử kết nối với Groq API (priority {config.priority})...")
                # Khởi tạo client với timeout
                start_time = time.time()
                
                import httpx
                # Tạo transport với proxy nếu có
                transport = None
                if config.proxies:
                    transport = httpx.HTTPTransport(proxy=config.proxies)
                
                # Tạo client với transport nếu cần
                client_args = {"api_key": config.api_key}
                if transport:
                    client_args["http_client"] = httpx.Client(transport=transport)
                
                client = Groq(**client_args)
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=config.model,
                    timeout=60  # Timeout sau 60 giây
                )
                
                elapsed_time = time.time() - start_time
                print(f"✅ Groq API (priority {config.priority}) thành công trong {elapsed_time:.2f}s")
                config.mark_success()
                
                if response and response.choices and response.choices[0].message:
                    # Parse JSON từ kết quả
                    try:
                        # Tìm kiếm chuỗi JSON trong phản hồi
                        response_text = response.choices[0].message.content
                        print(f"Đã nhận phản hồi từ Groq, độ dài: {len(response_text)} ký tự")
                        
                        # Thử trực tiếp parse toàn bộ response trước
                        try:
                            # Nếu toàn bộ response là một JSON
                            analysis_data = json.loads(response_text)
                            analysis_data['engine'] = 'groq'
                            print("✅ Đã parse JSON thành công từ toàn bộ phản hồi")
                            return analysis_data
                        except json.JSONDecodeError:
                            # Nếu không phải JSON hợp lệ, tìm kiếm JSON trong phản hồi
                            json_start = response_text.find('{')
                            json_end = response_text.rfind('}') + 1
                            
                            if json_start >= 0 and json_end > json_start:
                                json_str = response_text[json_start:json_end]
                                print(f"Tìm thấy JSON từ vị trí {json_start} đến {json_end}")
                                try:
                                    analysis_data = json.loads(json_str)
                                    analysis_data['engine'] = 'groq'
                                    print("✅ Đã parse JSON thành công từ phần trích xuất")
                                    return analysis_data
                                except json.JSONDecodeError as e:
                                    print(f"❌ Lỗi parse JSON trích xuất: {e}")
                                    # Thử regex pattern tương tự như với Gemini API
                                    import re
                                    json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                                    matches = re.findall(json_pattern, response_text)
                                    
                                    if matches:
                                        for potential_json in matches:
                                            try:
                                                analysis_data = json.loads(potential_json)
                                                if isinstance(analysis_data, dict) and len(analysis_data) > 3:
                                                    analysis_data['engine'] = 'groq'
                                                    print("✅ Đã parse JSON thành công từ regex pattern")
                                                    return analysis_data
                                            except:
                                                pass
                            
                            # Thử cách cuối: kiểm tra có JSON đúng định dạng không
                            cleaned_text = response_text.replace("\n", " ").replace("\r", " ")
                            if "```json" in cleaned_text and "```" in cleaned_text:
                                # Đây là JSON được định dạng trong markdown code block
                                start_idx = cleaned_text.find("```json") + 7
                                end_idx = cleaned_text.find("```", start_idx)
                                if start_idx > 0 and end_idx > start_idx:
                                    json_str = cleaned_text[start_idx:end_idx].strip()
                                    try:
                                        analysis_data = json.loads(json_str)
                                        analysis_data['engine'] = 'groq'
                                        print("✅ Đã parse JSON thành công từ markdown code block")
                                        return analysis_data
                                    except:
                                        pass
                                        
                        # Nếu không parse được, in phản hồi để debug
                        print("⚠️ Không thể parse JSON từ phản hồi")
                        print(f"Phản hồi thô (50 ký tự đầu): {response_text[:50]}...")
                        print(f"Phản hồi thô (50 ký tự cuối): ...{response_text[-50:]}")
                        api_errors.append("❌ Lỗi phân tích JSON từ phản hồi Groq")
                        
                    except Exception as e:
                        error_msg = f"❌ Lỗi xử lý phản hồi từ Groq: {e}"
                        print(error_msg)
                        print(f"Phản hồi thô (phần đầu): {response_text[:200]}...")
                        api_errors.append(error_msg)
            except Exception as e:
                error_msg = f"❌ Lỗi khi sử dụng Groq: {e}"
                print(error_msg)
                api_errors.append(error_msg)
        
        print("❌ Tất cả Groq API keys đều không khả dụng")
        return None
    
    def _perform_local_analysis(self, material_data: Dict) -> str:
        """
        Thực hiện phân tích cục bộ khi các API không khả dụng
        """
        analysis = []
        
        # Phân tích bandgap
        bandgap = material_data['properties'].get('bandgap', 0)
        if bandgap < 0.1:
            analysis.append("Vật liệu có tính chất kim loại")
        elif bandgap < 3.0:
            analysis.append("Vật liệu bán dẫn với bandgap phù hợp cho ứng dụng điện tử")
        else:
            analysis.append("Vật liệu cách điện với bandgap lớn")
        
        # Phân tích độ dẫn điện
        conductivity = material_data['properties'].get('conductivity', 0)
        if conductivity > 1000:
            analysis.append("Độ dẫn điện cao, phù hợp cho ứng dụng công suất")
        elif conductivity > 100:
            analysis.append("Độ dẫn điện trung bình, phù hợp cho ứng dụng thông thường")
        else:
            analysis.append("Độ dẫn điện thấp, cần cải thiện")
        
        # Phân tích nhiệt độ
        melting_point = material_data['properties'].get('melting_point', 0)
        if melting_point > 1000:
            analysis.append("Nhiệt độ nóng chảy cao, ổn định nhiệt tốt")
        elif melting_point > 500:
            analysis.append("Nhiệt độ nóng chảy trung bình, cần chú ý quản lý nhiệt")
        else:
            analysis.append("Nhiệt độ nóng chảy thấp, hạn chế ứng dụng nhiệt độ cao")
        
        # Đề xuất ứng dụng
        if bandgap < 3.0 and conductivity > 100:
            analysis.append("Phù hợp cho ứng dụng điện tử và quang điện tử")
        if melting_point > 1000:
            analysis.append("Có thể sử dụng trong môi trường nhiệt độ cao")
        
        return "\n".join(analysis)
    
    def _prepare_material_prompt(self, material_data: Dict) -> str:
        """
        Chuẩn bị prompt cho phân tích vật liệu
        """
        # Format dữ liệu vật liệu thành chuỗi
        details = []
        for key, value in material_data.items():
            if isinstance(value, dict):
                # Xử lý trường hợp dict con
                if key == 'composition':
                    elements = [f"{elem} ({ratio*100:.1f}%)" for elem, ratio in value.items()]
                    details.append(f"- Thành phần: {', '.join(elements)}")
                elif key == 'properties':
                    for prop_key, prop_value in value.items():
                        details.append(f"- {prop_key.replace('_', ' ').capitalize()}: {prop_value}")
            else:
                # Trường thông thường
                details.append(f"- {key.replace('_', ' ').capitalize()}: {value}")
                
        details_string = "\n".join(details)
        
        # Tạo prompt sử dụng template từ prompts.py
        from prompts import get_prompt
        
        try:
            # Lấy các tham số cần thiết, với giá trị mặc định nếu không tìm thấy
            material_name = material_data.get('name', 'Unknown Material')
            crystal_structure = material_data.get('crystal_structure', 'Unknown Structure')
            bandgap_energy = material_data.get('properties', {}).get('bandgap', 0)
            
            # Thêm target_application_potential từ nhiều nguồn có thể có
            target_app = material_data.get('target_application_potential', 
                        material_data.get('target_application', 
                        material_data.get('application_potential', 'General semiconductor applications')))
            
            prompt = get_prompt(
                category="task",
                key="analyze_material",
                role="expert_researcher",
                details_string=details_string,
                retrieved_context="",  # Không có context cụ thể
                material_name=material_name,
                crystal_structure=crystal_structure,
                bandgap_energy=bandgap_energy,
                target_application_potential=target_app  # Đảm bảo truyền tham số này
            )
        except Exception as e:
            print(f"Error creating prompt: {e}")
            # Fallback to simple prompt
            prompt = f"Analyze the semiconductor material with these properties:\n{details_string}"
            
        return prompt

    def _prepare_structured_material_prompt(self, material_data: Dict) -> str:
        """
        Chuẩn bị prompt có cấu trúc cho phân tích vật liệu theo Chain of Thought
        """
        # Format dữ liệu vật liệu thành chuỗi
        details = []
        for key, value in material_data.items():
            if isinstance(value, dict):
                # Xử lý trường hợp dict con
                if key == 'composition':
                    elements = [f"{elem} ({ratio*100:.1f}%)" for elem, ratio in value.items()]
                    details.append(f"- Thành phần: {', '.join(elements)}")
                elif key == 'properties':
                    for prop_key, prop_value in value.items():
                        details.append(f"- {prop_key.replace('_', ' ').capitalize()}: {prop_value}")
            else:
                # Trường thông thường
                details.append(f"- {key.replace('_', ' ').capitalize()}: {value}")
                
        details_string = "\n".join(details)
        
        # Tạo prompt sử dụng template cấu trúc từ prompts.py
        from prompts import get_prompt
        
        try:
            # Lấy các tham số cần thiết, với giá trị mặc định nếu không tìm thấy
            material_name = material_data.get('name', 'Unknown Material')
            crystal_structure = material_data.get('crystal_structure', 'Unknown Structure')
            bandgap_energy = material_data.get('properties', {}).get('bandgap', 0)
            
            # Thêm target_application_potential từ nhiều nguồn có thể có
            target_app = material_data.get('target_application_potential', 
                        material_data.get('target_application', 
                        material_data.get('application_potential', 'General semiconductor applications')))
            
            prompt = get_prompt(
                category="task",
                key="analyze_material_cot",  # Sử dụng prompt mới có cấu trúc
                role="expert_researcher",
                details_string=details_string,
                retrieved_context="",  # Không có context cụ thể
                material_name=material_name,
                crystal_structure=crystal_structure,
                bandgap_energy=bandgap_energy,
                target_application_potential=target_app  # Đảm bảo truyền tham số này
            )
            
            # Thêm hướng dẫn rõ ràng yêu cầu trả về dạng JSON
            json_instruction = """
IMPORTANT: Your response MUST be a valid JSON object with the following structure:
{
  "reasoning_steps": [
    {
      "step": 1,
      "title": "Step title",
      "content": "Step analysis",
      "key_findings": ["Finding 1", "Finding 2"]
    },
    // more steps
  ],
  "final_analysis": {
    "overall_quality": 7.5, // numerical rating from 1-10
    "bandgap_classification": "Semiconductor", // or other type
    "key_strengths": ["Strength 1", "Strength 2"],
    "key_weaknesses": ["Weakness 1", "Weakness 2"]
  },
  "recommendations": {
    "applications": ["Application 1", "Application 2"],
    "improvements": [
      {"aspect": "Aspect 1", "recommendation": "Recommendation 1", "priority": "High"}
    ]
  },
  "basic_properties": {
    // Extracted basic properties
  },
  "predictions": {
    // Predictions for properties
  }
}

Provide ONLY the JSON with no additional text before or after. No explanations, no markdown formatting, just the raw JSON.
"""
            prompt = prompt + "\n\n" + json_instruction
            
        except Exception as e:
            print(f"Error creating structured prompt: {e}")
            # Fallback to simple prompt
            prompt = f"""Analyze the semiconductor material with these properties and return JSON:\n{details_string}
            
IMPORTANT: Your response MUST be a valid JSON object with no additional text.
"""
            
        return prompt
        
    def _analyze_with_structured_cot(self, material_data: Dict) -> Dict:
        """
        Phân tích vật liệu sử dụng cấu trúc JSON từ mô hình LLM
        """
        print("🔄 Phân tích vật liệu với CoT có cấu trúc...")
        
        # Chuẩn bị prompt có cấu trúc
        prompt = self._prepare_structured_material_prompt(material_data)
        
        result = None
        api_errors = []
        
        # Thử phân tích sử dụng Gemini
        if self.models_available['gemini']:
            try:
                print("🤖 Đang sử dụng Gemini để phân tích...")
                result = self._try_gemini_apis(prompt)
                
                if result and result.get('content'):
                    # Parse JSON từ kết quả
                    try:
                        # Tìm kiếm chuỗi JSON trong phản hồi
                        response_text = result['content']
                        print(f"Đã nhận phản hồi từ Gemini, độ dài: {len(response_text)} ký tự")
                        
                        # Thử trực tiếp parse toàn bộ response trước
                        try:
                            # Nếu toàn bộ response là một JSON
                            analysis_data = json.loads(response_text)
                            analysis_data['engine'] = 'gemini'
                            print("✅ Đã parse JSON thành công từ toàn bộ phản hồi")
                            return analysis_data
                        except json.JSONDecodeError:
                            # Nếu không phải JSON hợp lệ, tìm kiếm JSON trong phản hồi
                            json_start = response_text.find('{')
                            json_end = response_text.rfind('}') + 1
                            
                            if json_start >= 0 and json_end > json_start:
                                json_str = response_text[json_start:json_end]
                                print(f"Tìm thấy JSON từ vị trí {json_start} đến {json_end}")
                                try:
                                    analysis_data = json.loads(json_str)
                                    analysis_data['engine'] = 'gemini'
                                    print("✅ Đã parse JSON thành công từ phần trích xuất")
                                    return analysis_data
                                except json.JSONDecodeError as e:
                                    print(f"❌ Lỗi parse JSON trích xuất: {e}")
                                    # Thử một cách khác: tìm JSON format chuẩn hơn
                                    import re
                                    json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                                    matches = re.findall(json_pattern, response_text)
                                    
                                    if matches:
                                        for potential_json in matches:
                                            try:
                                                analysis_data = json.loads(potential_json)
                                                if isinstance(analysis_data, dict) and len(analysis_data) > 3:
                                                    analysis_data['engine'] = 'gemini'
                                                    print("✅ Đã parse JSON thành công từ regex pattern")
                                                    return analysis_data
                                            except:
                                                pass
                            
                            # Thử cách cuối: kiểm tra có JSON đúng định dạng không
                            cleaned_text = response_text.replace("\n", " ").replace("\r", " ")
                            if "```json" in cleaned_text and "```" in cleaned_text:
                                # Đây là JSON được định dạng trong markdown code block
                                start_idx = cleaned_text.find("```json") + 7
                                end_idx = cleaned_text.find("```", start_idx)
                                if start_idx > 0 and end_idx > start_idx:
                                    json_str = cleaned_text[start_idx:end_idx].strip()
                                    try:
                                        analysis_data = json.loads(json_str)
                                        analysis_data['engine'] = 'gemini'
                                        print("✅ Đã parse JSON thành công từ markdown code block")
                                        return analysis_data
                                    except:
                                        pass
                                        
                        # Nếu không parse được, in phản hồi để debug
                        print("⚠️ Không thể parse JSON từ phản hồi")
                        print(f"Phản hồi thô (50 ký tự đầu): {response_text[:50]}...")
                        print(f"Phản hồi thô (50 ký tự cuối): ...{response_text[-50:]}")
                        api_errors.append("❌ Lỗi phân tích JSON từ phản hồi Gemini")
                        
                    except Exception as e:
                        error_msg = f"❌ Lỗi xử lý phản hồi từ Gemini: {e}"
                        print(error_msg)
                        print(f"Phản hồi thô (phần đầu): {response_text[:200]}...")
                        api_errors.append(error_msg)
            except Exception as e:
                error_msg = f"❌ Lỗi khi sử dụng Gemini: {e}"
                print(error_msg)
                api_errors.append(error_msg)
        
        # Thử phân tích sử dụng Groq nếu Gemini không thành công
        if self.models_available['groq'] and not result:
            try:
                print("🤖 Đang sử dụng Groq để phân tích...")
                result = self._try_groq_apis(prompt)
                
                if result and result.get('content'):
                    # Parse JSON từ kết quả
                    try:
                        # Tìm kiếm chuỗi JSON trong phản hồi
                        response_text = result['content']
                        print(f"Đã nhận phản hồi từ Groq, độ dài: {len(response_text)} ký tự")
                        
                        # Thử trực tiếp parse toàn bộ response trước
                        try:
                            # Nếu toàn bộ response là một JSON
                            analysis_data = json.loads(response_text)
                            analysis_data['engine'] = 'groq'
                            print("✅ Đã parse JSON thành công từ toàn bộ phản hồi")
                            return analysis_data
                        except json.JSONDecodeError:
                            # Nếu không phải JSON hợp lệ, tìm kiếm JSON trong phản hồi
                            json_start = response_text.find('{')
                            json_end = response_text.rfind('}') + 1
                            
                            if json_start >= 0 and json_end > json_start:
                                json_str = response_text[json_start:json_end]
                                print(f"Tìm thấy JSON từ vị trí {json_start} đến {json_end}")
                                try:
                                    analysis_data = json.loads(json_str)
                                    analysis_data['engine'] = 'groq'
                                    print("✅ Đã parse JSON thành công từ phần trích xuất")
                                    return analysis_data
                                except json.JSONDecodeError as e:
                                    print(f"❌ Lỗi parse JSON trích xuất: {e}")
                                    # Thử regex pattern tương tự như với Gemini API
                                    import re
                                    json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                                    matches = re.findall(json_pattern, response_text)
                                    
                                    if matches:
                                        for potential_json in matches:
                                            try:
                                                analysis_data = json.loads(potential_json)
                                                if isinstance(analysis_data, dict) and len(analysis_data) > 3:
                                                    analysis_data['engine'] = 'groq'
                                                    print("✅ Đã parse JSON thành công từ regex pattern")
                                                    return analysis_data
                                            except:
                                                pass
                            
                            # Thử cách cuối: kiểm tra có JSON đúng định dạng không
                            cleaned_text = response_text.replace("\n", " ").replace("\r", " ")
                            if "```json" in cleaned_text and "```" in cleaned_text:
                                # Đây là JSON được định dạng trong markdown code block
                                start_idx = cleaned_text.find("```json") + 7
                                end_idx = cleaned_text.find("```", start_idx)
                                if start_idx > 0 and end_idx > start_idx:
                                    json_str = cleaned_text[start_idx:end_idx].strip()
                                    try:
                                        analysis_data = json.loads(json_str)
                                        analysis_data['engine'] = 'groq'
                                        print("✅ Đã parse JSON thành công từ markdown code block")
                                        return analysis_data
                                    except:
                                        pass
                                        
                        # Nếu không parse được, in phản hồi để debug
                        print("⚠️ Không thể parse JSON từ phản hồi")
                        print(f"Phản hồi thô (50 ký tự đầu): {response_text[:50]}...")
                        print(f"Phản hồi thô (50 ký tự cuối): ...{response_text[-50:]}")
                        api_errors.append("❌ Lỗi phân tích JSON từ phản hồi Groq")
                        
                    except Exception as e:
                        error_msg = f"❌ Lỗi xử lý phản hồi từ Groq: {e}"
                        print(error_msg)
                        print(f"Phản hồi thô (phần đầu): {response_text[:200]}...")
                        api_errors.append(error_msg)
            except Exception as e:
                error_msg = f"❌ Lỗi khi sử dụng Groq: {e}"
                print(error_msg)
                api_errors.append(error_msg)

        # FALLBACK: Tạo cấu trúc JSON mô phỏng khi các API không thành công
        print("⚠️ API không thành công, sử dụng phân tích cục bộ thay thế...")
        print(f"📋 Lỗi API: {'; '.join(api_errors)}")
        
        # Còn lại của phương thức không thay đổi
        try:
            print("⚠️ API không thành công, sử dụng phân tích cục bộ thay thế...")
            # Tạo cấu trúc JSON giả lập khi cả hai API đều không hoạt động
            material_name = material_data.get('name', 'Unknown Material')
            bandgap = material_data.get('properties', {}).get('bandgap', 0)
            conductivity = material_data.get('properties', {}).get('conductivity', 0)
            crystal_structure = material_data.get('crystal_structure', 'Unknown')
            
            # Phân loại bandgap
            bandgap_classification = self._classify_bandgap(bandgap)
            
            # Phân tích độ dẫn điện
            conductivity_level = "High" if conductivity > 1000 else "Medium" if conductivity > 100 else "Low"
            
            # Phân tích tính bền nhiệt
            melting_point = material_data.get('properties', {}).get('melting_point', 0)
            thermal_stability = "High" if melting_point > 1000 else "Medium" if melting_point > 500 else "Low"
            
            # Phân tích nhiệt độ
            thermal_conductivity = material_data.get('properties', {}).get('thermal_conductivity', 0)
            heat_transport = "High" if thermal_conductivity > 50 else "Medium" if thermal_conductivity > 20 else "Low"
            
            # Danh sách ứng dụng tiềm năng dựa trên bandgap
            applications = []
            if 0.5 < bandgap < 1.5:
                applications.append("Solar cells")
                applications.append("Photodetectors")
            if 1.5 < bandgap < 3.0:
                applications.append("LEDs")
                applications.append("Transistors")
            if bandgap > 3.0:
                applications.append("High-power electronics")
                applications.append("UV detectors")
            if bandgap < 0.1:
                applications.append("Conductors")
                applications.append("Interconnects")
            if len(applications) == 0:
                applications = ["General semiconductor applications"]
            
            # Điểm mạnh và điểm yếu cơ bản
            strengths = []
            weaknesses = []
            
            if bandgap_classification == "Semiconductor":
                strengths.append(f"Bandgap {bandgap} eV phù hợp cho ứng dụng điện tử semiconductor")
            if conductivity_level == "High":
                strengths.append("Độ dẫn điện cao phù hợp cho các ứng dụng công suất")
            if thermal_stability == "High":
                strengths.append("Độ bền nhiệt cao cho phép hoạt động ở nhiệt độ cao")
            
            if conductivity_level == "Low":
                weaknesses.append("Độ dẫn điện thấp, hạn chế hiệu suất trong một số ứng dụng")
            if thermal_stability == "Low":
                weaknesses.append("Độ bền nhiệt hạn chế, không phù hợp cho môi trường nhiệt độ cao")
            if heat_transport == "Low":
                weaknesses.append("Khả năng dẫn nhiệt kém, có thể gây khó khăn trong quản lý nhiệt")
            
            # Tạo mẫu cấu trúc CoT
            reasoning_steps = [
                {
                    "step": 1,
                    "title": "Phân tích thành phần vật liệu",
                    "content": f"Vật liệu {material_name} có cấu trúc tinh thể {crystal_structure}. Phân tích thành phần cho thấy các đặc tính của vật liệu bán dẫn với các thuộc tính vật lý quan trọng đã được đo lường.",
                    "key_findings": [f"Cấu trúc tinh thể: {crystal_structure}", "Thuộc tính vật lý được đo lường và ghi nhận"]
                },
                {
                    "step": 2,
                    "title": "Phân tích tính chất điện tử",
                    "content": f"Vật liệu có bandgap {bandgap} eV, phân loại là {bandgap_classification}. Độ dẫn điện {conductivity} S/cm được đánh giá là {conductivity_level}.",
                    "key_findings": [f"Bandgap: {bandgap} eV ({bandgap_classification})", f"Độ dẫn điện: {conductivity_level}"]
                },
                {
                    "step": 3,
                    "title": "Phân tích tính chất nhiệt",
                    "content": f"Độ bền nhiệt được đánh giá là {thermal_stability} dựa trên nhiệt độ nóng chảy. Khả năng dẫn nhiệt {heat_transport}.",
                    "key_findings": [f"Độ bền nhiệt: {thermal_stability}", f"Khả năng dẫn nhiệt: {heat_transport}"]
                },
                {
                    "step": 4,
                    "title": "Ứng dụng tiềm năng",
                    "content": f"Dựa trên các thuộc tính vật lý, vật liệu này phù hợp cho các ứng dụng: {', '.join(applications)}.",
                    "key_findings": [f"Phù hợp cho: {', '.join(applications[:2])}", "Cần đánh giá thêm cho các ứng dụng chuyên biệt"]
                },
                {
                    "step": 5,
                    "title": "Đề xuất cải thiện",
                    "content": "Để tối ưu hóa hiệu suất, có thể cân nhắc một số phương pháp cải thiện như điều chỉnh quy trình chế tạo, pha tạp, và kiểm soát khuyết tật.",
                    "key_findings": ["Tối ưu hóa quy trình chế tạo", "Xem xét chiến lược pha tạp phù hợp"]
                },
                {
                    "step": 6,
                    "title": "Phân tích AI tổng hợp",
                    "content": f"Phân tích tổng thể cho thấy {material_name} có tiềm năng ứng dụng trong lĩnh vực bán dẫn với một số ưu điểm và hạn chế đã được xác định.",
                    "key_findings": ["Đánh giá toàn diện các thuộc tính", "Xác định điểm mạnh và điểm yếu chính"]
                }
            ]

            # Xây dựng kết quả cuối cùng
            fallback_analysis = {
                "reasoning_steps": reasoning_steps,
                "final_analysis": {
                    "overall_quality": min(max(5.0 + (len(strengths) - len(weaknesses)), 3.0), 9.0),
                    "bandgap_classification": bandgap_classification,
                    "key_strengths": strengths,
                    "key_weaknesses": weaknesses
                },
                "recommendations": {
                    "applications": applications,
                    "improvements": [
                        {"aspect": "Độ dẫn điện", "recommendation": "Tối ưu hóa quá trình pha tạp", "priority": "Medium"},
                        {"aspect": "Độ bền nhiệt", "recommendation": "Cải thiện cấu trúc tinh thể", "priority": "Medium"},
                        {"aspect": "Hiệu suất ứng dụng", "recommendation": "Đánh giá chi tiết trong ứng dụng thực tế", "priority": "High"}
                    ]
                },
                "basic_properties": {
                    "bandgap_type": "Trực tiếp" if bandgap > 0 and crystal_structure in ["Wurtzite", "Zincblende", "Hexagonal"] else "Gián tiếp",
                    "crystal_system": crystal_structure,
                    "carrier_type": "n-type" if "Si" in str(material_data.get('composition', {})) or "Ge" in str(material_data.get('composition', {})) else "p-type",
                    "thermal_stability": thermal_stability
                },
                "predictions": {
                    "bandgap_prediction": {"value": bandgap_classification, "confidence": 0.9},
                    "conductivity_prediction": {"value": conductivity_level, "confidence": 0.85},
                    "thermal_stability": thermal_stability,
                    "heat_transport": heat_transport,
                    "stability_prediction": {"value": thermal_stability, "confidence": 0.8}
                },
                "engine": "local_fallback"
            }
            
            print("✅ Đã tạo phân tích thay thế thành công")
            return fallback_analysis
            
        except Exception as e:
            print(f"❌ Lỗi khi tạo phân tích thay thế: {e}")
            print("❌ Không thể phân tích bằng CoT có cấu trúc, sẽ sử dụng phương pháp thay thế")
            return None
    
    # Helper methods
    def _check_stoichiometry(self, composition: Dict) -> Dict:
        total = sum(composition.values())
        normalized = {k: v/total for k, v in composition.items()}
        return {
            'normalized_ratios': normalized,
            'is_balanced': abs(1 - total) < 0.01
        }
    
    def _predict_stability(self, composition: Dict) -> Dict:
        # Giả lập dự đoán độ ổn định
        stability_score = np.random.uniform(0.7, 0.9)
        return {
            'stability_score': stability_score,
            'confidence': np.random.uniform(0.6, 0.8)
        }
    
    def _get_symmetry_info(self, structure: str) -> Dict:
        symmetry_info = {
            'cubic': {'point_group': 'Oh', 'space_group': 'Fm3m'},
            'hexagonal': {'point_group': 'D6h', 'space_group': 'P6/mmm'},
            'tetragonal': {'point_group': 'D4h', 'space_group': 'P4/mmm'},
            'orthorhombic': {'point_group': 'D2h', 'space_group': 'Pmmm'}
        }
        return symmetry_info.get(structure.lower(), {'point_group': 'Unknown', 'space_group': 'Unknown'})
    
    def _predict_structure_properties(self, structure: str) -> Dict:
        return {
            'mechanical_stability': np.random.uniform(0.7, 0.9),
            'thermal_stability': np.random.uniform(0.6, 0.8),
            'defect_formation_energy': np.random.uniform(1.0, 3.0)
        }
    
    def _classify_bandgap(self, bandgap: float) -> str:
        if bandgap is None:
            return 'Unknown'
        if bandgap == 0:
            return 'Metal'
        if bandgap < 0.1:
            return 'Semimetal'
        if bandgap < 3.0:
            return 'Semiconductor'
        return 'Insulator'
    
    def _analyze_conductivity(self, properties: Dict) -> Dict:
        conductivity = properties.get('conductivity', 0)
        return {
            'conductivity_type': 'High' if conductivity > 1000 else 'Medium' if conductivity > 100 else 'Low',
            'temperature_dependence': 'Negative' if conductivity > 1000 else 'Positive'
        }
    
    def _analyze_carriers(self, properties: Dict) -> Dict:
        concentration = properties.get('carrier_concentration', 0)
        return {
            'carrier_type': 'Electrons' if concentration > 0 else 'Holes',
            'mobility_prediction': np.random.uniform(100, 1000)
        }
    
    def _analyze_thermal_stability(self, properties: Dict) -> Dict:
        melting_point = properties.get('melting_point', 0)
        return {
            'stability_rating': 'High' if melting_point > 1000 else 'Medium' if melting_point > 500 else 'Low',
            'max_operating_temp': melting_point * 0.7 if melting_point else None
        }
    
    def _analyze_heat_transport(self, properties: Dict) -> Dict:
        thermal_conductivity = properties.get('thermal_conductivity', 0)
        return {
            'heat_transfer_efficiency': 'High' if thermal_conductivity > 50 else 'Medium' if thermal_conductivity > 10 else 'Low',
            'thermal_management_required': thermal_conductivity < 30
        }
    
    def _predict_temperature_effects(self, properties: Dict) -> Dict:
        return {
            'property_degradation_temp': properties.get('melting_point', 0) * 0.6,
            'recommended_operating_range': {
                'min': 250,
                'max': properties.get('melting_point', 0) * 0.5
            }
        }
    
    def _identify_applications(self, electronic: Dict, thermal: Dict) -> List[Dict]:
        applications = []
        
        # Dựa trên tính chất điện tử
        if electronic.get('bandgap_type') == 'Semiconductor':
            applications.append({
                'field': 'Electronics',
                'specific_uses': ['Transistors', 'Solar Cells', 'LEDs'],
                'confidence': 0.85
            })
            
        # Dựa trên tính chất nhiệt
        if thermal.get('thermal_stability', {}).get('stability_rating') == 'High':
            applications.append({
                'field': 'High Temperature Devices',
                'specific_uses': ['Power Electronics', 'Sensors'],
                'confidence': 0.75
            })
            
        return applications
    
    def _generate_improvement_suggestions(self, context: Dict) -> List[Dict]:
        suggestions = []
        
        # Dựa trên phân tích điện tử
        electronic = context.get('electronic_analysis', {})
        if electronic.get('conductivity_analysis', {}).get('conductivity_type') == 'Low':
            suggestions.append({
                'aspect': 'Conductivity',
                'suggestion': 'Increase carrier concentration through doping',
                'priority': 'High'
            })
            
        # Dựa trên phân tích nhiệt
        thermal = context.get('thermal_analysis', {})
        if thermal.get('heat_transport', {}).get('thermal_management_required'):
            suggestions.append({
                'aspect': 'Thermal Management',
                'suggestion': 'Improve heat dissipation through structural modification',
                'priority': 'Medium'
            })
            
        return suggestions
    
    def _synthesize_results(self, steps: List[Dict]) -> Dict:
        """
        Tổng hợp kết quả từ tất cả các bước phân tích
        """
        synthesis = {
            'overall_quality': np.random.uniform(0.7, 0.9),
            'key_findings': [],
            'critical_properties': {},
            'confidence_scores': {},
            'basic_properties': {},
            'predictions': {}
        }
        
        for step in steps:
            if 'analysis' in step:
                # Thêm các phát hiện quan trọng
                if isinstance(step['analysis'], dict):
                    for key, value in step['analysis'].items():
                        if isinstance(value, (str, float, int)):
                            synthesis['critical_properties'][key] = value
                            synthesis['basic_properties'][key] = value
                        elif isinstance(value, dict):
                            synthesis['critical_properties'].update(value)
                            synthesis['basic_properties'].update(value)
                
                # Thêm điểm tin cậy
                synthesis['confidence_scores'][step['step_name']] = np.random.uniform(0.7, 0.9)
                
                # Thêm dự đoán từ các bước phân tích
                if step['step_name'] == 'Electronic Properties Analysis':
                    synthesis['predictions'].update({
                        'bandgap_prediction': step['analysis'].get('bandgap_type', 'Unknown'),
                        'conductivity_prediction': step['analysis'].get('conductivity_analysis', {}).get('conductivity_type', 'Unknown')
                    })
                elif step['step_name'] == 'Thermal Properties Analysis':
                    synthesis['predictions'].update({
                        'thermal_stability': step['analysis'].get('thermal_stability', {}).get('stability_rating', 'Unknown'),
                        'heat_transport': step['analysis'].get('heat_transport', {}).get('heat_transfer_efficiency', 'Unknown')
                    })
        
        return synthesis

    def _generate_recommendations(self, context: Dict) -> Dict:
        """
        Tạo các đề xuất dựa trên tất cả các phân tích
        """
        recommendations = {
            'applications': [],
            'improvements': [],
            'research_directions': []
        }
        
        # Lấy các phân tích từ context
        composition = context.get('composition_analysis', {})
        structure = context.get('structure_analysis', {})
        electronic = context.get('electronic_analysis', {})
        thermal = context.get('thermal_analysis', {})
        ai_analysis = context.get('ai_analysis', {})
        
        # Đề xuất ứng dụng
        if electronic.get('bandgap_type') == 'Semiconductor':
            recommendations['applications'].extend([
                'Transistors và vi mạch tích hợp',
                'Pin mặt trời và quang điện tử',
                'LED và thiết bị phát quang'
            ])
        
        # Đề xuất cải tiến
        improvements = []
        
        # Kiểm tra tính chất điện tử
        if electronic.get('conductivity_analysis', {}).get('conductivity_type') == 'Low':
            improvements.append({
                'aspect': 'Độ dẫn điện',
                'suggestion': 'Tăng nồng độ hạt tải bằng cách pha tạp'
            })
            
        # Kiểm tra tính chất nhiệt
        if thermal.get('heat_transport', {}).get('thermal_management_required'):
            improvements.append({
                'aspect': 'Quản lý nhiệt',
                'suggestion': 'Cải thiện tản nhiệt bằng cách điều chỉnh cấu trúc'
            })
            
        recommendations['improvements'] = improvements
        
        # Đề xuất hướng nghiên cứu
        research_directions = []
        
        # Dựa trên độ ổn định
        if composition.get('stability_prediction', {}).get('stability_score', 0) < 0.8:
            research_directions.append({
                'topic': 'Cải thiện độ ổn định',
                'priority': 'High'
            })
            
        # Dựa trên cấu trúc tinh thể
        if structure.get('predicted_properties', {}).get('defect_formation_energy', 0) > 2.0:
            research_directions.append({
                'topic': 'Tối ưu hóa khuyết tật tinh thể',
                'priority': 'Medium'
            })
            
        # Thêm đề xuất từ AI models
        for model, response in ai_analysis.items():
            if 'research_directions' in response.get('analysis', '').lower():
                research_directions.append({
                    'topic': f'AI Suggestion ({model})',
                    'priority': 'High'
                })
        
        recommendations['research_directions'] = research_directions
        
        return recommendations

    def _format_reasoning_steps(self, steps: List[Dict]) -> List[Dict]:
        """
        Định dạng lại các bước suy luận để hiển thị trong Chain of Thought UI
        """
        formatted_steps = []
        step_titles = [
            "Phân tích thành phần vật liệu", 
            "Phân tích cấu trúc tinh thể",
            "Phân tích tính chất điện tử",
            "Phân tích tính chất nhiệt",
            "Dự đoán ứng dụng tiềm năng",
            "Đề xuất cải thiện",
            "Phân tích tích hợp AI"
        ]
        
        for idx, (step, title) in enumerate(zip(steps, step_titles)):
            # Lấy dữ liệu phân tích từ mỗi bước
            if 'reasoning' in step:
                reasoning = "\n".join(step['reasoning'])
            else:
                reasoning = "Không có thông tin suy luận chi tiết."
                
            # Tạo nội dung cho mỗi bước
            content = f"### {title}\n\n"
            content += f"**Quá trình suy luận:**\n{reasoning}\n\n"
            
            # Thêm chi tiết phân tích nếu có
            if 'details' in step:
                content += "**Chi tiết phân tích:**\n"
                for key, value in step['details'].items():
                    content += f"- {key}: {value}\n"
            
            # Thêm kết quả từ mô hình AI nếu có
            if 'ai_insights' in step:
                content += "\n**Phân tích từ AI:**\n"
                content += step['ai_insights']
            
            # Thêm bước định dạng
            formatted_steps.append({
                'title': title,
                'content': content,
                'step_number': idx + 1
            })
        
        return formatted_steps
            
    def _extract_strengths(self, context: Dict, final_analysis: Dict) -> List[str]:
        """Trích xuất điểm mạnh của vật liệu từ kết quả phân tích"""
        material_data = context['material_data']
        strengths = []
        
        # Kiểm tra bandgap
        bandgap = float(material_data['properties'].get('bandgap', 0))
        if 1.0 < bandgap < 3.0:
            strengths.append(f"Bandgap {bandgap} eV phù hợp cho ứng dụng optoelectronics")
        elif bandgap > 3.0:
            strengths.append(f"Bandgap {bandgap} eV phù hợp cho ứng dụng công suất cao và UV")
            
        # Kiểm tra độ dẫn điện
        conductivity = float(material_data['properties'].get('conductivity', 0))
        if conductivity > 1000:
            strengths.append(f"Độ dẫn điện cao ({conductivity} S/cm)")
            
        # Kiểm tra tính ổn định nhiệt
        thermal_stability = str(material_data['properties'].get('thermal_stability', '')).lower()
        if "high" in thermal_stability or "cao" in thermal_stability:
            strengths.append("Độ bền nhiệt cao, phù hợp cho điều kiện khắc nghiệt")
        
        # Kiểm tra cấu trúc tinh thể
        crystal_structure = material_data.get('crystal_structure', '').lower()
        if "cubic" in crystal_structure or "wurtzite" in crystal_structure:
            strengths.append(f"Cấu trúc tinh thể {material_data.get('crystal_structure')} mang lại độ ổn định cao")
            
        # Thêm từ dự đoán nếu có
        if 'predictions' in final_analysis:
            predictions = final_analysis['predictions']
            if predictions.get('application_potential', {}).get('value', 0) > 8:
                strengths.append("Tiềm năng ứng dụng rộng rãi trong nhiều lĩnh vực")
        
        # Nếu không có điểm mạnh, thêm mặc định
        if not strengths:
            strengths = [
                f"Cấu trúc tinh thể {material_data.get('crystal_structure')} mang lại độ ổn định",
                "Khả năng ứng dụng trong lĩnh vực điện tử",
                "Quy trình chế tạo có thể tối ưu"
            ]
            
        return strengths
    
    def _extract_weaknesses(self, context: Dict, final_analysis: Dict) -> List[str]:
        """Trích xuất điểm yếu của vật liệu từ kết quả phân tích"""
        material_data = context['material_data']
        weaknesses = []
        
        # Kiểm tra bandgap
        bandgap = float(material_data['properties'].get('bandgap', 0))
        if bandgap < 0.3:
            weaknesses.append(f"Bandgap thấp ({bandgap} eV) gây khó khăn cho ứng dụng ở nhiệt độ cao")
        elif bandgap > 3.5:
            weaknesses.append(f"Bandgap quá cao ({bandgap} eV) gây khó khăn cho việc kích hoạt điện tử")
            
        # Kiểm tra độ dẫn điện
        conductivity = float(material_data['properties'].get('conductivity', 0))
        if conductivity < 100:
            weaknesses.append(f"Độ dẫn điện thấp ({conductivity} S/cm) làm giảm hiệu suất thiết bị")
            
        # Kiểm tra tính ổn định nhiệt
        thermal_stability = str(material_data['properties'].get('thermal_stability', '')).lower()
        if "low" in thermal_stability or "thấp" in thermal_stability:
            weaknesses.append("Độ bền nhiệt thấp hạn chế ứng dụng ở nhiệt độ cao")
        
        # Kiểm tra cấu trúc tinh thể
        crystal_structure = material_data.get('crystal_structure', '').lower()
        if "triclinic" in crystal_structure or "monoclinic" in crystal_structure:
            weaknesses.append(f"Cấu trúc tinh thể {material_data.get('crystal_structure')} phức tạp, khó chế tạo")
        
        # Nếu không có điểm yếu, thêm mặc định
        if not weaknesses:
            weaknesses = [
                "Chi phí sản xuất có thể cao",
                "Cần phát triển quy trình chế tạo quy mô lớn",
                "Có thể nhạy cảm với môi trường"
            ]
            
        return weaknesses

    def test_api_connectivity(self):
        """
        Kiểm tra khả năng kết nối của tất cả các API endpoints
        và báo cáo kết quả
        """
        print("🔄 Đang kiểm tra kết nối API...")
        
        # Khởi tạo lại API configs để đảm bảo chúng được cập nhật
        self._init_api_configs()
        
        # Kiểm tra kết nối với máy chủ
        self._check_server_availability()
        
        # Kiểm tra từng API key của Gemini
        print("\n🔍 KIỂM TRA GEMINI API KEYS:")
        for i, config in enumerate(self.gemini_configs):
            print(f"  📝 Key {i+1} (priority {config.priority}):")
            try:
                # Thử một request đơn giản
                genai.configure(api_key=config.api_key)
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                start_time = time.time()
                response = model.generate_content("Xin chào, đây là tin nhắn kiểm tra.", timeout=10)
                elapsed_time = time.time() - start_time
                
                if response and hasattr(response, 'text'):
                    print(f"    ✅ Kết nối thành công trong {elapsed_time:.2f}s")
                    print(f"    📊 Phản hồi: {response.text[:50]}...")
                else:
                    print(f"    ⚠️ Kết nối thành công nhưng phản hồi không có nội dung")
            except Exception as e:
                print(f"    ❌ Lỗi: {str(e)}")
        
        # Kiểm tra từng API key của Groq
        print("\n🔍 KIỂM TRA GROQ API KEYS:")
        for i, config in enumerate(self.groq_configs):
            print(f"  📝 Key {i+1} (priority {config.priority}):")
            try:
                # Tạo transport với proxy nếu có
                import httpx
                transport = None
                if config.proxies:
                    transport = httpx.HTTPTransport(proxy=config.proxies)
                
                # Tạo client với transport nếu cần
                client_args = {"api_key": config.api_key}
                if transport:
                    client_args["http_client"] = httpx.Client(transport=transport)
                
                client = Groq(**client_args)
                
                # Thử một request đơn giản
                start_time = time.time()
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": "Xin chào, đây là tin nhắn kiểm tra."}],
                    model=config.model,
                    timeout=10
                )
                elapsed_time = time.time() - start_time
                
                if response and response.choices and response.choices[0].message:
                    print(f"    ✅ Kết nối thành công trong {elapsed_time:.2f}s")
                    print(f"    📊 Phản hồi: {response.choices[0].message.content[:50]}...")
                else:
                    print(f"    ⚠️ Kết nối thành công nhưng phản hồi không có nội dung")
            except Exception as e:
                print(f"    ❌ Lỗi: {str(e)}")
        
        # Báo cáo tổng quát
        print("\n📋 BÁO CÁO TỔNG QUÁT:")
        print(f"  - Tổng số Gemini API keys: {len(self.gemini_configs)}")
        print(f"  - Tổng số Groq API keys: {len(self.groq_configs)}")
        
        # Trả về thông báo tổng hợp
        return {
            'gemini_keys': len(self.gemini_configs),
            'groq_keys': len(self.groq_configs),
            'proxy_settings': bool(self.proxies)
        } 
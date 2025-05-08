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

class APIConfig:
    """Cấu hình cho mỗi API endpoint"""
    def __init__(self, api_key: str, model: str, priority: int = 1):
        self.api_key = api_key
        self.model = model
        self.priority = priority
        self.last_used = None
        self.error_count = 0
        self.max_errors = 3
        self.cooldown_minutes = 5

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
        """Khởi tạo cấu hình cho nhiều API endpoints"""
        self.gemini_configs = []
        self.groq_configs = []
        
        # Load Gemini API keys
        gemini_keys = os.getenv('GEMINI_API_KEYS', '').split(',')
        for i, key in enumerate(gemini_keys):
            if key.strip():
                try:
                    # Kiểm tra key có hợp lệ không
                    genai.configure(api_key=key.strip())
                    test_model = genai.GenerativeModel('models/gemini-1.5-flash')
                    test_model.generate_content("Test")
                    
                    self.gemini_configs.append(APIConfig(
                        api_key=key.strip(),
                        model='models/gemini-1.5-flash',
                        priority=i+1
                    ))
                    print(f"✅ Gemini API key {i+1} validated successfully")
                except Exception as e:
                    print(f"⚠️ Invalid Gemini API key {i+1}: {str(e)}")
        
        # Load Groq API keys
        groq_keys = os.getenv('GROQ_API_KEYS', '').split(',')
        for i, key in enumerate(groq_keys):
            if key.strip():
                try:
                    # Kiểm tra key có hợp lệ không
                    client = Groq(api_key=key.strip())
                    client.chat.completions.create(
                        messages=[{"role": "user", "content": "Test"}],
                        model="deepseek-r1-distill-llama-70b"
                    )
                    
                    self.groq_configs.append(APIConfig(
                        api_key=key.strip(),
                        model='deepseek-r1-distill-llama-70b',
                        priority=i+1
                    ))
                    print(f"✅ Groq API key {i+1} validated successfully")
                except Exception as e:
                    print(f"⚠️ Invalid Groq API key {i+1}: {str(e)}")
        
        # Sắp xếp theo độ ưu tiên
        self.gemini_configs.sort(key=lambda x: x.priority)
        self.groq_configs.sort(key=lambda x: x.priority)
        
        # Thông báo số lượng API keys hợp lệ
        print(f"ℹ️ Found {len(self.gemini_configs)} valid Gemini API keys")
        print(f"ℹ️ Found {len(self.groq_configs)} valid Groq API keys")

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
            ai_responses['gemini'] = gemini_analysis
        
        # Thử tất cả các API endpoints cho Groq
        groq_analysis = self._try_groq_apis(prompt)
        if groq_analysis:
            ai_responses['groq'] = groq_analysis
        
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
                genai.configure(api_key=config.api_key)
                model = genai.GenerativeModel(config.model)
                response = model.generate_content(prompt)
                config.mark_success()
                
                return {
                    'analysis': response.text,
                    'confidence': 0.85,
                    'status': 'success',
                    'api_priority': config.priority
                }
            except Exception as e:
                error_msg = str(e)
                if "API_KEY_INVALID" in error_msg or "API key expired" in error_msg:
                    print(f"⚠️ Gemini API (priority {config.priority}) key expired or invalid")
                    config.error_count = config.max_errors  # Đánh dấu key không hợp lệ
                else:
                    print(f"⚠️ Gemini API (priority {config.priority}) failed: {error_msg}")
                config.mark_error()
        
        return None

    def _try_groq_apis(self, prompt: str) -> Dict:
        """Thử tất cả các API endpoints của Groq"""
        for config in self.groq_configs:
            if not config.is_available():
                continue
                
            try:
                # Khởi tạo client không có proxies
                client = Groq(api_key=config.api_key)
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=config.model
                )
                config.mark_success()
                
                return {
                    'analysis': response.choices[0].message.content,
                    'confidence': 0.8,
                    'status': 'success',
                    'api_priority': config.priority
                }
            except Exception as e:
                error_msg = str(e)
                if "API key" in error_msg.lower():
                    print(f"⚠️ Groq API (priority {config.priority}) key invalid")
                    config.error_count = config.max_errors
                else:
                    print(f"⚠️ Groq API (priority {config.priority}) failed: {error_msg}")
                config.mark_error()
        
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
        except Exception as e:
            print(f"Error creating structured prompt: {e}")
            # Fallback to simple prompt
            prompt = f"Analyze the semiconductor material with these properties and return JSON:\n{details_string}"
            
        return prompt
        
    def _analyze_with_structured_cot(self, material_data: Dict) -> Dict:
        """
        Phân tích vật liệu sử dụng cấu trúc JSON từ mô hình LLM
        """
        print("🔄 Phân tích vật liệu với CoT có cấu trúc...")
        
        # Chuẩn bị prompt có cấu trúc
        prompt = self._prepare_structured_material_prompt(material_data)
        
        result = None
        
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
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            json_str = response_text[json_start:json_end]
                            analysis_data = json.loads(json_str)
                            analysis_data['engine'] = 'gemini'
                            return analysis_data
                    except json.JSONDecodeError as e:
                        print(f"❌ Lỗi phân tích JSON từ phản hồi: {e}")
            except Exception as e:
                print(f"❌ Lỗi khi sử dụng Gemini: {e}")
        
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
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            json_str = response_text[json_start:json_end]
                            analysis_data = json.loads(json_str)
                            analysis_data['engine'] = 'groq'
                            return analysis_data
                    except json.JSONDecodeError as e:
                        print(f"❌ Lỗi phân tích JSON từ phản hồi: {e}")
            except Exception as e:
                print(f"❌ Lỗi khi sử dụng Groq: {e}")
        
        # Không có kết quả phân tích từ API nào, trả về None
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
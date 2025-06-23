import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple
import json
import os
from dotenv import load_dotenv
from model_manager import init_chat_engine, chat_with_engine, analyze_prompt_concurrently
from datetime import datetime

class MaterialAIAnalyzer:
    def __init__(self, data_path: str):
        """
        Khởi tạo class phân tích vật liệu sử dụng AI
        Args:
            data_path: Đường dẫn đến file dữ liệu CSV
        """
        self.data = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Khởi tạo các mô hình
        self.init_models()
        
    def init_models(self):
        """Khởi tạo các mô hình AI"""
        # 1. Deep Learning Model cho dự đoán đa thuộc tính
        self.dl_model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)  # 3 thuộc tính: bandgap, conductivity, absorption
        ).to(self.device)
        
        # 2. Khởi tạo các LLM engines từ model_manager
        self.engines = []
        for engine_name in ["gemini", "groq"]:
            engine_tuple = init_chat_engine(engine_name)
            if engine_tuple[1] is not None or engine_name == "groq":  # Groq không cần chat object
                self.engines.append(engine_tuple)
        
    def analyze_with_llm(self, processed_data):
        """Phân tích vật liệu sử dụng LLMs"""
        # Tạo prompt cho phân tích
        prompt = f"""Analyze the following semiconductor material and provide insights:

Material Properties:
- Name: {processed_data['name']}
- Crystal Structure: {processed_data['crystal_structure']}
- Composition: {', '.join(processed_data['composition']['elements'])}
- Average Properties:
  * Atomic Mass: {processed_data['composition']['average_properties']['atomic_mass']:.2f}
  * Electronegativity: {processed_data['composition']['average_properties']['electronegativity']:.2f}

Physical Properties:
{self._format_properties(processed_data['properties'])}

Please provide:
1. Potential applications based on these properties
2. Advantages and limitations
3. Suggestions for property improvements
4. Similar materials for comparison

Format your response in a structured way with clear sections."""

        try:
            # Phân tích song song với nhiều mô hình
            results = analyze_prompt_concurrently(prompt, self.engines)
            
            # Cấu trúc kết quả
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'model_responses': {}
            }
            
            for result in results:
                engine_name = result.get('engine', 'unknown')
                response = result.get('response', 'No response')
                execution_time = result.get('execution_time', 0)
                
                analysis_results['model_responses'][engine_name] = {
                    'analysis': response,
                    'execution_time': execution_time
                }
            
            return analysis_results
            
        except Exception as e:
            print(f"❌ Lỗi trong quá trình phân tích LLM: {str(e)}")
            return {
                'error': f"LLM Analysis failed: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def _format_properties(self, properties):
        """Format thuộc tính vật lý cho prompt"""
        formatted = []
        
        if 'bandgap' in properties:
            formatted.append(f"- Bandgap: {properties['bandgap']['value']} {properties['bandgap']['unit']}")
            
        if 'conductivity' in properties:
            formatted.append(f"- Conductivity: {properties['conductivity']['value']} {properties['conductivity']['unit']}")
            
        if 'carrier_concentration' in properties:
            formatted.append(f"- Carrier Concentration: {properties['carrier_concentration']['value']} {properties['carrier_concentration']['unit']}")
            
        return "\n".join(formatted)
    
    def predict_properties(self, input_features: np.ndarray) -> Dict[str, Any]:
        """
        Dự đoán đa thuộc tính của vật liệu sử dụng Deep Learning
        Args:
            input_features: Đặc trưng đầu vào
        Returns:
            Dict chứa các thuộc tính dự đoán
        """
        # Chuẩn hóa đầu vào
        scaled_features = self.scaler.transform(input_features)
        
        # Chuyển sang tensor
        x = torch.FloatTensor(scaled_features).to(self.device)
        
        # Dự đoán
        self.dl_model.eval()
        with torch.no_grad():
            predictions = self.dl_model(x)
            
        # Chuyển về numpy
        predictions = predictions.cpu().numpy()
        
        return {
            'predicted_properties': {
                'bandgap': predictions[0],
                'conductivity': predictions[1],
                'absorption': predictions[2]
            },
            'model_confidence': self.calculate_confidence(predictions)
        }
    
    def analyze_crystal_structure(self, structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phân tích cấu trúc tinh thể sử dụng Graph Neural Network
        Args:
            structure_data: Thông tin về cấu trúc tinh thể
        Returns:
            Dict chứa kết quả phân tích cấu trúc
        """
        # Chuyển đổi dữ liệu cấu trúc thành graph
        graph_data = self.convert_to_graph(structure_data)
        
        # Dự đoán sử dụng GNN
        self.gnn_model.eval()
        with torch.no_grad():
            pred = self.gnn_model(graph_data)
            
        return {
            'structure_analysis': {
                'stability_score': float(pred[0]),
                'symmetry_features': pred[1].tolist(),
                'predicted_properties': pred[2].tolist()
            }
        }
    
    def calculate_confidence(self, predictions: np.ndarray) -> float:
        """Tính độ tin cậy của dự đoán"""
        # Implement phương pháp tính độ tin cậy phù hợp
        return float(np.mean(np.abs(predictions)))
    
    def convert_to_graph(self, structure_data: Dict[str, Any]) -> Data:
        """Chuyển đổi dữ liệu cấu trúc thành graph"""
        # Implement chuyển đổi dữ liệu cấu trúc thành graph
        # Trả về đối tượng torch_geometric.data.Data
        pass
    
    def save_results(self, results: Dict[str, Any], output_path: str = 'results/ai_analysis'):
        """Lưu kết quả phân tích"""
        os.makedirs(output_path, exist_ok=True)
        
        with open(f'{output_path}/ai_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)

class MaterialGNN(nn.Module):
    """Graph Neural Network cho phân tích cấu trúc tinh thể"""
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.fc = nn.Linear(16, 3)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = nn.functional.relu(x)
        x = self.conv3(x, edge_index)
        
        # Global pooling
        x = global_mean_pool(x, data.batch)
        
        # Fully connected layer
        x = self.fc(x)
        return x

class DataProcessor:
    """Xử lý và chuẩn hóa dữ liệu vật liệu"""
    def __init__(self):
        self.element_properties = {
            'Zn': {'atomic_number': 30, 'atomic_mass': 65.38, 'electronegativity': 1.65},
            'O': {'atomic_number': 8, 'atomic_mass': 16.00, 'electronegativity': 3.44},
            'Ga': {'atomic_number': 31, 'atomic_mass': 69.72, 'electronegativity': 1.81},
            'As': {'atomic_number': 33, 'atomic_mass': 74.92, 'electronegativity': 2.18},
            'Cu': {'atomic_number': 29, 'atomic_mass': 63.55, 'electronegativity': 1.90},
            'Si': {'atomic_number': 14, 'atomic_mass': 28.09, 'electronegativity': 1.90}
        }

    def process(self, raw_data):
        """Xử lý dữ liệu thô"""
        processed_data = {
            'name': self._standardize_name(raw_data.get('name', '')),
            'crystal_structure': self._process_structure(raw_data.get('structure', {})),
            'composition': self._analyze_composition(raw_data.get('composition', {})),
            'properties': self._normalize_properties(raw_data.get('properties', {}))
        }
        return processed_data

    def _standardize_name(self, name):
        """Chuẩn hóa tên vật liệu"""
        return name.strip().upper()

    def _process_structure(self, structure):
        """Xử lý thông tin cấu trúc tinh thể"""
        structure_types = {
            'cubic': 'Cubic',
            'hexagonal': 'Hexagonal',
            'wurtzite': 'Wurtzite',
            'zincblende': 'Zincblende',
            'tetragonal': 'Tetragonal',
            'orthorhombic': 'Orthorhombic'
        }
        return structure_types.get(structure.lower(), structure)

    def _analyze_composition(self, composition):
        """Phân tích thành phần hóa học"""
        result = {
            'elements': [],
            'stoichiometry': {},
            'average_properties': {
                'atomic_mass': 0,
                'electronegativity': 0
            }
        }
        
        total_atoms = sum(composition.values())
        
        for element, count in composition.items():
            if element in self.element_properties:
                result['elements'].append(element)
                result['stoichiometry'][element] = count / total_atoms
                
                # Tính toán thuộc tính trung bình
                fraction = count / total_atoms
                props = self.element_properties[element]
                result['average_properties']['atomic_mass'] += props['atomic_mass'] * fraction
                result['average_properties']['electronegativity'] += props['electronegativity'] * fraction
        
        return result

    def _normalize_properties(self, properties):
        """Chuẩn hóa các thuộc tính vật lý"""
        normalized = {}
        
        # Chuẩn hóa bandgap
        if 'bandgap' in properties:
            normalized['bandgap'] = {
                'value': float(properties['bandgap']),
                'unit': 'eV',
                'normalized': min(1.0, float(properties['bandgap']) / 10.0)  # Chuẩn hóa về thang 0-1
            }
        
        # Chuẩn hóa độ dẫn điện
        if 'conductivity' in properties:
            conductivity = float(properties['conductivity'])
            normalized['conductivity'] = {
                'value': conductivity,
                'unit': 'S/cm',
                'normalized': min(1.0, np.log10(conductivity + 1e-10) / 10.0)  # Chuẩn hóa logarit
            }
        
        # Chuẩn hóa nồng độ hạt tải
        if 'carrier_concentration' in properties:
            carrier_conc = float(properties['carrier_concentration'])
            normalized['carrier_concentration'] = {
                'value': carrier_conc,
                'unit': 'cm^-3',
                'normalized': min(1.0, np.log10(carrier_conc) / 20.0)  # Chuẩn hóa logarit
            }
        
        return normalized

class PropertyPredictor:
    """Dự đoán thuộc tính vật liệu"""
    def __init__(self):
        self.bandgap_model = self._init_bandgap_model()
        self.conductivity_model = self._init_conductivity_model()
        self.stability_model = self._init_stability_model()
        
    def _init_bandgap_model(self):
        """Khởi tạo mô hình dự đoán bandgap"""
        model = nn.Sequential(
            nn.Linear(14, 32),  # Thay đổi input size từ 11 thành 14
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        return model
        
    def _init_conductivity_model(self):
        """Khởi tạo mô hình dự đoán độ dẫn điện"""
        model = nn.Sequential(
            nn.Linear(14, 32),  # Thay đổi input size từ 11 thành 14
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        return model
        
    def _init_stability_model(self):
        """Khởi tạo mô hình dự đoán độ ổn định"""
        model = nn.Sequential(
            nn.Linear(14, 32),  # Thay đổi input size từ 11 thành 14
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        return model

    def predict(self, processed_data):
        """Dự đoán các thuộc tính của vật liệu"""
        # Chuẩn bị input features
        features = self._prepare_features(processed_data)
        
        # Thực hiện dự đoán
        with torch.no_grad():
            bandgap = self._predict_bandgap(features)
            conductivity = self._predict_conductivity(features)
            stability = self._predict_stability(features)
        
        return {
            'bandgap_prediction': {
                'value': float(bandgap),
                'unit': 'eV',
                'confidence': self._calculate_confidence(bandgap)
            },
            'conductivity_prediction': {
                'value': float(conductivity),
                'unit': 'S/cm',
                'confidence': self._calculate_confidence(conductivity)
            },
            'stability_prediction': {
                'value': float(stability),
                'normalized': float(stability),
                'confidence': self._calculate_confidence(stability)
            }
        }
    
    def _prepare_features(self, processed_data):
        """Chuẩn bị features cho mô hình"""
        features = []
        
        # Thêm thuộc tính thành phần
        comp = processed_data['composition']['average_properties']
        features.extend([
            comp['atomic_mass'],
            comp['electronegativity']
        ])
        
        # Thêm thuộc tính cấu trúc
        structure_encoding = self._encode_crystal_structure(processed_data['crystal_structure'])
        features.extend(structure_encoding)  # 6 features cho cấu trúc
        
        # Thêm thuộc tính vật lý đã biết
        props = processed_data['properties']
        if 'bandgap' in props:
            features.append(props['bandgap'].get('value', 0.0))
            features.append(props['bandgap'].get('normalized', 0.0))
        else:
            features.extend([0.0, 0.0])
            
        if 'conductivity' in props:
            features.append(props['conductivity'].get('value', 0.0))
            features.append(props['conductivity'].get('normalized', 0.0))
        else:
            features.extend([0.0, 0.0])
            
        if 'carrier_concentration' in props:
            features.append(props['carrier_concentration'].get('value', 0.0))
            features.append(props['carrier_concentration'].get('normalized', 0.0))
        else:
            features.extend([0.0, 0.0])
            
        return torch.tensor(features, dtype=torch.float32)
    
    def _encode_crystal_structure(self, structure):
        """Mã hóa cấu trúc tinh thể"""
        structure_types = ['Cubic', 'Hexagonal', 'Wurtzite', 'Zincblende', 'Tetragonal', 'Orthorhombic']
        encoding = [1.0 if s.lower() == structure.lower() else 0.0 for s in structure_types]
        return encoding
    
    def _predict_bandgap(self, features):
        """Dự đoán bandgap"""
        return self.bandgap_model(features)
    
    def _predict_conductivity(self, features):
        """Dự đoán độ dẫn điện"""
        return self.conductivity_model(features)
    
    def _predict_stability(self, features):
        """Dự đoán độ ổn định"""
        return self.stability_model(features)
    
    def _calculate_confidence(self, prediction):
        """Tính toán độ tin cậy của dự đoán"""
        # Đơn giản hóa: độ tin cậy dựa trên khoảng cách đến giá trị trung bình
        return min(1.0, max(0.0, 1.0 - abs(0.5 - float(prediction))))

class IntegratedMaterialAnalysis:
    def __init__(self):
        """Khởi tạo IntegratedMaterialAnalysis"""
        self.data_processor = DataProcessor()
        self.property_predictor = PropertyPredictor()
        self.result_manager = ResultManager()

    def analyze_new_material(self, material_data):
        """Phân tích vật liệu mới"""
        try:
            # Xử lý dữ liệu
            processed_data = self.data_processor.process(material_data)
            
            # Phân tích cơ bản
            basic_analysis = self.analyze_basic_properties(processed_data)
            
            # Dự đoán thuộc tính
            property_predictions = self.property_predictor.predict(processed_data)
            
            # Tổng hợp kết quả
            results = {
                'basic_analysis': basic_analysis,
                'predictions': property_predictions,
                'recommendations': self.generate_recommendations({
                    **basic_analysis,
                    **property_predictions
                })
            }
            
            return results
            
        except Exception as e:
            return {
                'error': f"Analysis failed: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_basic_properties(self, processed_data):
        """Phân tích các thuộc tính cơ bản"""
        basic_analysis = {
            'composition_analysis': self._analyze_composition(processed_data),
            'structure_analysis': self._analyze_structure(processed_data),
            'property_analysis': self._analyze_properties(processed_data)
        }
        return basic_analysis
    
    def _analyze_composition(self, processed_data):
        """Phân tích thành phần"""
        comp = processed_data['composition']
        return {
            'elements': comp['elements'],
            'stoichiometry': comp['stoichiometry'],
            'average_atomic_mass': comp['average_properties']['atomic_mass'],
            'average_electronegativity': comp['average_properties']['electronegativity']
        }
    
    def _analyze_structure(self, processed_data):
        """Phân tích cấu trúc"""
        return {
            'crystal_system': processed_data['crystal_structure'],
            'expected_symmetry': self._determine_symmetry(processed_data['crystal_structure']),
            'potential_defects': self._predict_defects(processed_data)
        }
    
    def _analyze_properties(self, processed_data):
        """Phân tích các thuộc tính vật lý"""
        props = processed_data['properties']
        analysis = {}
        
        if 'bandgap' in props:
            analysis['bandgap_category'] = self._categorize_bandgap(props['bandgap']['value'])
            
        if 'conductivity' in props:
            analysis['conductivity_category'] = self._categorize_conductivity(props['conductivity']['value'])
            
        if 'carrier_concentration' in props:
            analysis['carrier_type'] = self._determine_carrier_type(props['carrier_concentration']['value'])
        
        return analysis
    
    def _determine_symmetry(self, crystal_structure):
        """Xác định đối xứng từ cấu trúc tinh thể"""
        symmetry_map = {
            'Cubic': 'Oh',
            'Hexagonal': 'D6h',
            'Wurtzite': 'C6v',
            'Zincblende': 'Td',
            'Tetragonal': 'D4h',
            'Orthorhombic': 'D2h'
        }
        return symmetry_map.get(crystal_structure, 'Unknown')
    
    def _predict_defects(self, processed_data):
        """Dự đoán các khuyết tật có thể có"""
        defects = []
        structure = processed_data['crystal_structure'].lower()
        
        if 'cubic' in structure:
            defects.extend(['Vacancies', 'Interstitials', 'Stacking faults'])
        elif 'hexagonal' in structure:
            defects.extend(['Screw dislocations', 'Twin boundaries'])
        elif 'wurtzite' in structure:
            defects.extend(['Point defects', 'Stacking faults'])
            
        return defects
    
    def _categorize_bandgap(self, bandgap):
        """Phân loại vật liệu theo bandgap"""
        if bandgap < 0.1:
            return 'Metal/Semimetal'
        elif bandgap < 1.5:
            return 'Narrow bandgap semiconductor'
        elif bandgap < 3.0:
            return 'Medium bandgap semiconductor'
        else:
            return 'Wide bandgap semiconductor'
    
    def _categorize_conductivity(self, conductivity):
        """Phân loại vật liệu theo độ dẫn điện"""
        if conductivity > 1e4:
            return 'Metal'
        elif conductivity > 1e-8:
            return 'Semiconductor'
        else:
            return 'Insulator'
    
    def _determine_carrier_type(self, carrier_concentration):
        """Xác định loại hạt tải"""
        if carrier_concentration > 0:
            return 'n-type'
        elif carrier_concentration < 0:
            return 'p-type'
        else:
            return 'intrinsic'
    
    def generate_recommendations(self, results):
        """Tạo các đề xuất dựa trên kết quả phân tích"""
        recommendations = {
            'applications': self._suggest_applications(results),
            'improvements': self._suggest_improvements(results),
            'research_directions': self._suggest_research(results)
        }
        return recommendations
    
    def _suggest_applications(self, results):
        """Đề xuất ứng dụng tiềm năng"""
        apps = []
        props = results['basic_properties']['property_analysis']
        
        if 'bandgap_category' in props:
            if props['bandgap_category'] == 'Wide bandgap semiconductor':
                apps.extend(['Power electronics', 'UV devices', 'High temperature applications'])
            elif props['bandgap_category'] == 'Medium bandgap semiconductor':
                apps.extend(['Solar cells', 'LEDs', 'Transistors'])
            elif props['bandgap_category'] == 'Narrow bandgap semiconductor':
                apps.extend(['Infrared detectors', 'Thermoelectric devices'])
        
        return apps
    
    def _suggest_improvements(self, results):
        """Đề xuất cải tiến"""
        improvements = []
        basic_props = results['basic_properties']
        
        # Kiểm tra cấu trúc
        if 'potential_defects' in basic_props['structure_analysis']:
            improvements.append({
                'aspect': 'Crystal quality',
                'suggestion': 'Optimize growth conditions to reduce defects',
                'methods': ['Temperature control', 'Pressure regulation', 'Growth rate optimization']
            })
        
        # Kiểm tra độ dẫn
        if 'conductivity_category' in basic_props['property_analysis']:
            if basic_props['property_analysis']['conductivity_category'] == 'Semiconductor':
                improvements.append({
                    'aspect': 'Conductivity',
                    'suggestion': 'Enhance carrier mobility',
                    'methods': ['Doping optimization', 'Interface engineering', 'Strain engineering']
                })
        
        return improvements
    
    def _suggest_research(self, results):
        """Đề xuất hướng nghiên cứu"""
        research_directions = []
        predictions = results['predictions']
        
        # Dựa trên độ tin cậy của các dự đoán
        if predictions['stability_prediction']['confidence'] < 0.7:
            research_directions.append({
                'topic': 'Stability investigation',
                'methods': ['In-situ characterization', 'Aging tests', 'Environmental testing'],
                'priority': 'High'
            })
        
        # Thêm các hướng nghiên cứu tiên tiến
        research_directions.extend([
            {
                'topic': 'Nanostructuring',
                'methods': ['Quantum dots', 'Nanowires', '2D structures'],
                'priority': 'Medium'
            },
            {
                'topic': 'Interface engineering',
                'methods': ['Heterojunction design', 'Surface modification', 'Buffer layer optimization'],
                'priority': 'Medium'
            },
            {
                'topic': 'Advanced characterization',
                'methods': ['Synchrotron studies', 'In-situ TEM', 'Advanced spectroscopy'],
                'priority': 'Low'
            }
        ])
        
        return research_directions

class ResultManager:
    def __init__(self, db_path="results/material_analysis.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Khởi tạo cơ sở dữ liệu"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    def save_analysis(self, material_id, results):
        """Lưu kết quả phân tích"""
        timestamp = datetime.now().isoformat()
        
        # Chuẩn bị dữ liệu để lưu
        analysis_record = {
            'material_id': material_id,
            'timestamp': timestamp,
            'results': results
        }
        
        # Tạo tên file dựa trên material_id và timestamp
        filename = f"results/ai_analysis/{material_id}_{timestamp.replace(':', '-')}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Lưu kết quả dưới dạng JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_record, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Đã lưu kết quả phân tích vào {filename}")
    
    def load_analysis(self, material_id=None, timestamp=None):
        """Tải kết quả phân tích"""
        results = []
        
        # Lấy danh sách các file trong thư mục kết quả
        result_dir = "results/ai_analysis"
        if not os.path.exists(result_dir):
            return results
            
        for filename in os.listdir(result_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(result_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Lọc theo material_id nếu được chỉ định
                if material_id and data['material_id'] != material_id:
                    continue
                    
                # Lọc theo timestamp nếu được chỉ định
                if timestamp and data['timestamp'] != timestamp:
                    continue
                    
                results.append(data)
                    
            except Exception as e:
                print(f"⚠️ Lỗi khi đọc file {filename}: {str(e)}")
                
        return results
    
    def get_latest_analysis(self, material_id):
        """Lấy kết quả phân tích mới nhất cho một vật liệu"""
        results = self.load_analysis(material_id=material_id)
        if not results:
            return None
            
        # Sắp xếp theo timestamp và lấy kết quả mới nhất
        return sorted(results, key=lambda x: x['timestamp'], reverse=True)[0]
    
    def get_analysis_summary(self, material_id):
        """Tạo bản tóm tắt các phân tích cho một vật liệu"""
        results = self.load_analysis(material_id=material_id)
        if not results:
            return None
            
        summary = {
            'material_id': material_id,
            'total_analyses': len(results),
            'latest_analysis': self.get_latest_analysis(material_id),
            'analysis_dates': [r['timestamp'] for r in results],
            'property_trends': self._analyze_trends(results)
        }
        
        return summary
    
    def _analyze_trends(self, results):
        """Phân tích xu hướng trong các kết quả phân tích"""
        trends = {
            'bandgap': [],
            'conductivity': [],
            'stability': []
        }
        
        for result in results:
            predictions = result['results'].get('predictions', {})
            
            if 'bandgap_prediction' in predictions:
                trends['bandgap'].append({
                    'timestamp': result['timestamp'],
                    'value': predictions['bandgap_prediction']['value'],
                    'confidence': predictions['bandgap_prediction']['confidence']
                })
                
            if 'conductivity_prediction' in predictions:
                trends['conductivity'].append({
                    'timestamp': result['timestamp'],
                    'value': predictions['conductivity_prediction']['value'],
                    'confidence': predictions['conductivity_prediction']['confidence']
                })
                
            if 'stability_prediction' in predictions:
                trends['stability'].append({
                    'timestamp': result['timestamp'],
                    'value': predictions['stability_prediction']['value'],
                    'confidence': predictions['stability_prediction']['confidence']
                })
        
        return trends

def main():
    print("🔬 Khởi tạo phân tích vật liệu bán dẫn sử dụng AI...")
    
    # Khởi tạo hệ thống phân tích
    analyzer = IntegratedMaterialAnalysis()
    
    # Dữ liệu vật liệu mới
    new_material = {
        'name': 'GaAs',
        'crystal_structure': 'Zincblende',
        'composition': {'Ga': 1, 'As': 1},
        'properties': {
            'bandgap': 1.42,
            'conductivity': 1000,
            'carrier_concentration': 1e18
        }
    }
    
    # Thực hiện phân tích
    results = analyzer.analyze_new_material(new_material)
    
    # In kết quả
    print("\n=== Kết quả Phân tích ===")
    print("\n1. Thuộc tính cơ bản:")
    print(results['basic_properties'])
    
    print("\n2. Dự đoán:")
    print(results['predictions'])
    
    print("\n3. Đề xuất:")
    print(results['recommendations'])

if __name__ == "__main__":
    main() 
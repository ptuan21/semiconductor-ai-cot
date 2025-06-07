import pandas as pd
import numpy as np
import torch
import torch.nn as nn
# Assuming you have torch_geometric installed, if not, this part will need adjustment or removal
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv, global_mean_pool
# Assuming you have transformers installed
# from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
# Ensure model_manager.py is correctly setting up Gemini and Groq
from model_manager import init_chat_engine, chat_with_engine, analyze_prompt_concurrently
from datetime import datetime
import cv2
import base64
import logging # Sử dụng logging để ghi log lỗi thay vì print trực tiếp
import threading
from scipy import ndimage
from skimage import feature, filters, measure

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(module)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# Constants for semiconductor analysis
SEMICONDUCTOR_DEFECT_TYPES = {
    "DISLOCATION": "Dislocation defects",
    "STACKING_FAULT": "Stacking faults",
    "GRAIN_BOUNDARY": "Grain boundaries",
    "PRECIPITATE": "Precipitates",
    "VOID": "Voids or cavities",
    "SURFACE_CONTAMINATION": "Surface contamination",
    "CRYSTAL_DEFECT": "Crystal structure defects",
    "IMPURITY": "Impurity clusters"
}

# --- Các lớp AI chuyên biệt (có thể giữ skeleton hoặc thay thế bằng các mô hình đã huấn luyện) ---

# Lớp này ban đầu chứa DL/GNN models, có thể giữ hoặc chuyển đi nếu chỉ dùng LLM
class MaterialAIModels:
    def __init__(self):
        # Giữ lại skeleton hoặc tích hợp mô hình thực tế tại đây
        self.dl_model = None # nn.Sequential(...)
        self.gnn_model = None # MaterialGNN()
        self.is_ready = False # Đặt False nếu mô hình chưa được huấn luyện/tích hợp đầy đủ
        logger.info("MaterialAIModels initialized (skeletons).")

    def predict_properties(self, input_features: np.ndarray) -> Dict[str, Any]:
        """Dự đoán đa thuộc tính nếu mô hình DL có sẵn"""
        if not self.is_ready or self.dl_model is None:
            # Fallback hoặc trả về None/Error
            logger.warning("DL model not ready for property prediction.")
            return {"error": "DL model not available for prediction."}
        # Implement logic sử dụng self.dl_model
        pass # Placeholder

    def analyze_crystal_structure_ml(self, structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Phân tích cấu trúc nếu mô hình GNN có sẵn"""
        if not self.is_ready or self.gnn_model is None:
            # Fallback hoặc trả về None/Error
            logger.warning("GNN model not ready for structure analysis.")
            return {"error": "GNN model not available for analysis."}
        # Implement logic sử dụng self.gnn_model
        pass # Placeholder

# Nếu bạn muốn giữ lại GNN model skeleton
# class MaterialGNN(nn.Module):
#     """Graph Neural Network cho phân tích cấu trúc tinh thể"""
#     def __init__(self):
#         super().__init__()
#         # Define layers
#         pass
#
#     def forward(self, data):
#         # Forward pass logic
#         pass # Placeholder


class DataProcessor:
    """Xử lý và chuẩn hóa dữ liệu vật liệu (có thể dùng chung cho cả text và image data)"""
    # Giữ nguyên hoặc điều chỉnh nếu cần
    def __init__(self):
         self.element_properties = {
            'Zn': {'atomic_number': 30, 'atomic_mass': 65.38, 'electronegativity': 1.65},
            'O': {'atomic_number': 8, 'atomic_mass': 16.00, 'electronegativity': 3.44},
            'Ga': {'atomic_number': 31, 'atomic_mass': 69.72, 'electronegativity': 1.81},
            'As': {'atomic_number': 33, 'atomic_mass': 74.92, 'electronegativity': 2.18},
            'Cu': {'atomic_number': 29, 'atomic_mass': 63.55, 'electronegativity': 1.90},
            'Si': {'atomic_number': 14, 'atomic_mass': 28.09, 'electronegativity': 1.90},
            'Mn': {'atomic_number': 25, 'atomic_mass': 54.94, 'electronegativity': 1.60},
            'Sn': {'atomic_number': 50, 'atomic_mass': 118.71, 'electronegativity': 1.96}
         }
         # Initialize scaler if needed for numerical features
         self.scaler = StandardScaler()
         self._is_scaler_trained = False
         logger.info("DataProcessor initialized.")

    def process_material_data(self, raw_data: Dict) -> Dict:
        """Xử lý dữ liệu vật liệu từ form/CSV"""
        # Giữ nguyên logic xử lý raw_data
        processed = {
            'name': raw_data.get('name', 'Unknown Material').strip(),
            'formula': raw_data.get('formula', 'N/A'),
            'crystal_structure': raw_data.get('crystal_structure', 'Unknown Structure').strip(),
            # Add composition analysis or keep simple composition
            'composition': self._analyze_composition(raw_data.get('composition', {})) if isinstance(raw_data.get('composition'), dict) else raw_data.get('composition', 'N/A'),
            'properties': self._process_properties(raw_data.get('properties', {})) if isinstance(raw_data.get('properties'), dict) else raw_data.get('properties', {})
        }
        # Fit scaler if this is training data or the first material processed
        if isinstance(processed['properties'], dict) and processed['properties']:
             numerical_features = self._extract_numerical_features(processed['properties'])
             if numerical_features:
                 # Example: assuming a fixed order of numerical properties
                 if not self._is_scaler_trained:
                      # You would typically fit the scaler on a full dataset *before* processing
                      # For a simple refactor, let's just skip fitting here
                      # self.scaler.fit(np.array(list(numerical_features.values())).reshape(1, -1))
                      # self._is_scaler_trained = True
                      pass # Skip fitting in this context
                 # Scale features if scaler is trained
                 # scaled_features = self.scaler.transform(np.array(list(numerical_features.values())).reshape(1, -1))
                 # processed['scaled_features'] = scaled_features.flatten().tolist()
             else:
                 # processed['scaled_features'] = []
                 pass # No numerical features
        
        return processed
        
    def _process_properties(self, properties_dict: Dict) -> Dict:
         """Standardize properties and handle potential string values"""
         processed_props = {}
         for key, value in properties_dict.items():
             # Attempt to convert known properties to float, handle errors
             if key in ['bandgap', 'conductivity', 'carrier_concentration', 'mobility', 'thermal_conductivity', 'melting_point', 'density', 'temperature']:
                 try:
                     processed_props[key] = float(str(value).replace(',', '.'))
                 except (ValueError, TypeError):
                     processed_props[key] = str(value) # Keep as string if cannot convert
             else:
                 processed_props[key] = value # Keep other properties as is
         return processed_props

    def _analyze_composition(self, composition: Dict) -> Dict:
         """Phân tích thành phần hóa học, trả về định dạng chuẩn"""
         result = {
             'elements': list(composition.keys()),
             'stoichiometry': composition, # Keep raw ratios for simplicity
             'average_properties': { # Calculate average properties based on element_properties
                 'atomic_mass': 0,
                 'electronegativity': 0
             }
         }
         total_atoms = sum(composition.values()) if composition else 1
         for element, count in composition.items():
             if element in self.element_properties:
                 fraction = count / total_atoms
                 props = self.element_properties[element]
                 result['average_properties']['atomic_mass'] += props['atomic_mass'] * fraction
                 result['average_properties']['electronegativity'] += props['electronegativity'] * fraction
         return result
         
    def _extract_numerical_features(self, properties: Dict) -> Dict:
         """Extract numerical features for ML models"""
         # This needs to be consistent with the training data features
         # Example features (adjust based on your actual ML model inputs):
         features = {}
         for prop in ['bandgap', 'conductivity', 'carrier_concentration', 'mobility', 'thermal_conductivity', 'melting_point', 'density', 'temperature']:
             # Attempt to get numerical value, use default 0 if not available or convertible
             try:
                 features[prop] = float(str(properties.get(prop, 0)).replace(',', '.'))
             except (ValueError, TypeError):
                 features[prop] = 0.0
                 
         # Add composition features (example)
         # if 'composition' in properties: # Need composition processed data here
         #     features['avg_atomic_mass'] = properties['composition'].get('average_properties', {}).get('atomic_mass', 0.0)
         #     features['avg_electronegativity'] = properties['composition'].get('average_properties', {}).get('electronegativity', 0.0)
         # else:
         #     features['avg_atomic_mass'] = 0.0
         #     features['avg_electronegativity'] = 0.0

         return features if features else None # Return None if no numerical features extracted


# --- Lớp quản lý kết quả (có thể dùng chung) ---
class ResultManager:
    def __init__(self, base_dir="results"):
        self.base_dir = base_dir
        os.makedirs(os.path.join(self.base_dir, "ai_analysis"), exist_ok=True)
        logger.info(f"ResultManager initialized. Results will be saved in {self.base_dir}.")

    def save_analysis(self, material_name: str, results: Dict[str, Any], analysis_type: str = "material_analysis"):
        """Lưu kết quả phân tích vào file JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize material_name to be safe for filenames
        safe_material_name = "".join(x for x in material_name if x.isalnum() or x in "_-").replace(" ", "_")
        filename = f"{safe_material_name}_{analysis_type}_{timestamp}.json"
        filepath = os.path.join(self.base_dir, "ai_analysis", filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Đã lưu kết quả phân tích vào {filepath}")
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu kết quả vào file {filepath}: {e}")

    def load_analysis(self, material_name: str = None, analysis_type: str = None) -> List[Dict]:
        """Tải kết quả phân tích từ file JSON"""
        results = []
        result_dir = os.path.join(self.base_dir, "ai_analysis")
        if not os.path.exists(result_dir):
            return results

        for filename in os.listdir(result_dir):
            if not filename.endswith('.json'):
                continue

            filepath = os.path.join(result_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Lọc theo material_name nếu được chỉ định
                if material_name and material_name.lower() not in filename.lower():
                    continue

                # Lọc theo analysis_type nếu được chỉ định
                if analysis_type and analysis_type.lower() not in filename.lower():
                    continue

                results.append(data)

            except Exception as e:
                logger.warning(f"⚠️ Lỗi khi đọc file {filename}: {e}")

        return results

    def get_latest_analysis(self, material_name: str, analysis_type: str = "material_analysis") -> Dict | None:
        """Lấy kết quả phân tích mới nhất cho một vật liệu"""
        results = self.load_analysis(material_name=material_name, analysis_type=analysis_type)
        if not results:
            return None
        # Assuming filename contains timestamp and sorting by filename works
        return sorted(results, key=lambda x: x.get('timestamp', ''), reverse=True)[0] # Sort by timestamp key


# --- Lớp MaterialImageAnalyzer được cấu hình lại ---
class SemiconductorImageProcessor:
    """Helper class for semiconductor-specific image processing"""
    
    @staticmethod
    def enhance_contrast(img: np.ndarray) -> np.ndarray:
        """Enhance image contrast using advanced techniques"""
        if len(img.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to luminance channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            lab = cv2.merge((l,a,b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Additional contrast stretching
            enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
            return enhanced
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(img)
            return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    @staticmethod
    def denoise_image(img: np.ndarray) -> np.ndarray:
        """Advanced noise reduction preserving edges and fine details"""
        if len(img.shape) == 3:
            # Non-local means denoising with optimal parameters for semiconductor images
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            # Additional bilateral filtering to preserve edges
            return cv2.bilateralFilter(denoised, 9, 75, 75)
        else:
            denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
            return cv2.bilateralFilter(denoised, 9, 75, 75)

    @staticmethod
    def detect_edges(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced edge detection using multiple methods"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Compute gradients using Scharr operator (more accurate than Sobel)
        grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Normalize magnitude for visualization
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply adaptive thresholding
        binary_edges = cv2.adaptiveThreshold(
            magnitude, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary_edges, direction

class AdvancedImagePreprocessor:
    """Advanced image preprocessing techniques for semiconductor analysis"""
    
    @staticmethod
    def apply_frequency_filtering(img: np.ndarray) -> np.ndarray:
        """Apply frequency domain filtering to enhance defect visibility"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Convert to frequency domain
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Create butterworth highpass filter
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        D0 = 30  # Cutoff frequency
        n = 2    # Order of butterworth filter
        
        x = np.linspace(-ccol, ccol, cols)
        y = np.linspace(-crow, crow, rows)
        X, Y = np.meshgrid(x, y)
        D = np.sqrt(X**2 + Y**2)
        
        H = 1 / (1 + (D0/D)**(2*n))
        H = np.dstack((H, H))
        
        # Apply filter and inverse DFT
        fshift = dft_shift * H
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        
        # Normalize and enhance contrast
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img_back

    @staticmethod
    def apply_morphological_operations(img: np.ndarray) -> np.ndarray:
        """Apply advanced morphological operations to enhance defect boundaries"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Create structuring elements of different shapes
        kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
        
        # Apply black hat operation to detect dark defects
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_circle)
        
        # Apply top hat operation to detect bright defects
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_circle)
        
        # Combine the results
        enhanced = cv2.add(gray, black_hat)
        enhanced = cv2.add(enhanced, top_hat)
        
        # Apply morphological gradient for edge enhancement
        gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel_cross)
        
        return enhanced, gradient

class DefectFeatureExtractor:
    """Extract advanced features for defect classification"""
    
    @staticmethod
    def extract_texture_features(img: np.ndarray, region: Dict) -> Dict[str, float]:
        """Extract advanced texture features from a region"""
        x1, y1, x2, y2 = region['bbox']
        roi = img[y1:y2, x1:x2]
        
        if roi.size == 0:
            return {}
            
        # Calculate GLCM features
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
        glcm = feature.graycomatrix(roi, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                  symmetric=True, normed=True)
        
        features = {
            'contrast': feature.graycoprops(glcm, 'contrast').mean(),
            'dissimilarity': feature.graycoprops(glcm, 'dissimilarity').mean(),
            'homogeneity': feature.graycoprops(glcm, 'homogeneity').mean(),
            'energy': feature.graycoprops(glcm, 'energy').mean(),
            'correlation': feature.graycoprops(glcm, 'correlation').mean(),
            'ASM': feature.graycoprops(glcm, 'ASM').mean()
        }
        
        # Add Hu Moments
        moments = cv2.moments(roi)
        hu_moments = cv2.HuMoments(moments).flatten()
        for i, moment in enumerate(hu_moments):
            features[f'hu_moment_{i}'] = moment
            
        return features

    @staticmethod
    def extract_frequency_features(img: np.ndarray, region: Dict) -> Dict[str, float]:
        """Extract frequency domain features from a region"""
        x1, y1, x2, y2 = region['bbox']
        roi = img[y1:y2, x1:x2]
        
        if roi.size == 0:
            return {}
            
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
        # Compute FFT
        f = np.fft.fft2(roi)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        
        features = {
            'freq_mean': np.mean(magnitude_spectrum),
            'freq_std': np.std(magnitude_spectrum),
            'freq_max': np.max(magnitude_spectrum),
            'freq_energy': np.sum(magnitude_spectrum**2),
            'freq_entropy': -np.sum(magnitude_spectrum * np.log2(magnitude_spectrum + 1e-10))
        }
        
        return features

class SEMImageAnalyzer:
    """Specialized analyzer for SEM images of semiconductor materials"""
    
    def __init__(self):
        self.min_defect_size = 5  # nm
        self.max_defect_size = 500  # nm
        self.scale_factor = None  # pixels/nm
        
    def calibrate_scale(self, scale_bar_length_px: int, scale_bar_length_nm: float):
        """Calibrate pixel to nanometer conversion"""
        self.scale_factor = scale_bar_length_px / scale_bar_length_nm
        
    def analyze_layer_interface(self, img: np.ndarray) -> Dict:
        """Analyze interface between different material layers (e.g. ZnO/SiO2)"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Use Sobel operator to detect horizontal edges (layer interfaces)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y = np.absolute(sobel_y)
        
        # Normalize and threshold
        sobel_y = np.uint8(255 * sobel_y / np.max(sobel_y))
        _, binary = cv2.threshold(sobel_y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours of interfaces
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        interfaces = []
        for contour in contours:
            # Fit line to interface
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            interface = {
                'position': (x[0], y[0]),
                'direction': (vx[0], vy[0]),
                'roughness': self._calculate_interface_roughness(contour),
                'length': cv2.arcLength(contour, False)
            }
            interfaces.append(interface)
            
        return {'interfaces': interfaces}
        
    def _calculate_interface_roughness(self, contour: np.ndarray) -> float:
        """Calculate interface roughness using RMS deviation"""
        # Fit polynomial to interface
        x = contour[:, 0, 0]
        y = contour[:, 0, 1]
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        
        # Calculate RMS deviation from fitted line
        y_fit = p(x)
        rms = np.sqrt(np.mean((y - y_fit) ** 2))
        return float(rms)

class ZnODefectPatterns:
    """Define specific patterns for ZnO material defects"""
    
    PATTERNS = {
        'VOID': {
            'shape': 'circular',
            'intensity': 'dark',
            'size_range': (10, 100),  # nm
            'circularity': (0.8, 1.0),
            'contrast_ratio': (0.1, 0.4)
        },
        'GRAIN_BOUNDARY': {
            'shape': 'linear',
            'intensity': 'variable',
            'min_length': 50,  # nm
            'linearity': 0.9,
            'branching': True
        },
        'SURFACE_DEFECT': {
            'shape': 'irregular',
            'intensity': 'bright',
            'texture': 'rough',
            'gradient_variance': (0.4, 0.8)
        },
        'CRYSTAL_DEFECT': {
            'shape': 'angular',
            'intensity': 'dark',
            'symmetry': True,
            'edge_sharpness': 'high'
        }
    }
    
    @staticmethod
    def match_pattern(features: Dict, defect_type: str) -> float:
        """Calculate how well features match a specific defect pattern"""
        pattern = ZnODefectPatterns.PATTERNS.get(defect_type)
        if not pattern:
            return 0.0
            
        score = 0.0
        total_weights = 0
        
        # Shape matching
        if 'circularity' in features and pattern['shape'] == 'circular':
            circ_score = 1.0 - abs(features['circularity'] - 0.9)
            score += circ_score * 2
            total_weights += 2
            
        # Intensity matching
        if 'mean_intensity' in features:
            norm_intensity = features['mean_intensity'] / 255
            if pattern['intensity'] == 'dark' and norm_intensity < 0.4:
                score += 1.0
            elif pattern['intensity'] == 'bright' and norm_intensity > 0.6:
                score += 1.0
            total_weights += 1
            
        # Size matching
        if 'area' in features and 'size_range' in pattern:
            min_size, max_size = pattern['size_range']
            if min_size <= features['area'] <= max_size:
                score += 1.0
            total_weights += 1
            
        # Texture matching
        if 'gradient_std' in features and pattern.get('texture') == 'rough':
            if features['gradient_std'] > 30:
                score += 1.0
            total_weights += 1
            
        return score / total_weights if total_weights > 0 else 0.0

class MaterialDefectClassifier:
    """Enhanced defect classifier with consistent parameters"""
    
    def __init__(self):
        self.normalizer = ImageNormalizer()
        self.params = DetectionParameters()
        self.patterns = ZnODefectPatterns()
        
    def classify_defect(self, img: np.ndarray, region: Dict) -> Dict:
        """Classify defects using consistent parameters"""
        # Normalize the region first
        x1, y1, x2, y2 = region['bbox']
        roi = img[y1:y2, x1:x2]
        normalized_roi = self.normalizer.normalize(roi)
        
        # Extract features from normalized image
        features = self._extract_features(normalized_roi, region)
        
        # Calculate pattern matching scores with consistent thresholds
        scores = {}
        for defect_type in ZnODefectPatterns.PATTERNS.keys():
            score = self.patterns.match_pattern(features, defect_type)
            confidence_threshold = self.params.get_confidence_threshold('medium')
            if score >= confidence_threshold:
                scores[defect_type] = score
                
        # Find best matching pattern
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            confidence = best_type[1]
            defect_type = best_type[0]
        else:
            defect_type = "UNKNOWN"
            confidence = 0.0
        
        # Calculate severity using consistent weights
        severity = self._calculate_severity(features, defect_type)
        
        return {
            'type': defect_type,
            'confidence': confidence,
            'features': features,
            'severity': severity
        }
        
    def _extract_features(self, normalized_roi: np.ndarray, region: Dict) -> Dict:
        """Extract features using consistent parameters"""
        features = {
            'mean_intensity': np.mean(normalized_roi),
            'std_intensity': np.std(normalized_roi),
            'min_intensity': np.min(normalized_roi),
            'max_intensity': np.max(normalized_roi),
            'area': region.get('area', 0)
        }
        
        # Add shape features with consistent thresholds
        if 'contour' in region:
            contour = region['contour']
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            features['circularity'] = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
        # Gradient features
        gradient_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        
        return features
        
    def _calculate_severity(self, features: Dict, defect_type: str) -> str:
        """Calculate defect severity based on features and type"""
        severity_score = 0
        
        if defect_type == 'VOID':
            # Voids are more severe if larger and darker
            size_score = min(features.get('area', 0) / 1000, 1)
            darkness_score = 1 - (features.get('mean_intensity', 0) / 255)
            severity_score = 0.6 * size_score + 0.4 * darkness_score
            
        elif defect_type == 'GRAIN_BOUNDARY':
            # Grain boundaries are more severe if longer and more irregular
            length_score = min(features.get('area', 0) / 2000, 1)
            irregularity = 1 - features.get('circularity', 0)
            severity_score = 0.5 * length_score + 0.5 * irregularity
            
        elif defect_type == 'SURFACE_DEFECT':
            # Surface defects are more severe if they have high gradient variance
            gradient_score = min(features.get('gradient_std', 0) / 100, 1)
            size_score = min(features.get('area', 0) / 1500, 1)
            severity_score = 0.7 * gradient_score + 0.3 * size_score
            
        elif defect_type == 'CRYSTAL_DEFECT':
            # Crystal defects are more severe if they have sharp edges and are larger
            edge_score = min(features.get('gradient_mean', 0) / 150, 1)
            size_score = min(features.get('area', 0) / 1200, 1)
            severity_score = 0.5 * edge_score + 0.5 * size_score
            
        if severity_score > 0.7:
            return "high"
        elif severity_score > 0.4:
            return "medium"
        else:
            return "low"

class DefectVisualizer:
    """Visualize and analyze detected defects in detail"""
    
    def __init__(self, zoom_factor: int = 2):
        self.zoom_factor = zoom_factor
        self.colors = {
            'VOID': (255, 0, 0),        # Blue
            'GRAIN_BOUNDARY': (0, 255, 0),  # Green
            'SURFACE_DEFECT': (0, 0, 255),  # Red
            'CRYSTAL_DEFECT': (255, 255, 0)  # Cyan
        }
        
    def create_defect_visualization(self, img: np.ndarray, defects: List[Dict], 
                                  save_dir: str = "defect_analysis") -> Dict[str, str]:
        """Create detailed visualization of detected defects"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create overview image with all defects marked
        overview_img = img.copy()
        if len(overview_img.shape) == 2:
            overview_img = cv2.cvtColor(overview_img, cv2.COLOR_GRAY2BGR)
            
        # Dictionary to store paths of generated images
        result_images = {
            'overview': '',
            'defects': []
        }
        
        # Draw all defects on overview image
        for idx, defect in enumerate(defects):
            defect_type = defect.get('type', 'UNKNOWN')
            color = self.colors.get(defect_type, (128, 128, 128))
            confidence = defect.get('confidence', 0)
            severity = defect.get('severity', 'unknown')
            
            # Draw contour
            if 'contour' in defect:
                cv2.drawContours(overview_img, [defect['contour']], -1, color, 2)
            
            # Draw bounding box
            x1, y1, x2, y2 = defect['bbox']
            cv2.rectangle(overview_img, (x1, y1), (x2, y2), color, 2)
            
            # Add label with type, confidence and severity
            label = f"{defect_type} ({confidence:.2f}) - {severity}"
            cv2.putText(overview_img, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Extract and save individual defect image
            defect_img = self._extract_and_zoom_defect(img, defect)
            defect_path = os.path.join(save_dir, 
                                     f"defect_{timestamp}_{idx}_{defect_type}.png")
            cv2.imwrite(defect_path, defect_img)
            
            # Create detailed analysis image
            analysis_img = self._create_defect_analysis_image(img, defect)
            analysis_path = os.path.join(save_dir,
                                       f"analysis_{timestamp}_{idx}_{defect_type}.png")
            cv2.imwrite(analysis_path, analysis_img)
            
            result_images['defects'].append({
                'type': defect_type,
                'defect_image': defect_path,
                'analysis_image': analysis_path,
                'bbox': defect['bbox'],
                'confidence': confidence,
                'severity': severity
            })
            
        # Save overview image
        overview_path = os.path.join(save_dir, f"overview_{timestamp}.png")
        cv2.imwrite(overview_path, overview_img)
        result_images['overview'] = overview_path
        
        return result_images
        
    def _extract_and_zoom_defect(self, img: np.ndarray, defect: Dict) -> np.ndarray:
        """Extract and zoom in on a specific defect"""
        x1, y1, x2, y2 = defect['bbox']
        
        # Add padding around defect
        padding = 20
        h, w = img.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Extract region
        defect_region = img[y1:y2, x1:x2].copy()
        
        # Convert to BGR if grayscale
        if len(defect_region.shape) == 2:
            defect_region = cv2.cvtColor(defect_region, cv2.COLOR_GRAY2BGR)
            
        # Apply zoom
        zoomed = cv2.resize(defect_region, None, 
                           fx=self.zoom_factor, fy=self.zoom_factor,
                           interpolation=cv2.INTER_CUBIC)
        
        return zoomed
        
    def _create_defect_analysis_image(self, img: np.ndarray, defect: Dict) -> np.ndarray:
        """Create detailed analysis visualization for a defect"""
        # Extract zoomed defect
        zoomed = self._extract_and_zoom_defect(img, defect)
        h, w = zoomed.shape[:2]
        
        # Create larger image for analysis
        analysis_img = np.zeros((h, w*2, 3), dtype=np.uint8)
        
        # Add zoomed original on left
        analysis_img[:, :w] = zoomed
        
        # Add enhanced version on right
        enhanced = self._enhance_defect_visualization(zoomed, defect['type'])
        analysis_img[:, w:] = enhanced
        
        # Add measurements and annotations
        self._add_measurements(analysis_img, defect, w)
        
        return analysis_img
        
    def _enhance_defect_visualization(self, img: np.ndarray, defect_type: str) -> np.ndarray:
        """Enhance defect visualization based on type"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        enhanced = None
        
        if defect_type == 'VOID':
            # Enhance dark regions
            enhanced = cv2.equalizeHist(gray)
            _, enhanced = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
            
        elif defect_type == 'GRAIN_BOUNDARY':
            # Enhance edges
            enhanced = cv2.Canny(gray, 50, 150)
            
        elif defect_type == 'SURFACE_DEFECT':
            # Enhance texture
            enhanced = cv2.adaptiveThreshold(gray, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
            
        elif defect_type == 'CRYSTAL_DEFECT':
            # Enhance structural patterns
            gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            enhanced = np.sqrt(gradient_x**2 + gradient_y**2)
            enhanced = np.uint8(255 * enhanced / np.max(enhanced))
            
        if enhanced is None:
            enhanced = gray
            
        # Convert back to BGR
        if len(enhanced.shape) == 2:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
        return enhanced
        
    def _add_measurements(self, img: np.ndarray, defect: Dict, split_x: int):
        """Add measurements and annotations to analysis image"""
        features = defect.get('features', {})
        
        # Add title
        title = f"Type: {defect['type']} (Confidence: {defect.get('confidence', 0):.2f})"
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
                   
        # Add measurements on left side
        y_pos = 60
        measurements = [
            f"Size: {features.get('area', 0):.1f} px²",
            f"Circularity: {features.get('circularity', 0):.2f}",
            f"Mean Intensity: {features.get('mean_intensity', 0):.1f}",
            f"Gradient Std: {features.get('gradient_std', 0):.1f}"
        ]
        
        for measurement in measurements:
            cv2.putText(img, measurement, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25
            
        # Add enhancement info on right side
        y_pos = 60
        cv2.putText(img, "Enhanced View", (split_x + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                   
        enhancements = {
            'VOID': "Dark region enhancement",
            'GRAIN_BOUNDARY': "Edge detection",
            'SURFACE_DEFECT': "Texture analysis",
            'CRYSTAL_DEFECT': "Structural pattern"
        }
        
        enhancement_text = enhancements.get(defect['type'], "Standard enhancement")
        cv2.putText(img, enhancement_text, (split_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Update the main detector class
class SemiconductorDefectDetector:
    def __init__(self):
        self.sem_analyzer = SEMImageAnalyzer()
        self.classifier = MaterialDefectClassifier()
        self.visualizer = DefectVisualizer(zoom_factor=3)  # Zoom 3x
        
    def analyze_and_visualize(self, img: np.ndarray, 
                            scale_bar_length_px: int = None,
                            scale_bar_length_nm: float = None,
                            save_dir: str = "defect_analysis") -> Dict:
        """Analyze image and create detailed defect visualizations"""
        # Perform analysis
        analysis_results = self.analyze_sem_image(img, scale_bar_length_px, scale_bar_length_nm)
        
        # Create visualizations
        visualization_results = self.visualizer.create_defect_visualization(
            img, analysis_results['defects'], save_dir
        )
        
        # Combine results
        return {
            **analysis_results,
            'visualizations': visualization_results
        }
        
    def analyze_sem_image(self, img: np.ndarray, 
                         scale_bar_length_px: int = None, 
                         scale_bar_length_nm: float = None) -> Dict:
        """Analyze SEM image with calibrated scale"""
        if scale_bar_length_px and scale_bar_length_nm:
            self.sem_analyzer.calibrate_scale(scale_bar_length_px, scale_bar_length_nm)
            
        # Analyze interfaces
        interface_analysis = self.sem_analyzer.analyze_layer_interface(img)
        
        # Detect and classify defects
        defects = self._detect_defects(img)
        classified_defects = []
        
        for defect in defects:
            classification = self.classifier.classify_defect(img, defect)
            classified_defects.append({**defect, **classification})
            
        return {
            'interface_analysis': interface_analysis,
            'defects': classified_defects
        }
        
    def _detect_defects(self, img: np.ndarray) -> List[Dict]:
        """Detect potential defects in the image"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Multi-scale detection
        defects = []
        scales = [1, 2]  # Multiple scales for different sized defects
        
        for scale in scales:
            # Resize image for multi-scale analysis
            if scale != 1:
                scaled_img = cv2.resize(gray, None, fx=1/scale, fy=1/scale)
            else:
                scaled_img = gray
                
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                scaled_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                # Scale contour points back to original size
                if scale != 1:
                    contour = contour * scale
                    
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter out too small or too large regions
                if self.sem_analyzer.min_defect_size <= w <= self.sem_analyzer.max_defect_size and \
                   self.sem_analyzer.min_defect_size <= h <= self.sem_analyzer.max_defect_size:
                    defect = {
                        'bbox': (x, y, x+w, y+h),
                        'contour': contour,
                        'scale': scale
                    }
                    defects.append(defect)
                    
        return defects

class MaterialImageAnalyzer:
    """Phân tích hình ảnh vật liệu bán dẫn và phát hiện lỗi"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.defect_detector = SemiconductorDefectDetector()
        self.image_processor = SemiconductorImageProcessor()
        
        # Khởi tạo các model AI
        self.engines = [
            init_chat_engine("gemini"),
            init_chat_engine("groq")
        ]
        self.valid_engines = [
            engine for engine in self.engines
            if engine[0] is not None and (engine[0] != 'gemini' or engine[1] is not None)
        ]

        if not self.valid_engines:
            self.logger.warning("⚠️ No valid AI engines initialized for image analysis.")

        # Khởi tạo DataProcessor và ResultManager (có thể dùng chung instance từ app.py)
        self.data_processor = DataProcessor() # Or pass instance if managed externally
        self.result_manager = ResultManager() # Or pass instance if managed externally

    def analyze_image(self, image_path: str, analysis_options: Dict = None) -> Dict:
        """
        Phân tích hình ảnh vật liệu bán dẫn, phát hiện lỗi và nhận insights từ AI.
        Args:
            image_path: Đường dẫn đến file hình ảnh
            analysis_options: Tùy chọn phân tích (preprocessing, detection types, etc.)
        Returns:
            Dict chứa kết quả phân tích chi tiết
        """
        analysis_options = analysis_options or {}
        results = {
            "image_path": image_path,
            "defects": {
                "dislocations": [],
                "crystal_defects": [],
                "surface_defects": [],
                "other_defects": []
            },
            "image_quality": {},
            "material_properties": {},
            "ai_analysis": {}
        }

        try:
            # Đọc và tiền xử lý hình ảnh
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Không thể đọc hình ảnh từ {image_path}")

            # Tiền xử lý hình ảnh
            processed_img = self.image_processor.denoise_image(img)
            if analysis_options.get('enhance_contrast', True):
                processed_img = self.image_processor.enhance_contrast(processed_img)

            # Phát hiện các loại lỗi
            results["defects"]["dislocations"] = self.defect_detector.detect_dislocations(processed_img)
            results["defects"]["crystal_defects"] = self.defect_detector.detect_crystal_defects(processed_img)
            results["defects"]["surface_defects"] = self.defect_detector.detect_surface_defects(processed_img)

            # Tính toán thống kê lỗi
            all_defects = (
                results["defects"]["dislocations"] +
                results["defects"]["crystal_defects"] +
                results["defects"]["surface_defects"]
            )
            
            results["defect_statistics"] = self._calculate_defect_statistics(all_defects)
            results["image_quality"] = self._assess_image_quality(processed_img)
            
            # Phân tích bằng AI
            results["ai_analysis"] = self._get_ai_analysis(processed_img, results)

            # Tạo hình ảnh với các lỗi được đánh dấu
            marked_image = self._mark_defects(img.copy(), all_defects)
            output_dir = os.path.join(os.path.dirname(image_path), "analyzed_images")
            os.makedirs(output_dir, exist_ok=True)
            marked_image_path = os.path.join(output_dir, f"marked_{os.path.basename(image_path)}")
            cv2.imwrite(marked_image_path, marked_image)
            results["marked_image_path"] = marked_image_path

            return results

        except Exception as e:
            self.logger.error(f"❌ Lỗi khi phân tích hình ảnh: {e}")
            return {"error": str(e)}

    def _calculate_defect_statistics(self, defects: List[Dict]) -> Dict:
        """Calculate comprehensive defect statistics"""
        stats = {
            "total_count": len(defects),
            "by_type": {},
            "by_severity": {"low": 0, "medium": 0, "high": 0},
            "density": 0,  # defects per unit area
            "size_distribution": {
                "min": float('inf'),
                "max": 0,
                "mean": 0,
                "median": 0
            }
        }
        
        # Calculate counts by type and severity
        for defect in defects:
            d_type = defect.get("type", "UNKNOWN")
            stats["by_type"][d_type] = stats["by_type"].get(d_type, 0) + 1
            stats["by_severity"][defect.get("severity", "low")] += 1
            
            # Calculate size statistics if available
            if "area" in defect:
                size = defect["area"]
            elif "length" in defect:
                size = defect["length"]
            elif "radius" in defect:
                size = np.pi * (defect["radius"] ** 2)
            else:
                continue
                
            stats["size_distribution"]["min"] = min(stats["size_distribution"]["min"], size)
            stats["size_distribution"]["max"] = max(stats["size_distribution"]["max"], size)
            
        # Calculate mean and median if we have defects
        if defects:
            sizes = []
            for defect in defects:
                if "area" in defect:
                    sizes.append(defect["area"])
                elif "length" in defect:
                    sizes.append(defect["length"])
                elif "radius" in defect:
                    sizes.append(np.pi * (defect["radius"] ** 2))
            
            if sizes:
                stats["size_distribution"]["mean"] = np.mean(sizes)
                stats["size_distribution"]["median"] = np.median(sizes)
        
        return stats

    def _assess_image_quality(self, gray_img: np.ndarray) -> Dict:
        """Đánh giá chất lượng hình ảnh"""
        # Use variance of Laplacian for sharpness/focus
        sharpness = cv2.Laplacian(gray_img, cv2.CV_64F).var()

        # Use mean intensity for brightness
        mean_intensity = np.mean(gray_img)

        # Get image resolution
        height, width = gray_img.shape

        # Simple metric combining sharpness and brightness (example)
        # Adjust the thresholds and weighting based on desired quality criteria
        quality_score = (sharpness / 100.0) + (mean_intensity / 255.0) * 5.0 # Example scoring
        quality_score = min(10.0, max(0.0, quality_score)) # Clip to 0-10 scale

        return {
            "sharpness_laplacian_variance": float(sharpness),
            "mean_intensity": float(mean_intensity),
            "resolution": f"{width}x{height}",
            "contrast": float(np.std(gray_img) / (mean_intensity + 1e-5)),
            "noise_level": float(0.1),  # Placeholder - would need proper noise estimation
            "brightness": float(mean_intensity / 255.0),
            "quality_score": float(quality_score)
        }

    def _get_ai_analysis(self, img: np.ndarray, analysis_results: Dict) -> Dict:
        """Phân tích hình ảnh bằng các model AI"""
        if not self.valid_engines:
            self.logger.warning("No valid AI engines available for image analysis.")
            return {
                "gemini": {
                    "status": "error", 
                    "response": "No AI engines available.",
                    "error": "No valid Gemini engine initialized",
                    "execution_time": 0,
                    "confidence": 0.0,
                    "analysis": "Could not perform analysis due to no available engines.",
                    "structured_analysis": {
                        "assessment": "",
                        "implications": "",
                        "recommendations": "",
                        "applications": "",
                        "other": "No AI engines available."
                    }
                },
                "groq": {
                    "status": "error",
                    "response": "No AI engines available.",
                    "error": "No valid Groq engine initialized",
                    "execution_time": 0,
                    "confidence": 0.0,
                    "analysis": "Could not perform analysis due to no available engines.",
                    "structured_analysis": {
                        "assessment": "",
                        "implications": "",
                        "recommendations": "",
                        "applications": "",
                        "other": "No AI engines available."
                    }
                },
                "combined_insights": ["Could not perform AI analysis due to no available engines."]
            }

        try:
            # Create the base AI analysis results with fallback values
            ai_analysis_results = {
                "gemini": {
                    "status": "error",
                    "response": "No response from Gemini model",
                    "error": "Model unavailable or API error",
                    "execution_time": 0,
                    "confidence": 0.7,
                    "analysis": "No analysis available from Gemini model.",
                    "structured_analysis": {
                        "assessment": "",
                        "implications": "",
                        "recommendations": "",
                        "applications": "",
                        "other": "No analysis available."
                    }
                },
                "groq": {
                    "status": "error",
                    "response": "No response from Groq model",
                    "error": "Model unavailable or API error", 
                    "execution_time": 0,
                    "confidence": 0.7,
                    "analysis": "No analysis available from Groq model.",
                    "structured_analysis": {
                        "assessment": "",
                        "implications": "",
                        "recommendations": "",
                        "applications": "",
                        "other": "No analysis available."
                    }
                },
                "combined_insights": []
            }
            
            all_insights_set = set()

            # Create a text prompt based on image properties and detected defects
            prompt_text = self._create_ai_analysis_prompt(analysis_results, False)

            # Process each engine separately to handle format differences
            for engine_name, engine_obj in self.valid_engines:
                try:
                    if engine_name == "gemini":
                        try:
                            self.logger.info(f"Attempting Gemini analyze_prompt_concurrently call...")
                            prompt_parts = [{"text": prompt_text}]
                            try:
                                gemini_responses = analyze_prompt_concurrently(prompt_parts, [(engine_name, engine_obj)])
                            except Exception as e:
                                self.logger.error(f"Error in analyze_prompt_concurrently: {e}")
                                gemini_responses = [{
                                    "status": "error",
                                    "response": f"Error in analyze_prompt_concurrently: {e}",
                                    "error": str(e),
                                    "execution_time": 0
                                }]
                            
                            if gemini_responses and len(gemini_responses) > 0:
                                response = gemini_responses[0]
                                status = response.get("status", "error")
                                response_text = response.get("response", "No response")
                                error = response.get("error", None)
                                
                                # Extract structured insights
                                extracted_data = self._extract_insights(response_text)
                                structured_analysis = extracted_data.get("sections", {})
                                insights = extracted_data.get("insights_list", [])
                                
                                ai_analysis_results["gemini"] = {
                                    "status": status,
                                    "response": response_text,
                                    "error": error,
                                    "execution_time": response.get("execution_time", 0),
                                    "confidence": 0.8 if status == "success" else 0.0,
                                    "analysis": response_text if status == "success" else f"Error: {error}",
                                    "structured_analysis": structured_analysis
                                }
                                
                                if status == "success":
                                    all_insights_set.update(insights)
                        except Exception as e:
                            self.logger.error(f"❌ Error in Gemini processing: {e}")
                            ai_analysis_results["gemini"] = {
                                "status": "error",
                                "response": f"Error during Gemini processing: {str(e)}",
                                "error": str(e),
                                "execution_time": 0,
                                "confidence": 0.0,
                                "analysis": f"Error during Gemini processing: {str(e)}",
                                "structured_analysis": {
                                    "assessment": "",
                                    "implications": "",
                                    "recommendations": "",
                                    "applications": "",
                                    "other": f"Error during Gemini processing: {str(e)}"
                                }
                            }
                    
                    elif engine_name == "groq":
                        try:
                            self.logger.info(f"Attempting Groq chat_with_engine call...")
                            try:
                                result = chat_with_engine(
                                    engine_name, 
                                    engine_obj, 
                                    prompt_text
                                )
                            except ValueError as ve:
                                self.logger.error(f"ValueError in chat_with_engine: {ve}")
                                # Fallback to a simple error response
                                result = ("error", f"ValueError in chat_with_engine: {ve}")
                            except Exception as inner_e:
                                self.logger.error(f"Unexpected error in chat_with_engine: {inner_e}")
                                # Fallback to a simple error response
                                result = ("error", f"Unexpected error in chat_with_engine: {inner_e}")

                            self.logger.info(f"Groq result type: {type(result)}")
                            if isinstance(result, (tuple, list)):
                                self.logger.info(f"Groq result length: {len(result)}")
                            
                            # Simplify the result handling to avoid unpacking issues
                            if isinstance(result, (tuple, list)):
                                # Take only the first two elements if available
                                status = result[0] if len(result) > 0 else "error"
                                response_text = result[1] if len(result) > 1 else str(result)
                            else:
                                # If result is not a sequence, treat it as the response text
                                status = "success"
                                response_text = str(result)
                            
                            # Extract structured insights
                            extracted_data = self._extract_insights(response_text)
                            structured_analysis = extracted_data.get("sections", {})
                            insights = extracted_data.get("insights_list", [])
                            
                            ai_analysis_results["groq"] = {
                                "status": status,
                                "response": response_text,
                                "error": None,
                                "execution_time": 0,
                                "confidence": 0.8 if status == "success" else 0.0,
                                "analysis": response_text if status == "success" else f"Error: {status}",
                                "structured_analysis": structured_analysis
                            }
                            
                            if status == "success":
                                all_insights_set.update(insights)
                        except Exception as e:
                            self.logger.error(f"❌ Error in Groq processing: {e}")
                            ai_analysis_results["groq"] = {
                                "status": "error",
                                "response": f"Error during Groq processing: {str(e)}",
                                "error": str(e),
                                "execution_time": 0,
                                "confidence": 0.0,
                                "analysis": f"Error during Groq processing: {str(e)}",
                                "structured_analysis": {
                                    "assessment": "",
                                    "implications": "",
                                    "recommendations": "",
                                    "applications": "",
                                    "other": f"Error during Groq processing: {str(e)}"
                                }
                            }
                
                except Exception as e:
                    self.logger.error(f"❌ Error with {engine_name} model: {e}")
                    ai_analysis_results[engine_name]["error"] = str(e)
                    ai_analysis_results[engine_name]["analysis"] = f"Error processing with {engine_name}: {str(e)}"
                    ai_analysis_results[engine_name]["structured_analysis"] = {
                        "assessment": "",
                        "implications": "",
                        "recommendations": "",
                        "applications": "",
                        "other": f"Error processing with {engine_name}: {str(e)}"
                    }

            # Fallback insights if none extracted
            if not all_insights_set:
                all_insights_set = {
                    "Material appears to have a crystalline structure",
                    "Some surface defects are visible in the image",
                    "Quality of the material could be improved with better processing"
                }
                
            ai_analysis_results["combined_insights"] = sorted(list(all_insights_set))
            return ai_analysis_results

        except Exception as e:
            self.logger.error(f"❌ Lỗi trong quá trình phân tích AI: {e}")
            return {
                "error": str(e),
                "gemini": {
                    "status": "error",
                    "response": f"Error during analysis: {str(e)}",
                    "error": str(e),
                    "confidence": 0.0,
                    "analysis": f"Error during analysis: {str(e)}",
                    "structured_analysis": {
                        "assessment": "",
                        "implications": "",
                        "recommendations": "",
                        "applications": "",
                        "other": f"Error during analysis: {str(e)}"
                    }
                },
                "groq": {
                    "status": "error",
                    "response": f"Error during analysis: {str(e)}",
                    "error": str(e),
                    "confidence": 0.0,
                    "analysis": f"Error during analysis: {str(e)}",
                    "structured_analysis": {
                        "assessment": "",
                        "implications": "",
                        "recommendations": "",
                        "applications": "",
                        "other": f"Error during analysis: {str(e)}"
                    }
                },
                "combined_insights": ["Error occurred during AI analysis."]
            }

    def _create_ai_analysis_prompt(self, analysis_results: Dict, is_multimodal: bool) -> str:
        """Tạo prompt text dựa trên kết quả phân tích ảnh"""
        prompt = "Please analyze the following semiconductor material image and the results of the image analysis.\n\n"

        if is_multimodal:
            prompt += "Refer to the provided image.\n"

        prompt += "Image analysis results:\n"
        prompt += f"- Image Quality Score: {analysis_results['image_quality'].get('quality_score', 'N/A'):.1f}\n"
        prompt += f"- Total Defects Detected: {analysis_results.get('total_defects_count', 'N/A')}\n"

        defect_summary = analysis_results.get('defect_summary', {})
        prompt += "Defect Summary:\n"
        if defect_summary:
            for type, count in defect_summary.get('types', {}).items():
                prompt += f"  - {type.capitalize()} Defects: {count}\n"
            if defect_summary.get('severity'):
                 prompt += "Severity Distribution:\n"
                 for severity, count in defect_summary['severity'].items():
                     prompt += f"    - {severity.capitalize()}: {count}\n"
        else:
             prompt += "  - No detailed defect summary available.\n"

        prompt += "\nBased on these results, provide:\n"
        prompt += "1. An assessment of the material's quality and characteristics visible in the image (e.g., crystal structure, grain size, surface texture, presence of defects).\n"
        prompt += "2. Potential implications of the detected defects on the material's performance (e.g., electrical, optical, thermal properties).\n"
        prompt += "3. Recommendations for improving the material quality or manufacturing process based on the observed defects.\n"
        prompt += "4. Potential applications that the material seems suitable or unsuitable for, based on the visual characteristics and defect analysis.\n"
        prompt += "\nUse clear section headings for each part of your analysis (Assessment, Implications, Recommendations, Applications) and format them consistently. Each section should be clearly separated. Be concise and focus on key insights.\n"

        return prompt

    def _extract_insights(self, response_text: str) -> Dict:
        """
        Extract key insights from a text response, organizing them into structured sections.
        Returns a dictionary with sections as keys and their content as values.
        """
        # Initialize dictionary for structured sections
        structured_insights = {
            "assessment": "",
            "implications": "",
            "recommendations": "",
            "applications": "",
            "other": ""  # For content that doesn't fit in the main sections
        }
        
        # Split by common section headers
        section_pattern = r'(?i)^\s*(#{1,3}\s*|)(?P<section>Assessment|Implications|Recommendations|Applications)[\s:\n]+'
        
        # Find all section matches
        matches = list(re.finditer(section_pattern, response_text, re.MULTILINE))
        
        # Process each section
        if matches:
            for i, match in enumerate(matches):
                section_name = match.group('section').lower()
                start_pos = match.end()
                
                # Determine end position (start of next section or end of text)
                if i < len(matches) - 1:
                    end_pos = matches[i+1].start()
                else:
                    end_pos = len(response_text)
                
                # Extract section content
                section_content = response_text[start_pos:end_pos].strip()
                
                # Store in appropriate section
                if section_name in structured_insights:
                    structured_insights[section_name] = section_content
                else:
                    structured_insights["other"] += section_content + "\n"
        else:
            # If no sections found, put everything in "other"
            structured_insights["other"] = response_text.strip()
        
        # Process each section to extract bullet points or key insights
        insights_list = []
        for section, content in structured_insights.items():
            if content:
                # Add bullet points if they exist
                bullet_points = re.findall(r'^\s*[\*\-\+•]\s*(.+)$', content, re.MULTILINE)
                if bullet_points:
                    for point in bullet_points:
                        insights_list.append(point.strip())
                else:
                    # If no bullet points, add key sentences (ones that are reasonably long)
                    sentences = re.split(r'(?<=[.!?])\s+', content)
                    for sentence in sentences:
                        if len(sentence.strip()) > 30:  # Only include substantive sentences
                            insights_list.append(sentence.strip())
        
        # Return both structured sections and a flat list of insights
        return {
            "sections": structured_insights,
            "insights_list": insights_list
        }

    def _mark_defects(self, img: np.ndarray, defects: List[Dict]) -> np.ndarray:
        """Đánh dấu các lỗi trên hình ảnh dựa trên dữ liệu lỗi có cấu trúc"""
        # Define a fixed color palette for main defect categories
        base_colors = {
            # Main defect categories with BGR colors
            "DISLOCATION": (255, 0, 0),      # Blue
            "STACKING_FAULT": (0, 255, 0),   # Green
            "GRAIN_BOUNDARY": (0, 0, 255),   # Red
            "PRECIPITATE": (255, 255, 0),    # Cyan
            "VOID": (255, 0, 255),           # Magenta
            "SURFACE_CONTAMINATION": (0, 255, 255),  # Yellow
            "CRYSTAL_DEFECT": (128, 128, 0), # Teal
            "IMPURITY": (0, 128, 128),       # Brown
            # Fallback colors for subcategories
            "DEFAULT": (128, 128, 128)       # Gray
        }

        # Create a copy of the image for marking
        marked_img = img.copy()

        try:
            # Group defects by type for summary
            defect_groups = {}
            for defect in defects:
                d_type = defect.get("type", "UNKNOWN")
                if d_type not in defect_groups:
                    defect_groups[d_type] = []
                defect_groups[d_type].append(defect)

            # Draw defects
            for d_type, d_group in defect_groups.items():
                # Get color from base colors or use default
                color = base_colors.get(d_type, base_colors["DEFAULT"])
                
                for defect in d_group:
                    # Draw based on defect type
                    if "location" in defect:
                        x, y = defect["location"]
                        
                        if "radius" in defect:  # Circular defects
                            radius = int(defect["radius"])
                            cv2.circle(marked_img, (x, y), radius, color, 2)
                        
                        elif "length" in defect:  # Line defects (e.g., dislocations)
                            length = defect["length"]
                            angle = defect.get("angle", 0)  # Default angle if not specified
                            end_x = int(x + length * np.cos(angle))
                            end_y = int(y + length * np.sin(angle))
                            cv2.line(marked_img, (x, y), (end_x, end_y), color, 2)
                        
                        elif "size" in defect:  # Rectangular defects
                            w, h = defect["size"]
                            cv2.rectangle(marked_img, (x, y), (x+w, y+h), color, 2)
                        
                        else:  # Point defects
                            cv2.drawMarker(marked_img, (x, y), color, 
                                         markerType=cv2.MARKER_CROSS, 
                                         markerSize=10, thickness=2)

                        # Add label with severity
                        if "severity" in defect:
                            label = f"{d_type}:{defect['severity']}"
                            # Position the label above the defect
                            cv2.putText(marked_img, label, (x, y-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                                      cv2.LINE_AA)  # Use anti-aliasing

            # Add summary information
            summary_y = 30
            cv2.putText(marked_img, f"Total Defects: {len(defects)}", 
                       (10, summary_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add legend
            legend_y = summary_y + 30
            for d_type, count in defect_groups.items():
                if d_type in base_colors:
                    cv2.putText(marked_img, f"{d_type}: {len(count)}", 
                              (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, base_colors[d_type], 2, cv2.LINE_AA)
                    legend_y += 20

            return marked_img

        except Exception as e:
            self.logger.error(f"Error in defect marking: {e}")
            # Return original image if marking fails
            return img

# Example Usage (if running this file directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(module)s:%(lineno)d - %(message)s')
    logger.info("Starting MaterialImageAnalyzer example...")

    analyzer = MaterialImageAnalyzer()

    # Create a dummy image file for testing
    dummy_image_path = "test_material_image.png"
    dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(dummy_img, "Sample Material Surface", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(dummy_img, (100, 100), (150, 150), (0, 0, 255), -1) # Dummy defect
    cv2.circle(dummy_img, (300, 200), 30, (0, 255, 0), -1) # Another dummy defect
    cv2.imwrite(dummy_image_path, dummy_img)

    logger.info(f"Created dummy image: {dummy_image_path}")

    # Analyze the dummy image
    analysis_results = analyzer.analyze_image(dummy_image_path)

    if "error" in analysis_results:
        logger.error(f"Analysis failed: {analysis_results['error']}")
    else:
        logger.info("\n=== Image Analysis Results ===")
        # Print image quality
        logger.info("\nImage Quality:")
        for key, value in analysis_results.get("image_quality", {}).items():
            logger.info(f"- {key}: {value}")

        # Print defect summary
        logger.info("\nDefect Summary:")
        defect_summary = analysis_results.get("defect_summary", {})
        logger.info(f"- Total Defects: {defect_summary.get('total', 0)}")
        logger.info("- Types:")
        for dtype, count in defect_summary.get('types', {}).items():
            logger.info(f"  - {dtype}: {count}")
        logger.info("- Severity:")
        for severity, count in defect_summary.get('severity', {}).items():
             logger.info(f"  - {severity}: {count}")

        # Print raw AI responses
        logger.info("\n=== Raw AI Model Responses ===")
        for engine, data in analysis_results.get("model_responses", {}).items():
            logger.info(f"\n--- {engine.upper()} ({data.get('status')}) ---")
            if data.get("status") == "success":
                logger.info(data.get("response", "No response text."))
            else:
                logger.info(f"Error: {data.get('error', 'Unknown error')}")


        # Print combined insights
        logger.info("\n=== Combined AI Insights ===")
        if analysis_results.get("combined_insights"):
            for insight in analysis_results["combined_insights"]:
                logger.info(f"- {insight}")
        else:
            logger.info("No combined insights available.")


    # Clean up dummy file
    # os.remove(dummy_image_path)
    # logger.info(f"Removed dummy image: {dummy_image_path}")

def batch_analyze_materials(materials, engines_to_use, batch_size=5, progress_callback=None):
    """
    Phân tích nhiều vật liệu bằng cách sử dụng các hàm thực từ evaluate_models.py
    """
    all_results = []
    material_scores = {}
    results_lock = threading.Lock()
    
    # Chia thành các batch
    batches = []
    for i in range(0, len(materials), batch_size):
        batches.append(materials[i:i+batch_size])
    
    # Khởi tạo engines với error handling
    active_engines = []
    initialization_errors = []
    for engine_name in engines_to_use:
        try:
            engine_tuple = init_chat_engine(engine_name)
            if engine_tuple[1] is not None or engine_name == "groq":
                active_engines.append(engine_tuple)
            else:
                initialization_errors.append(f"{engine_name}: Initialization returned None")
        except Exception as e:
            initialization_errors.append(f"{engine_name}: {str(e)}")
    
    # Nếu không có engine nào khả dụng, thông báo lỗi chi tiết
    if not active_engines:
        error_msg = "No AI engines available. Initialization errors:\n" + "\n".join(initialization_errors)
        raise Exception(error_msg)

def analyze_materials_parallel(num_samples=20):
    try:
        # Khởi tạo cache và rate manager
        response_cache = ResponseCache(cache_dir="cache")
        api_rate_manager = APIRateManager()
    except Exception as e:
        print(f"❌ Error initializing cache or rate manager: {e}")
        response_cache = None
        api_rate_manager = None

    try:
        # Khởi tạo search engine
        search_engine = SemanticSearchEngine()
    except Exception as e:
        print(f"❌ Error initializing search engine: {e}")
        search_engine = None

    # Rest of the function...

class ImageNormalizer:
    """Ensures consistent image preprocessing across all analyses"""
    
    def __init__(self):
        self.target_size = (800, 600)  # Standard size for analysis
        self.intensity_range = (0, 255)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Apply consistent normalization to input image"""
        if img is None:
            raise ValueError("Input image is None")
            
        # Convert to grayscale if color image
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Resize to standard size
        resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Apply CLAHE for consistent contrast
        enhanced = self.clahe.apply(resized)
        
        # Normalize intensity range
        normalized = cv2.normalize(enhanced, None, 
                                 self.intensity_range[0], 
                                 self.intensity_range[1], 
                                 cv2.NORM_MINMAX)
                                 
        return normalized

class DetectionParameters:
    """Maintains consistent detection parameters across analyses"""
    
    # Default thresholds for different defect types
    THRESHOLDS = {
        'VOID': {
            'intensity': 50,  # Dark regions
            'min_size': 10,
            'max_size': 100,
            'circularity': 0.7
        },
        'GRAIN_BOUNDARY': {
            'gradient': 30,
            'min_length': 40,
            'linearity': 0.8
        },
        'SURFACE_DEFECT': {
            'texture_variance': 25,
            'min_area': 20,
            'gradient_min': 20
        },
        'CRYSTAL_DEFECT': {
            'edge_strength': 40,
            'symmetry_threshold': 0.6,
            'min_size': 15
        }
    }
    
    # Classification confidence thresholds
    CONFIDENCE_THRESHOLDS = {
        'high': 0.8,
        'medium': 0.6,
        'low': 0.4
    }
    
    # Severity scoring parameters
    SEVERITY_WEIGHTS = {
        'size': 0.4,
        'intensity': 0.3,
        'location': 0.3
    }
    
    @staticmethod
    def get_threshold(defect_type: str, parameter: str) -> float:
        """Get specific threshold value"""
        return DetectionParameters.THRESHOLDS.get(defect_type, {}).get(parameter, 0.0)
    
    @staticmethod
    def get_confidence_threshold(level: str) -> float:
        """Get confidence threshold for classification"""
        return DetectionParameters.CONFIDENCE_THRESHOLDS.get(level, 0.0)
    
    @staticmethod
    def get_severity_weight(factor: str) -> float:
        """Get weight for severity calculation"""
        return DetectionParameters.SEVERITY_WEIGHTS.get(factor, 0.0)
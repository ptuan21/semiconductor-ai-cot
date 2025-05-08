import pandas as pd
import numpy as np
import random
import os
import json
import re

# Định nghĩa dữ liệu vật liệu bán dẫn phổ biến với thuộc tính thực tế
SEMICONDUCTOR_MATERIALS = {
    # Group IV semiconductors
    "Si": {
        "name": "Silicon",
        "formula": "Si",
        "crystal_structures": ["Cubic (Diamond)", "Amorphous"],
        "typical_bandgap": 1.12,
        "typical_conductivity": 1000,
        "thermal_stability": "High",
        "common_applications": ["Solar cells", "Transistors", "Integrated circuits"]
    },
    "Ge": {
        "name": "Germanium",
        "formula": "Ge",
        "crystal_structures": ["Cubic (Diamond)", "Amorphous"],
        "typical_bandgap": 0.67,
        "typical_conductivity": 2000,
        "thermal_stability": "Medium",
        "common_applications": ["Transistors", "Infrared optics", "Photovoltaics"]
    },
    "SiC": {
        "name": "Silicon Carbide",
        "formula": "SiC",
        "crystal_structures": ["Hexagonal (4H, 6H)", "Cubic (3C)"],
        "typical_bandgap": 3.26,
        "typical_conductivity": 500,
        "thermal_stability": "Very High",
        "common_applications": ["High-power electronics", "High-temperature devices", "LED"]
    },
    
    # III-V semiconductors
    "GaAs": {
        "name": "Gallium Arsenide",
        "formula": "GaAs",
        "crystal_structures": ["Zincblende"],
        "typical_bandgap": 1.43,
        "typical_conductivity": 5000,
        "thermal_stability": "Medium",
        "common_applications": ["High-speed devices", "Solar cells", "LEDs", "Lasers"]
    },
    "GaN": {
        "name": "Gallium Nitride",
        "formula": "GaN",
        "crystal_structures": ["Wurtzite", "Zincblende"],
        "typical_bandgap": 3.4,
        "typical_conductivity": 1200,
        "thermal_stability": "High",
        "common_applications": ["Blue LEDs", "High-power transistors", "RF devices"]
    },
    "InP": {
        "name": "Indium Phosphide",
        "formula": "InP",
        "crystal_structures": ["Zincblende"],
        "typical_bandgap": 1.35,
        "typical_conductivity": 4600,
        "thermal_stability": "Medium",
        "common_applications": ["High-frequency applications", "Photonic integrated circuits", "Solar cells"]
    },
    "AlGaAs": {
        "name": "Aluminium Gallium Arsenide",
        "formula": "AlGaAs",
        "crystal_structures": ["Zincblende"],
        "typical_bandgap": 1.8,
        "typical_conductivity": 3000,
        "thermal_stability": "Medium",
        "common_applications": ["Laser diodes", "Heterojunction transistors", "Solar cells"]
    },
    
    # II-VI semiconductors
    "ZnO": {
        "name": "Zinc Oxide",
        "formula": "ZnO",
        "crystal_structures": ["Wurtzite", "Zincblende", "Rocksalt"],
        "typical_bandgap": 3.37,
        "typical_conductivity": 200,
        "thermal_stability": "High",
        "common_applications": ["Transparent electronics", "UV sensors", "Varistors"]
    },
    "CdTe": {
        "name": "Cadmium Telluride",
        "formula": "CdTe",
        "crystal_structures": ["Zincblende"],
        "typical_bandgap": 1.5,
        "typical_conductivity": 650,
        "thermal_stability": "Medium",
        "common_applications": ["Solar cells", "Radiation detectors", "Medical imaging"]
    },
    "CdS": {
        "name": "Cadmium Sulfide",
        "formula": "CdS",
        "crystal_structures": ["Wurtzite", "Zincblende"],
        "typical_bandgap": 2.42,
        "typical_conductivity": 10,
        "thermal_stability": "Medium",
        "common_applications": ["Photoresistors", "Solar cells", "Optical filters"]
    },
    
    # Oxides
    "Cu2O": {
        "name": "Copper(I) Oxide",
        "formula": "Cu2O",
        "crystal_structures": ["Cubic"],
        "typical_bandgap": 2.0,
        "typical_conductivity": 1,
        "thermal_stability": "Medium",
        "common_applications": ["Solar cells", "Photocatalysts", "Sensors"]
    },
    "TiO2": {
        "name": "Titanium Dioxide",
        "formula": "TiO2",
        "crystal_structures": ["Rutile", "Anatase", "Brookite"],
        "typical_bandgap": 3.2,
        "typical_conductivity": 0.1,
        "thermal_stability": "High",
        "common_applications": ["Photocatalysts", "Solar cells", "Memristors"]
    },
    "SnO2": {
        "name": "Tin Dioxide",
        "formula": "SnO2",
        "crystal_structures": ["Rutile", "Cassiterite"],
        "typical_bandgap": 3.6,
        "typical_conductivity": 5,
        "thermal_stability": "High",
        "common_applications": ["Transparent conductors", "Gas sensors", "Touch screens"]
    },
    
    # Perovskites
    "MAPbI3": {
        "name": "Methylammonium Lead Iodide",
        "formula": "CH3NH3PbI3",
        "crystal_structures": ["Tetragonal", "Cubic"],
        "typical_bandgap": 1.55,
        "typical_conductivity": 50,
        "thermal_stability": "Low",
        "common_applications": ["Solar cells", "Photodetectors", "Light-emitting devices"]
    },
    
    # 2D materials
    "Graphene": {
        "name": "Graphene",
        "formula": "C",
        "crystal_structures": ["Hexagonal"],
        "typical_bandgap": 0,
        "typical_conductivity": 10000,
        "thermal_stability": "Very High",
        "common_applications": ["Transparent conductors", "Field-effect transistors", "Sensors"]
    },
    "MoS2": {
        "name": "Molybdenum Disulfide",
        "formula": "MoS2",
        "crystal_structures": ["Hexagonal"],
        "typical_bandgap": 1.8,
        "typical_conductivity": 200,
        "thermal_stability": "High",
        "common_applications": ["Field-effect transistors", "Photodetectors", "Lubricants"]
    }
}

# Danh sách biến thể vật liệu
MATERIAL_VARIATIONS = [
    "Pure",
    "Doped (n-type)",
    "Doped (p-type)",
    "Nanostructured",
    "Thin Film",
    "Polycrystalline",
    "Single Crystal",
    "Amorphous",
    "Quantum Dots",
    "Nanowires"
]

# Danh sách danh mục ứng dụng tiềm năng
APPLICATION_CATEGORIES = {
    "Solar Cells": {
        "ideal_bandgap": [1.0, 1.7],  # eV
        "ideal_conductivity": [10, 5000],  # S/cm
        "absorption_coef_threshold": 1e5,  # cm^-1
        "thermal_stability": ["Medium", "High", "Very High"]
    },
    "LEDs": {
        "ideal_bandgap": [1.5, 3.5],  # eV
        "ideal_conductivity": [100, 5000],  # S/cm
        "direct_bandgap": True
    },
    "High Power Electronics": {
        "ideal_bandgap": [2.5, 5.0],  # eV
        "thermal_stability": ["High", "Very High"],
        "breakdown_field_threshold": 1.0  # MV/cm
    },
    "Transistors": {
        "ideal_mobility": [100, 10000],  # cm^2/Vs
        "ideal_conductivity": [500, 10000]  # S/cm
    },
    "Photodetectors": {
        "ideal_bandgap": [0.5, 3.0],  # eV
        "absorption_coef_threshold": 5e4  # cm^-1
    },
    "Thermoelectrics": {
        "ideal_bandgap": [0.1, 1.0],  # eV
        "ideal_conductivity": [500, 5000],  # S/cm
        "thermal_conductivity_threshold": 10  # W/mK (thấp là tốt)
    },
    "Transparent Conductors": {
        "ideal_bandgap": [3.0, 5.0],  # eV
        "ideal_conductivity": [100, 5000]  # S/cm
    }
}

# Hàm để tạo sự tương quan giữa các thuộc tính
def create_correlated_properties(base_material, variation):
    """Tạo thuộc tính vật liệu có tương quan thực tế với nhau dựa trên biến thể"""
    material_data = SEMICONDUCTOR_MATERIALS[base_material]
    
    # Lấy thông số cơ bản
    bandgap_base = material_data["typical_bandgap"]
    conductivity_base = material_data["typical_conductivity"]
    thermal_stability = material_data["thermal_stability"]
    
    # Khởi tạo các thuộc tính phụ thuộc
    mobility = conductivity_base * random.uniform(0.8, 1.5)  # cm²/Vs
    carrier_concentration = conductivity_base / (1.6e-19 * mobility) * random.uniform(0.7, 1.3)  # cm^-3
    thermal_conductivity = np.random.uniform(10, 150)  # W/mK
    refractive_index = 1.5 + bandgap_base * random.uniform(0.3, 0.5)
    
    # Điều chỉnh dựa trên biến thể vật liệu
    if "Doped" in variation:
        if "n-type" in variation:
            bandgap = bandgap_base * random.uniform(0.95, 1.05)
            conductivity = conductivity_base * random.uniform(1.5, 5.0)
            carrier_concentration *= random.uniform(10, 100)  # Tăng đáng kể
            mobility *= random.uniform(0.7, 0.9)  # Giảm nhẹ
        else:  # p-type
            bandgap = bandgap_base * random.uniform(0.95, 1.05)
            conductivity = conductivity_base * random.uniform(1.2, 3.0)
            carrier_concentration *= random.uniform(5, 50)
            mobility *= random.uniform(0.5, 0.8)  # Giảm nhiều hơn do p-type
    
    elif "Nano" in variation or "Quantum" in variation:
        bandgap = bandgap_base * random.uniform(1.1, 1.5)  # Quantum confinement tăng bandgap
        conductivity = conductivity_base * random.uniform(0.3, 0.8)  # Giảm do tán xạ biên
        thermal_conductivity *= random.uniform(0.3, 0.7)  # Giảm đáng kể 
        refractive_index += random.uniform(0.1, 0.5)  # Có thể tăng do hiệu ứng kích thước
    
    elif "Thin Film" in variation:
        bandgap = bandgap_base * random.uniform(0.9, 1.1)
        conductivity = conductivity_base * random.uniform(0.5, 0.9)
        mobility *= random.uniform(0.6, 0.9)
        thermal_conductivity *= random.uniform(0.5, 0.8)
    
    elif "Amorphous" in variation:
        bandgap = bandgap_base * random.uniform(0.8, 1.0)
        conductivity = conductivity_base * random.uniform(0.01, 0.3)
        mobility *= random.uniform(0.05, 0.3)
        thermal_conductivity *= random.uniform(0.2, 0.5)
    
    elif "Polycrystalline" in variation:
        bandgap = bandgap_base * random.uniform(0.95, 1.05)
        conductivity = conductivity_base * random.uniform(0.4, 0.8)
        mobility *= random.uniform(0.4, 0.7)
        thermal_conductivity *= random.uniform(0.6, 0.9)
    
    else:  # Pure or Single Crystal
        bandgap = bandgap_base * random.uniform(0.98, 1.02)
        conductivity = conductivity_base * random.uniform(0.9, 1.1)
        mobility *= random.uniform(0.9, 1.1)
        thermal_conductivity *= random.uniform(0.9, 1.1)
    
    # Tính hệ số hấp thụ (phụ thuộc vào bandgap)
    # Giả định: vật liệu có bandgap thấp thường hấp thụ tốt hơn
    if bandgap < 0.01:  # Kiểm tra bandgap gần bằng 0
        # Với vật liệu không có bandgap như Graphene, sử dụng giá trị cao
        absorption_coefficient = 1e6 * random.uniform(0.8, 1.2)
    else:
        absorption_coefficient = 1e5 / (bandgap ** 2) * random.uniform(0.8, 1.2)
    
    # Xác định kiểu bandgap (direct/indirect)
    # Giả định đơn giản: một số vật liệu điển hình có bandgap trực tiếp
    direct_bandgap_materials = ["GaAs", "InP", "GaN", "CdTe", "CdS", "MAPbI3"]
    is_direct = 1 if base_material in direct_bandgap_materials else 0
    
    # Xác định breakdown field dựa trên bandgap
    # Quy tắc thực nghiệm: breakdown field tỷ lệ với (bandgap)^2.5
    breakdown_field = (bandgap ** 2.5) * 0.1 * random.uniform(0.8, 1.2)  # MV/cm
    
    # Chọn ngẫu nhiên một cấu trúc tinh thể từ danh sách có sẵn
    if "Amorphous" in variation:
        crystal_structure = "Amorphous"
    else:
        crystal_structure = random.choice(material_data["crystal_structures"])
    
    # Map "Very High", "High", "Medium", "Low" thành giá trị số cho thermal stability
    thermal_stability_map = {
        "Very High": random.uniform(800, 1200),
        "High": random.uniform(500, 800),
        "Medium": random.uniform(300, 500),
        "Low": random.uniform(100, 300)
    }
    max_temperature = thermal_stability_map.get(thermal_stability, 300)
    
    # Xác định các ứng dụng tiềm năng dựa trên thuộc tính
    potential_applications = []
    
    for app, criteria in APPLICATION_CATEGORIES.items():
        matches_criteria = True
        
        if "ideal_bandgap" in criteria:
            min_bg, max_bg = criteria["ideal_bandgap"]
            if not (min_bg <= bandgap <= max_bg):
                matches_criteria = False
                
        if "ideal_conductivity" in criteria:
            min_cond, max_cond = criteria["ideal_conductivity"]
            if not (min_cond <= conductivity <= max_cond):
                matches_criteria = False
                
        if "absorption_coef_threshold" in criteria and absorption_coefficient < criteria["absorption_coef_threshold"]:
            matches_criteria = False
            
        if "thermal_stability" in criteria and thermal_stability not in criteria["thermal_stability"]:
            matches_criteria = False
            
        if "ideal_mobility" in criteria:
            min_mob, max_mob = criteria["ideal_mobility"]
            if not (min_mob <= mobility <= max_mob):
                matches_criteria = False
                
        if "breakdown_field_threshold" in criteria and breakdown_field < criteria["breakdown_field_threshold"]:
            matches_criteria = False
            
        if "direct_bandgap" in criteria and criteria["direct_bandgap"] != bool(is_direct):
            matches_criteria = False
            
        if matches_criteria:
            potential_applications.append(app)
    
    # Tạo ID duy nhất cho vật liệu
    material_id = base_material + "_" + variation.replace(" ", "_").replace("(", "").replace(")", "")
    
    return {
        "material_id": material_id,
        "material_name": material_data["name"],
        "variation": variation,
        "formula": material_data["formula"],
        "crystal_structure": crystal_structure,
        "bandgap_energy_eV": round(bandgap, 3),
        "is_direct_bandgap": bool(is_direct),
        "conductivity_S_cm": round(conductivity, 2),
        "mobility_cm2_Vs": round(mobility, 2),
        "carrier_concentration_cm3": format(carrier_concentration, ".2e"),
        "thermal_conductivity_W_mK": round(thermal_conductivity, 2),
        "breakdown_field_MV_cm": round(breakdown_field, 3),
        "absorption_coefficient_cm1": format(absorption_coefficient, ".2e"),
        "refractive_index": round(refractive_index, 2),
        "max_operating_temp_C": round(max_temperature, 1),
        "thermal_stability": thermal_stability,
        "potential_applications": ", ".join(potential_applications) if potential_applications else "General semiconductor",
    }

# Hàm tạo dữ liệu giả
def generate_fake_data(num_samples, include_variations=True):
    """
    Tạo dữ liệu giả lập các vật liệu bán dẫn với thuộc tính ngẫu nhiên và đa dạng.
    
    Args:
        num_samples: Số lượng mẫu cần tạo
        include_variations: Không còn được sử dụng (giữ lại để tương thích ngược)
        
    Returns:
        DataFrame chứa dữ liệu vật liệu bán dẫn
    """
    data = []
    
    # Danh sách các vật liệu cơ bản để tham khảo
    material_keys = list(SEMICONDUCTOR_MATERIALS.keys())
    
    # Tạo ngẫu nhiên hoàn toàn các vật liệu
    for i in range(num_samples):
        # Ngẫu nhiên chọn một trong các phương pháp tạo dữ liệu:
        # 1. Tạo vật liệu từ scratch
        # 2. Lấy từ vật liệu cơ bản nhưng thay đổi mạnh đặc tính
        # 3. Kết hợp giữa các vật liệu (hybrid)
        generation_method = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        
        if generation_method == 1:
            # Phương pháp 1: Tạo vật liệu hoàn toàn ngẫu nhiên
            material_id = f"RAND-{i+1:04d}"
            
            # Tạo tên ngẫu nhiên
            prefixes = ["Nova", "Quantum", "Flex", "Synth", "Neo", "Pico", "Macro", "Ultra", "Micro", "Nano"]
            suffixes = ["lite", "dyne", "tron", "zine", "flux", "dex", "nium", "mide", "phene", "cite"]
            base_elements = ["Si", "Ge", "Ga", "As", "C", "B", "Al", "In", "P", "S", "Se", "Te", "Zn", "Cd", "Ti", "Sn", "Pb"]
            
            material_name = np.random.choice(prefixes) + np.random.choice(base_elements) + np.random.choice(suffixes)
            formula = "".join([e + str(np.random.randint(1, 4)) if np.random.random() > 0.5 else e 
                             for e in np.random.choice(base_elements, np.random.randint(1, 4), replace=False)])
            
            # Tạo thuộc tính ngẫu nhiên
            crystal_structures = ["Cubic", "Hexagonal", "Tetragonal", "Orthorhombic", "Monoclinic", "Triclinic", "Amorphous"]
            crystal_structure = np.random.choice(crystal_structures)
            
            # Tạo bandgap ngẫu nhiên từ 0 đến 5.5 eV
            bandgap = np.random.uniform(0, 5.5)
            
            # Tạo conductivity ngẫu nhiên (phân phối log từ 0.01 đến 10000)
            conductivity = 10 ** np.random.uniform(-2, 4)
            
            # Tạo các thuộc tính khác
            is_direct = np.random.choice([0, 1])
            mobility = np.random.uniform(10, 5000)
            carrier_concentration = 10 ** np.random.uniform(15, 22)
            thermal_conductivity = np.random.uniform(0.5, 500)
            breakdown_field = np.random.uniform(0.1, 10)
            absorption_coefficient = 10 ** np.random.uniform(3, 6)
            refractive_index = np.random.uniform(1.5, 4.5)
            max_temperature = np.random.uniform(100, 1200)
            
            # Xác định thermal stability dựa trên max_temperature
            if max_temperature >= 800:
                thermal_stability = "Very High"
            elif max_temperature >= 500:
                thermal_stability = "High"
            elif max_temperature >= 300:
                thermal_stability = "Medium"
            else:
                thermal_stability = "Low"
            
        elif generation_method == 2:
            # Phương pháp 2: Lấy từ vật liệu cơ bản nhưng thay đổi mạnh đặc tính
            base_material = np.random.choice(material_keys)
            material_data = SEMICONDUCTOR_MATERIALS[base_material]
            
            material_id = f"MOD-{base_material}-{i+1:04d}"
            material_name = material_data["name"]
            
            # Thêm mô tả ngẫu nhiên
            modifiers = ["Enhanced", "Modified", "Engineered", "Advanced", "High-performance", "Custom", "Next-gen"]
            structure_mods = ["nanoporous", "mesoporous", "nanostructured", "quantum-confined", "2D", "strained", "epitaxial"]
            doping_mods = ["heavily doped", "lightly doped", "selectively doped", "co-doped", "gradient-doped"]
            
            # 50% chance áp dụng modifier
            if np.random.random() > 0.5:
                mod_type = np.random.choice(["structure", "doping", "both"])
                if mod_type == "structure":
                    material_name = f"{np.random.choice(modifiers)} {material_name} ({np.random.choice(structure_mods)})"
                elif mod_type == "doping":
                    material_name = f"{np.random.choice(modifiers)} {material_name} ({np.random.choice(doping_mods)})"
                else:
                    material_name = f"{np.random.choice(modifiers)} {material_name} ({np.random.choice(structure_mods)}, {np.random.choice(doping_mods)})"
            
            formula = material_data["formula"]
            crystal_structure = np.random.choice(material_data["crystal_structures"])
            
            # Thay đổi bandgap ± 50% so với giá trị gốc
            bandgap = material_data["typical_bandgap"] * np.random.uniform(0.5, 1.5)
            
            # Thay đổi conductivity ± 80% so với giá trị gốc
            conductivity = material_data["typical_conductivity"] * np.random.uniform(0.2, 1.8)
            
            # Các thuộc tính khác cũng ngẫu nhiên
            is_direct = 1 if base_material in ["GaAs", "InP", "GaN", "CdTe", "CdS", "MAPbI3"] else np.random.choice([0, 1])
            mobility = conductivity * np.random.uniform(0.5, 2.0)
            carrier_concentration = 10 ** np.random.uniform(15, 22)
            thermal_conductivity = np.random.uniform(0.5, 500)
            breakdown_field = (bandgap ** 2.5) * 0.1 * np.random.uniform(0.5, 1.5)
            absorption_coefficient = 10 ** np.random.uniform(3, 6)
            refractive_index = np.random.uniform(1.5, 4.5)
            
            # Thay đổi thermal stability ngẫu nhiên
            thermal_stabilities = ["Low", "Medium", "High", "Very High"]
            current_index = thermal_stabilities.index(material_data["thermal_stability"])
            new_index = current_index + np.random.randint(-1, 2)
            new_index = max(0, min(len(thermal_stabilities) - 1, new_index))
            thermal_stability = thermal_stabilities[new_index]
            
            # Xác định max_temperature dựa trên thermal_stability
            thermal_stability_map = {
                "Very High": np.random.uniform(800, 1200),
                "High": np.random.uniform(500, 800),
                "Medium": np.random.uniform(300, 500),
                "Low": np.random.uniform(100, 300)
            }
            max_temperature = thermal_stability_map[thermal_stability]
            
        else:
            # Phương pháp 3: Tạo vật liệu lai (hybrid)
            # Chọn ngẫu nhiên 2 vật liệu cơ bản để lai
            base1, base2 = np.random.choice(material_keys, 2, replace=False)
            material1 = SEMICONDUCTOR_MATERIALS[base1]
            material2 = SEMICONDUCTOR_MATERIALS[base2]
            
            material_id = f"HYB-{base1}-{base2}-{i+1:04d}"
            material_name = f"Hybrid {material1['name']}-{material2['name']}"
            
            # Tạo công thức hóa học lai
            elements1 = re.findall(r'([A-Z][a-z]*)(\d*)', material1["formula"])
            elements2 = re.findall(r'([A-Z][a-z]*)(\d*)', material2["formula"])
            
            # Đơn giản hóa bằng cách ghép công thức
            formula = material1["formula"] + "-" + material2["formula"]
            
            # Chọn ngẫu nhiên cấu trúc tinh thể từ một trong hai vật liệu
            crystal_structure = np.random.choice(material1["crystal_structures"] + material2["crystal_structures"])
            
            # Thuộc tính là giá trị trung bình có trọng số ngẫu nhiên
            weight = np.random.uniform(0.2, 0.8)
            bandgap = material1["typical_bandgap"] * weight + material2["typical_bandgap"] * (1 - weight)
            conductivity = material1["typical_conductivity"] * weight + material2["typical_conductivity"] * (1 - weight)
            
            # Các thuộc tính khác
            is_direct = np.random.choice([0, 1])
            mobility = conductivity * np.random.uniform(0.8, 1.5)
            carrier_concentration = 10 ** np.random.uniform(15, 22)
            thermal_conductivity = np.random.uniform(0.5, 500)
            breakdown_field = (bandgap ** 2.5) * 0.1 * np.random.uniform(0.8, 1.2)
            absorption_coefficient = 10 ** np.random.uniform(3, 6)
            refractive_index = np.random.uniform(1.5, 4.5)
            
            # Thermal stability là giá trị tốt hơn giữa hai vật liệu
            thermal_stabilities = ["Low", "Medium", "High", "Very High"]
            stab1_index = thermal_stabilities.index(material1["thermal_stability"])
            stab2_index = thermal_stabilities.index(material2["thermal_stability"])
            thermal_stability = thermal_stabilities[max(stab1_index, stab2_index)]
            
            # Max temperature
            thermal_stability_map = {
                "Very High": np.random.uniform(800, 1200),
                "High": np.random.uniform(500, 800),
                "Medium": np.random.uniform(300, 500),
                "Low": np.random.uniform(100, 300)
            }
            max_temperature = thermal_stability_map[thermal_stability]
        
        # Xác định các ứng dụng tiềm năng dựa trên thuộc tính
        potential_applications = []
        
        for app, criteria in APPLICATION_CATEGORIES.items():
            matches_criteria = True
            
            if "ideal_bandgap" in criteria:
                min_bg, max_bg = criteria["ideal_bandgap"]
                if not (min_bg <= bandgap <= max_bg):
                    matches_criteria = False
                    
            if "ideal_conductivity" in criteria:
                min_cond, max_cond = criteria["ideal_conductivity"]
                if not (min_cond <= conductivity <= max_cond):
                    matches_criteria = False
                    
            if "absorption_coef_threshold" in criteria and absorption_coefficient < criteria["absorption_coef_threshold"]:
                matches_criteria = False
                
            if "thermal_stability" in criteria and thermal_stability not in criteria["thermal_stability"]:
                matches_criteria = False
                
            if "ideal_mobility" in criteria:
                min_mob, max_mob = criteria["ideal_mobility"]
                if not (min_mob <= mobility <= max_mob):
                    matches_criteria = False
                    
            if "breakdown_field_threshold" in criteria and breakdown_field < criteria["breakdown_field_threshold"]:
                matches_criteria = False
                
            if "direct_bandgap" in criteria and criteria["direct_bandgap"] != bool(is_direct):
                matches_criteria = False
                
            if matches_criteria:
                potential_applications.append(app)
        
        # Đảm bảo luôn có ít nhất một ứng dụng tiềm năng
        if not potential_applications:
            potential_applications = ["General semiconductor applications"]
        
        # Tạo material object
        material = {
            "material_id": material_id,
            "material_name": material_name,
            "formula": formula,
            "crystal_structure": crystal_structure,
            "bandgap_energy_eV": round(bandgap, 3),
            "is_direct_bandgap": bool(is_direct),
            "conductivity_S_cm": round(conductivity, 2),
            "mobility_cm2_Vs": round(mobility, 2),
            "carrier_concentration_cm3": format(carrier_concentration, ".2e"),
            "thermal_conductivity_W_mK": round(thermal_conductivity, 2),
            "breakdown_field_MV_cm": round(breakdown_field, 3),
            "absorption_coefficient_cm1": format(absorption_coefficient, ".2e"),
            "refractive_index": round(refractive_index, 2),
            "max_operating_temp_C": round(max_temperature, 1),
            "thermal_stability": thermal_stability,
            "potential_applications": ", ".join(potential_applications),
        }
        
        data.append(material)
    
    return pd.DataFrame(data)

# Hàm lưu file CSV
def save_to_csv(df, filename='data/raw/documents/fake_materials_dataset.csv'):
    """Lưu dữ liệu vào file CSV"""
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"✅ Đã lưu dữ liệu vào {filename}: {len(df)} bản ghi")
    
    # Lưu thông tin về vật liệu gốc để tham khảo
    material_info_path = os.path.join(os.path.dirname(filename), "material_reference.json")
    with open(material_info_path, 'w', encoding='utf-8') as f:
        json.dump(SEMICONDUCTOR_MATERIALS, f, indent=2, ensure_ascii=False)
    print(f"✅ Đã lưu thông tin tham khảo vật liệu vào {material_info_path}")

# Hiển thị thống kê và phân phối dữ liệu
def print_data_statistics(df):
    """In ra một số thống kê cơ bản về dữ liệu"""
    print("\n📊 Thống kê dữ liệu:")
    print(f"Tổng số mẫu: {len(df)}")
    
    # Đếm số lượng của mỗi vật liệu
    material_counts = df['material_name'].value_counts()
    print("\nPhân phối vật liệu:")
    for material, count in material_counts.items():
        print(f"  - {material}: {count} mẫu")
    
    # Đếm số lượng của mỗi biến thể
    if 'variation' in df.columns:
        variation_counts = df['variation'].value_counts()
        print("\nPhân phối biến thể:")
        for variation, count in variation_counts.items():
            print(f"  - {variation}: {count} mẫu")
    
    # Thống kê về ứng dụng tiềm năng
    app_counts = {}
    for apps in df['potential_applications']:
        for app in apps.split(', '):
            app_counts[app] = app_counts.get(app, 0) + 1
    
    print("\nPhân phối ứng dụng tiềm năng:")
    for app, count in sorted(app_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {app}: {count} mẫu")
    
    # Thống kê các thuộc tính số
    numeric_columns = ['bandgap_energy_eV', 'conductivity_S_cm', 'mobility_cm2_Vs']
    print("\nThống kê thuộc tính số:")
    for col in numeric_columns:
        if col in df.columns:
            print(f"  - {col}:")
            print(f"    Min: {df[col].min():.3f}, Max: {df[col].max():.3f}, Avg: {df[col].mean():.3f}")

# Thực thi
if __name__ == "__main__":
    # Số lượng mẫu cần tạo
    NUM_SAMPLES = 2000
    
    # Tạo dữ liệu
    print(f"🔍 Đang tạo {NUM_SAMPLES} mẫu vật liệu bán dẫn...")
    df = generate_fake_data(NUM_SAMPLES, include_variations=True)
    
    # Hiển thị thống kê
    print_data_statistics(df)
    
    # Lưu dữ liệu
    save_to_csv(df)
    
    print("\n✨ Hoàn thành tạo dữ liệu!")
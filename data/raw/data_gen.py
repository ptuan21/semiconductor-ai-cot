import pandas as pd
import numpy as np
import random
import os

# Danh sách vật liệu và crystal structure mẫu
materials = ['CuO', 'Si', 'Ge', 'GaAs', 'CuMnSnO', 'ZnO']
structures = ['Monoclinic', 'Cubic', 'Tetragonal', 'Hexagonal', 'Orthorhombic']
surface_morphologies = ['Smooth', 'Nanoparticles', 'Rough', 'Grainy', 'Porous']

# Hàm tạo dữ liệu giả
def generate_fake_data(num_samples):
    data = []
    
    for _ in range(num_samples):
        material = random.choice(materials)
        structure = random.choice(structures)
        surface_morphology = random.choice(surface_morphologies)
        
        bandgap_energy = round(np.random.uniform(1.0, 3.5), 2)  # eV
        conductivity = round(np.random.uniform(1e-4, 10), 4)  # S/cm
        absorption_coefficient = round(np.random.uniform(1e4, 1e6), 2) # Đơn vị: cm^-1

        # --- Định nghĩa lại cột Target dựa trên ứng dụng tiềm năng ---
        target_category = 'General Semiconductor' # Giá trị mặc định

        # Điều kiện cho TCO (Khe băng rộng, dẫn điện tốt)
        if bandgap_energy > 3.0 and conductivity > 1.0: # Ngưỡng có thể điều chỉnh
            target_category = 'Transparent Conductor (TCO)'
        # Điều kiện cho ứng dụng UV (Khe băng rất rộng)
        elif bandgap_energy > 3.0: # Không cần dẫn điện quá cao cho mọi ứng dụng UV
             target_category = 'UV Application Potential'
        # Điều kiện cho hấp thụ mặt trời (Khe băng phù hợp, dẫn điện khá, hấp thụ tốt)
        elif 1.0 <= bandgap_energy <= 1.7 and conductivity > 0.01 and absorption_coefficient > 5e4: # Ngưỡng có thể điều chỉnh
            target_category = 'Solar Absorber Potential'

        # --- Gán giá trị target mới ---
        target = target_category
        # -----------------------------------------------------------

        data.append({
            'material_name': material,
            'crystal_structure': structure,
            'surface_morphology': surface_morphology,
            'bandgap_energy (eV)': bandgap_energy,
            'conductivity (S/cm)': conductivity,
            'absorption_coefficient (cm^-1)': absorption_coefficient,
            'target_application_potential': target # Đổi tên cột để rõ ràng hơn
        })

    return pd.DataFrame(data)

# Hàm lưu file CSV
def save_to_csv(df, filename='documents/fake_materials_dataset.csv'):
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Thực thi
if __name__ == "__main__":
    # Tạo lại dữ liệu với 2000 bản ghi theo file CSV gốc
    # Hoặc giữ 3000 nếu bạn muốn
    df = generate_fake_data(2000)
    save_to_csv(df) # Lưu vào data/raw/documents/
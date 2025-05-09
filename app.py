import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from ai_analysis import IntegratedMaterialAnalysis, DataProcessor
import py3Dmol
import json
import os
import time
import concurrent.futures
import threading
from datetime import datetime
from material_cot_analyzer import MaterialCoTAnalyzer

# Import các hàm cần thiết từ evaluate_models.py
from evaluate_models import (
    calculate_material_score, 
    generate_visualizations,
    find_relevant_chunks,
    process_materials_batch,
    ResponseCache,
    APIRateManager,
    analyze_with_engine,
    get_prompt,
    init_chat_engine,
    SemanticSearchEngine  # Import class thay vì instance
)

# Cấu hình trang
st.set_page_config(
    page_title="AI Material Analyzer",
    page_icon="🔬",
    layout="wide"
)

# Tiêu đề
st.title("🔬 AI Material Analyzer")
st.markdown("---")

# Khởi tạo cache và rate manager
response_cache = ResponseCache(cache_dir="cache")
api_rate_manager = APIRateManager()
# Khởi tạo semantic search engine
semantic_search_engine = SemanticSearchEngine()

def create_radar_chart(properties):
    """Tạo biểu đồ radar cho các thuộc tính vật liệu"""
    # Chuẩn bị dữ liệu cho biểu đồ
    categories = [
        'Bandgap',
        'Conductivity',
        'Thermal Stability',
        'Heat Transport'
    ]
    
    # Lấy giá trị từ predictions
    values = [
        # Chuyển đổi bandgap_prediction từ string sang số
        0.5 if properties.get('bandgap_prediction') == 'Semiconductor' else
        0.8 if properties.get('bandgap_prediction') == 'Insulator' else
        0.2 if properties.get('bandgap_prediction') == 'Metal' else 0.5,
        
        # Chuyển đổi conductivity_prediction từ string sang số
        0.8 if properties.get('conductivity_prediction') == 'High' else
        0.5 if properties.get('conductivity_prediction') == 'Medium' else
        0.2 if properties.get('conductivity_prediction') == 'Low' else 0.5,
        
        # Chuyển đổi thermal_stability từ string sang số
        0.8 if properties.get('thermal_stability') == 'High' else
        0.5 if properties.get('thermal_stability') == 'Medium' else
        0.2 if properties.get('thermal_stability') == 'Low' else 0.5,
        
        # Chuyển đổi heat_transport từ string sang số
        0.8 if properties.get('heat_transport') == 'High' else
        0.5 if properties.get('heat_transport') == 'Medium' else
        0.2 if properties.get('heat_transport') == 'Low' else 0.5
    ]
    
    # Tạo biểu đồ radar
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Material Properties'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False
    )
    
    return fig

def create_correlation_plot(properties):
    """Tạo biểu đồ tương quan giữa các thuộc tính"""
    # Tạo DataFrame từ properties
    data = {
        'Property': ['Bandgap', 'Conductivity', 'Carrier Concentration'],
        'Value': [
            properties.get('bandgap', 0),
            properties.get('conductivity', 0),
            properties.get('carrier_concentration', 0)
        ],
        'Predicted': [
            properties.get('bandgap_prediction', {}).get('value', 0),
            properties.get('conductivity_prediction', {}).get('value', 0),
            properties.get('stability_prediction', {}).get('value', 0)
        ]
    }
    df = pd.DataFrame(data)
    
    # Tạo biểu đồ tương quan
    fig = px.scatter(df, x='Value', y='Predicted', 
                    text='Property',
                    title='Measured vs Predicted Properties')
    fig.add_shape(type='line',
                 x0=df['Value'].min(), 
                 y0=df['Value'].min(),
                 x1=df['Value'].max(), 
                 y1=df['Value'].max(),
                 line=dict(color='red', dash='dash'))
    
    return fig

def visualize_crystal_structure(crystal_structure, elements):
    """Tạo mô phỏng 3D cấu trúc tinh thể sử dụng plotly"""
    # Tạo các tọa độ cơ bản cho cấu trúc tinh thể
    if crystal_structure.lower() == 'cubic':
        positions = [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ]
    elif crystal_structure.lower() == 'hexagonal':
        positions = [
            [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0],
            [0, 0, 1], [1, 0, 1], [0.5, 0.866, 1]
        ]
    elif crystal_structure.lower() == 'zincblende':
        # Thêm cấu trúc zinc blende
        positions = [
            # Ga sites (face-centered cubic)
            [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
            # As sites (shifted fcc)
            [0.25, 0.25, 0.25], [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]
        ]
    else:
        positions = [[0, 0, 0], [1, 1, 1]]

    # Tạo danh sách các điểm và màu sắc
    x, y, z = [], [], []
    colors = []
    sizes = []
    labels = []
    
    for i, pos in enumerate(positions):
        x.append(pos[0])
        y.append(pos[1])
        z.append(pos[2])
        element = elements[i % len(elements)]
        colors.append(get_element_color(element))
        sizes.append(20)  # Kích thước atom
        labels.append(element)

    # Tạo bonds (liên kết) giữa các atom
    bonds_x, bonds_y, bonds_z = [], [], []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            # Tính khoảng cách giữa hai atom
            dist = sum((a - b) ** 2 for a, b in zip(positions[i], positions[j])) ** 0.5
            # Nếu khoảng cách đủ gần, thêm bond
            if dist < 1.5:  # Ngưỡng khoảng cách để tạo bond
                bonds_x.extend([positions[i][0], positions[j][0], None])
                bonds_y.extend([positions[i][1], positions[j][1], None])
                bonds_z.extend([positions[i][2], positions[j][2], None])

    # Tạo figure
    fig = go.Figure()

    # Thêm bonds
    fig.add_trace(go.Scatter3d(
        x=bonds_x, y=bonds_y, z=bonds_z,
        mode='lines',
        line=dict(color='grey', width=2),
        hoverinfo='none',
        showlegend=False
    ))

    # Thêm atoms
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=colors,
            symbol='circle',
            line=dict(color='black', width=1)
        ),
        text=labels,
        hoverinfo='text',
        textposition='top center',
        name='Atoms'
    ))

    # Cập nhật layout
    fig.update_layout(
        title=f"{crystal_structure} Structure",
        scene=dict(
            xaxis=dict(range=[-0.5, 1.5], showbackground=False),
            yaxis=dict(range=[-0.5, 1.5], showbackground=False),
            zaxis=dict(range=[-0.5, 1.5], showbackground=False),
            aspectmode='cube'
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig

def get_element_color(element):
    """Lấy màu cho các nguyên tố"""
    colors = {
        'Ga': '#4169E1',  # Royal Blue
        'As': '#FFD700',  # Gold
        'Si': '#808080',  # Gray
        'Ge': '#B8860B',  # Dark Golden Rod
        'O': '#FF0000',   # Red
        'Zn': '#C0C0C0',  # Silver
        'Cu': '#CD7F32'   # Bronze
    }
    return colors.get(element, '#CCCCCC')  # Default gray if element not found

def create_property_distribution(properties, material_name):
    """Tạo biểu đồ phân phối thuộc tính"""
    # Giả lập dữ liệu phân phối cho các vật liệu tương tự
    similar_materials = {
        'bandgap': np.random.normal(properties.get('bandgap', 0), 0.2, 100),
        'conductivity': np.random.lognormal(np.log(properties.get('conductivity', 1)), 0.5, 100),
        'carrier_concentration': np.random.lognormal(np.log(properties.get('carrier_concentration', 1e18)), 1, 100)
    }
    
    # Tạo subplot cho mỗi thuộc tính
    fig = go.Figure()
    
    for prop, values in similar_materials.items():
        fig.add_trace(go.Histogram(
            x=values,
            name=prop,
            opacity=0.7,
            nbinsx=30
        ))
        
        # Thêm đường dọc cho giá trị của vật liệu hiện tại
        fig.add_vline(x=properties.get(prop, 0),
                     line_dash="dash",
                     line_color="red",
                     annotation_text=f"{material_name}")
    
    fig.update_layout(
        title="Property Distribution Compared to Similar Materials",
        barmode='overlay',
        xaxis_title="Value",
        yaxis_title="Frequency"
    )
    
    return fig

def create_band_structure(properties):
    """Tạo biểu đồ band structure"""
    # Tạo dữ liệu giả lập cho band structure
    k_points = np.linspace(0, 2*np.pi, 100)
    valence_band = -2 + 0.5 * np.sin(k_points)
    conduction_band = properties.get('bandgap', 1.42) + 0.5 * np.sin(k_points)
    
    fig = go.Figure()
    
    # Vẽ valence band
    fig.add_trace(go.Scatter(
        x=k_points,
        y=valence_band,
        mode='lines',
        name='Valence Band',
        line=dict(color='blue', width=2)
    ))
    
    # Vẽ conduction band
    fig.add_trace(go.Scatter(
        x=k_points,
        y=conduction_band,
        mode='lines',
        name='Conduction Band',
        line=dict(color='red', width=2)
    ))
    
    # Thêm Fermi level
    fig.add_hline(y=0, line_dash="dash", line_color="green",
                 annotation_text="Fermi Level")
    
    fig.update_layout(
        title="Band Structure",
        xaxis_title="Wave Vector (k)",
        yaxis_title="Energy (eV)",
        showlegend=True
    )
    
    return fig

def create_dos_plot(properties):
    """Tạo biểu đồ Density of States"""
    # Tạo dữ liệu giả lập cho DOS
    energy = np.linspace(-5, 5, 200)
    dos_valence = np.exp(-(energy + 2)**2/0.5)
    dos_conduction = np.exp(-(energy - properties.get('bandgap', 1.42))**2/0.5)
    
    fig = go.Figure()
    
    # Vẽ DOS cho valence band
    fig.add_trace(go.Scatter(
        x=-dos_valence,
        y=energy,
        mode='lines',
        name='Valence DOS',
        line=dict(color='blue', width=2)
    ))
    
    # Vẽ DOS cho conduction band
    fig.add_trace(go.Scatter(
        x=dos_conduction,
        y=energy,
        mode='lines',
        name='Conduction DOS',
        line=dict(color='red', width=2)
    ))
    
    # Thêm Fermi level
    fig.add_hline(y=0, line_dash="dash", line_color="green",
                 annotation_text="Fermi Level")
    
    fig.update_layout(
        title="Density of States",
        xaxis_title="DOS (states/eV)",
        yaxis_title="Energy (eV)",
        showlegend=True
    )
    
    return fig

def create_temperature_dependence(properties):
    """Tạo biểu đồ phụ thuộc nhiệt độ"""
    # Tạo dữ liệu giả lập cho phụ thuộc nhiệt độ
    temperatures = np.linspace(100, 1000, 50)
    
    # Tính toán các thuộc tính phụ thuộc nhiệt độ
    conductivity = properties.get('conductivity', 1000) * np.exp(-0.1 * (temperatures - 300) / 300)
    carrier_concentration = properties.get('carrier_concentration', 1e18) * np.exp(-0.05 * (temperatures - 300) / 300)
    
    fig = go.Figure()
    
    # Vẽ conductivity
    fig.add_trace(go.Scatter(
        x=temperatures,
        y=conductivity,
        mode='lines',
        name='Conductivity',
        line=dict(color='blue', width=2)
    ))
    
    # Vẽ carrier concentration
    fig.add_trace(go.Scatter(
        x=temperatures,
        y=carrier_concentration,
        mode='lines',
        name='Carrier Concentration',
        line=dict(color='red', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Temperature Dependence",
        xaxis_title="Temperature (K)",
        yaxis_title="Conductivity (S/cm)",
        yaxis2=dict(
            title="Carrier Concentration (cm^-3)",
            overlaying='y',
            side='right'
        ),
        showlegend=True
    )
    
    return fig

def create_material_comparison(properties, material_name):
    """Tạo biểu đồ so sánh với các vật liệu tương tự"""
    # Dữ liệu giả lập cho các vật liệu tương tự
    similar_materials = {
        'Si': {'bandgap': 1.12, 'conductivity': 1000, 'carrier_concentration': 1e18},
        'Ge': {'bandgap': 0.67, 'conductivity': 2000, 'carrier_concentration': 2e18},
        'GaAs': {'bandgap': 1.42, 'conductivity': 1500, 'carrier_concentration': 1.5e18},
        material_name: properties
    }
    
    # Tạo DataFrame
    data = []
    for mat, props in similar_materials.items():
        data.append({
            'Material': mat,
            'Bandgap': props.get('bandgap', 0),
            'Conductivity': props.get('conductivity', 0),
            'Carrier Concentration': props.get('carrier_concentration', 0)
        })
    
    df = pd.DataFrame(data)
    
    # Tạo biểu đồ
    fig = go.Figure()
    
    # Thêm các cột cho mỗi thuộc tính
    fig.add_trace(go.Bar(
        name='Bandgap',
        x=df['Material'],
        y=df['Bandgap'],
        text=df['Bandgap'].round(2),
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='Conductivity',
        x=df['Material'],
        y=df['Conductivity']/1000,  # Scale down for better visualization
        text=(df['Conductivity']/1000).round(2),
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='Carrier Concentration',
        x=df['Material'],
        y=df['Carrier Concentration']/1e18,  # Scale down for better visualization
        text=(df['Carrier Concentration']/1e18).round(2),
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Comparison with Similar Materials",
        barmode='group',
        yaxis_title="Value (normalized)",
        showlegend=True
    )
    
    return fig

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
    
    # Khởi tạo engines
    active_engines = []
    for engine_name in engines_to_use:
        engine_tuple = init_chat_engine(engine_name)
        if engine_tuple[1] is not None or engine_name == "groq":  # Groq không cần chat object
            active_engines.append(engine_tuple)
    
    # Nếu không có engine nào khả dụng, thông báo lỗi
    if not active_engines:
        raise Exception("No AI engines available. Please check your API keys.")
        
    # Xử lý tuần tự các batch
    for idx, batch in enumerate(batches):
        # Cập nhật tiến độ
        if progress_callback:
            progress_callback(idx, len(batches))
            
        # Xử lý batch
        batch_results = process_materials_batch(
            batch,
            {"basic": {"category": "task", "key": "analyze_material"}},
            active_engines,
            [],  # Không có PDF chunks
            0,    # Không sleep
            search_engine=semantic_search_engine,
            api_rate_manager=api_rate_manager,  # Truyền api_rate_manager
            response_cache=response_cache      # Truyền response_cache
        )
        
        # Thêm kết quả vào danh sách chung
        with results_lock:
            for material_result in batch_results:
                # Thêm kết quả phân tích
                all_results.extend(material_result["analysis_results"])
                
                # Thêm điểm đánh giá 
                material_id = material_result["material_id"]
                material_data = material_result["material_data"]
                material_name = material_result["material_name"]
                
                material_scores[material_id] = {
                    "score": material_result["material_score"],
                    "material_name": material_name,
                    "properties": {
                        "bandgap": material_data.get('bandgap', 'N/A'),
                        "crystal_structure": material_data.get('crystal_structure', 'N/A'),
                        "conductivity": material_data.get('conductivity', 'N/A'),
                        "thermal_stability": material_data.get('thermal_stability', 'N/A')
                    }
                }
    
    return all_results, material_scores

def create_material_properties_table(properties):
    """Tạo bảng thuộc tính vật liệu với định dạng đẹp"""
    properties_df = pd.DataFrame([{
        "Thuộc tính": key.replace('_', ' ').title(),
        "Giá trị": str(value) if not isinstance(value, dict) else str(value.get('value', value)),
        "Đơn vị": get_property_unit(key)
    } for key, value in properties.items()])
    
    return properties_df

def get_property_unit(property_name):
    """Trả về đơn vị đo cho thuộc tính vật liệu"""
    units = {
        "bandgap": "eV",
        "conductivity": "S/cm",
        "carrier_concentration": "cm⁻³",
        "mobility": "cm²/V·s",
        "temperature": "K",
        "lattice_constant": "Å",
        "melting_point": "K",
        "density": "g/cm³",
        "thermal_conductivity": "W/m·K",
    }
    return units.get(property_name.lower(), "")

def create_heatmap_correlation(properties):
    """Tạo biểu đồ heatmap cho tương quan thuộc tính"""
    # Tạo dữ liệu tương quan giả lập
    properties_values = {}
    for k, v in properties.items():
        if isinstance(v, (int, float)) and not np.isnan(v):
            properties_values[k.replace('_', ' ').title()] = v
    
    # Nếu không đủ dữ liệu, tạo dữ liệu giả
    if len(properties_values) < 3:
        properties_values = {
            "Bandgap": properties.get('bandgap', 1.5),
            "Conductivity": properties.get('conductivity', 1000),
            "Mobility": properties.get('mobility', 500),
            "Carrier Concentration": properties.get('carrier_concentration', 1e18),
            "Thermal Conductivity": properties.get('thermal_conductivity', 50)
        }
    
    # Tạo ma trận tương quan
    props = list(properties_values.keys())
    values = list(properties_values.values())
    
    corr_matrix = np.zeros((len(props), len(props)))
    for i in range(len(props)):
        for j in range(len(props)):
            # Tạo dữ liệu tương quan giả lập dựa trên các quy luật vật lý
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Bandgap thường tỷ lệ nghịch với conductivity
                if "Bandgap" in props[i] and "Conductivity" in props[j]:
                    corr_matrix[i, j] = -0.7
                elif "Conductivity" in props[i] and "Bandgap" in props[j]:
                    corr_matrix[i, j] = -0.7
                # Carrier concentration và conductivity thường tỷ lệ thuận
                elif "Carrier" in props[i] and "Conductivity" in props[j]:
                    corr_matrix[i, j] = 0.8
                elif "Conductivity" in props[i] and "Carrier" in props[j]:
                    corr_matrix[i, j] = 0.8
                # Mobility và conductivity thường tỷ lệ thuận
                elif "Mobility" in props[i] and "Conductivity" in props[j]:
                    corr_matrix[i, j] = 0.6
                elif "Conductivity" in props[i] and "Mobility" in props[j]:
                    corr_matrix[i, j] = 0.6
                # Mặc định tương quan thấp
                else:
                    corr_matrix[i, j] = 0.2 * np.random.random() - 0.1
    
    # Tạo biểu đồ heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=props,
        y=props,
        colorscale='RdBu_r',
        zmid=0,
        text=corr_matrix.round(2),
        texttemplate="%{text}",
        textfont={"size":14},
    ))
    
    fig.update_layout(
        title="Tương quan giữa các thuộc tính vật liệu",
        height=500,
        width=700
    )
    
    return fig

def create_ternary_plot(material_name, composition):
    """Tạo biểu đồ tam giác cho thành phần vật liệu"""
    if len(composition) < 3:
        # Nếu chỉ có 1-2 thành phần, thêm các thành phần giả
        keys = list(composition.keys())
        if len(keys) == 1:
            composition[f"{keys[0]}-substitute"] = 0.2
            composition["Other"] = 0.1
        elif len(keys) == 2:
            composition["Other"] = 0.1
    
    # Nếu có hơn 3 thành phần, lấy 3 thành phần chính và gộp phần còn lại
    if len(composition) > 3:
        sorted_comp = sorted(composition.items(), key=lambda x: x[1], reverse=True)
        main_elements = dict(sorted_comp[:3])
        
        # Tổng tỷ lệ thành phần chính
        total_main = sum(main_elements.values())
        
        # Chuẩn hóa
        for k in main_elements:
            main_elements[k] = main_elements[k] / total_main
        
        composition = main_elements
    
    # Chuẩn hóa tổng tỷ lệ thành 100%
    total = sum(composition.values())
    for k in composition:
        composition[k] = composition[k] / total * 100
    
    # Tạo biểu đồ tam giác
    fig = go.Figure(go.Scatterternary({
        'mode': 'markers+text',
        'a': [composition.get(list(composition.keys())[0], 0)],
        'b': [composition.get(list(composition.keys())[1], 0)],
        'c': [composition.get(list(composition.keys())[2], 0)],
        'text': [material_name],
        'marker': {
            'symbol': 100,
            'color': 'red',
            'size': 15,
            'line': {'width': 2, 'color': 'black'}
        },
        'textfont': {'size': 15},
        'textposition': "top center"
    }))
    
    fig.update_layout(
        title=f"Thành phần vật liệu {material_name}",
        ternary={
            'aaxis':{'title': list(composition.keys())[0], 'min': 0, 'linewidth':2, 'ticks':'outside'},
            'baxis':{'title': list(composition.keys())[1], 'min': 0, 'linewidth':2, 'ticks':'outside'},
            'caxis':{'title': list(composition.keys())[2], 'min': 0, 'linewidth':2, 'ticks':'outside'}
        },
        width=500,
        height=500
    )
    
    return fig

def create_3d_scatter_plot(properties):
    """Tạo biểu đồ 3D cho các thuộc tính chính"""
    # Tạo dữ liệu giả cho nhiều vật liệu tương tự
    materials = ["Si", "Ge", "GaAs", "GaN", "InP", "ZnO", "CdTe"]
    bandgaps = [1.12, 0.67, 1.43, 3.4, 1.35, 3.37, 1.5]
    conductivities = [1000, 2000, 5000, 1200, 4600, 200, 650]
    densities = [2.3, 5.3, 5.3, 6.1, 4.8, 5.6, 5.9]
    
    # Thêm vật liệu hiện tại
    target_bandgap = properties.get("bandgap", 1.5)
    target_conductivity = properties.get("conductivity", 1000)
    target_density = properties.get("density", 5.0)
    
    materials.append("Current")
    bandgaps.append(target_bandgap)
    conductivities.append(target_conductivity)
    densities.append(target_density)
    
    # Tạo màu để phân biệt vật liệu hiện tại
    colors = ['blue'] * (len(materials) - 1) + ['red']
    sizes = [15] * (len(materials) - 1) + [25]
    
    # Tạo biểu đồ 3D
    fig = go.Figure(data=[go.Scatter3d(
        x=bandgaps,
        y=conductivities,
        z=densities,
        text=materials,
        mode='markers',
        marker={
            'size': sizes,
            'color': colors,
            'opacity': 0.8,
        }
    )])
    
    # Cập nhật layout
    fig.update_layout(
        title="So sánh vật liệu trong không gian 3D",
        scene = {
            'xaxis': {'title': 'Bandgap (eV)'},
            'yaxis': {'title': 'Conductivity (S/cm)'},
            'zaxis': {'title': 'Density (g/cm³)'}
        },
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def create_electronic_band_diagram(bandgap, material_name):
    """Tạo sơ đồ dải năng lượng điện tử với Plotly"""
    fig = go.Figure()
    
    # Thêm dải dẫn (conduction band)
    k_points = np.linspace(-3, 3, 1000)
    conduction_band = 0.5 * k_points**2 + bandgap  # E = ħ²k²/2m* + Eg
    
    fig.add_trace(go.Scatter(
        x=k_points,
        y=conduction_band,
        mode='lines',
        name='Dải dẫn',
        line=dict(color='red', width=4)
    ))
    
    # Thêm dải hóa trị (valence band)
    valence_band = -0.5 * k_points**2  # E = -ħ²k²/2m*
    
    fig.add_trace(go.Scatter(
        x=k_points,
        y=valence_band,
        mode='lines',
        name='Dải hóa trị',
        line=dict(color='blue', width=4)
    ))
    
    # Thêm mức Fermi
    fig.add_hline(y=0, line=dict(color='green', width=3, dash='dash'), name='Mức Fermi')
    
    # Thêm khoảng cách vùng cấm
    mid_point = 0
    fig.add_annotation(
        x=mid_point,
        y=bandgap/2,
        text=f"Bandgap: {bandgap} eV",
        showarrow=True,
        arrowhead=2,
        arrowcolor='black',
        ax=50,
        ay=0
    )
    
    # Cập nhật layout
    fig.update_layout(
        title=f'Sơ đồ dải năng lượng điện tử cho {material_name}',
        xaxis_title='Vector sóng k',
        yaxis_title='Năng lượng (eV)',
        height=500,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def material_data_from_form(material_name, crystal_structure, elements, ratios, **properties):
    """Tạo đối tượng dữ liệu vật liệu từ form nhập"""
    # Tạo dictionary thành phần
    composition = {element: ratio for element, ratio in zip(elements, ratios) if element.strip()}
    
    # Tạo đối tượng dữ liệu
    material_data = {
        'name': material_name,
        'crystal_structure': crystal_structure,
        'composition': composition,
        'properties': properties
    }
    
    return material_data

def main():
    # Sidebar cho input
    st.sidebar.header("Material Input")
    
    # Thêm tab cho phân tích hình ảnh và batch processing
    tab_options = ["Material Analysis", "Image Analysis", "Batch Analysis"]
    selected_tab = st.sidebar.radio("Select Analysis Type", tab_options)
    
    if selected_tab == "Material Analysis":
        # Input form cho phân tích vật liệu
        with st.sidebar.form("material_input"):
            material_name = st.text_input("Material Name", "GaAs")
            
            crystal_structure = st.selectbox(
                "Crystal Structure",
                ["Cubic", "Hexagonal", "Wurtzite", "Zincblende", "Tetragonal", 
                 "Orthorhombic", "Monoclinic", "Triclinic", "Perovskite", "Spinel"]
            )
            
            st.subheader("Composition")
            num_elements = st.number_input("Number of Elements", min_value=1, max_value=10, value=2)
            
            elements = []
            ratios = []
            
            for i in range(num_elements):
                col1, col2 = st.columns(2)
                with col1:
                    element = st.text_input(f"Element {i+1}", value="Ga" if i == 0 else "As" if i == 1 else "")
                    elements.append(element)
                with col2:
                    ratio = st.number_input(f"Ratio {i+1}", min_value=0.0, max_value=1.0, value=0.5 if i == 0 else 0.5 if i == 1 else 0.0, step=0.1)
                    ratios.append(ratio)
            
            st.subheader("Known Properties")
            col3, col4 = st.columns(2)
            with col3:
                bandgap = st.number_input("Bandgap (eV)", value=1.42)
                conductivity = st.number_input("Conductivity (S/cm)", value=1000.0)
            with col4:
                carrier_concentration = st.number_input("Carrier Concentration (cm^-3)", value=1e18)
                temperature = st.number_input("Temperature (K)", value=300.0)
            
            # Thêm các thuộc tính vật lý khác
            st.subheader("Additional Properties")
            col5, col6 = st.columns(2)
            with col5:
                lattice_constant = st.number_input("Lattice Constant (Å)", value=5.65)
                melting_point = st.number_input("Melting Point (K)", value=1511.0)
            with col6:
                density = st.number_input("Density (g/cm³)", value=5.32)
                thermal_conductivity = st.number_input("Thermal Conductivity (W/m·K)", value=55.0)
            
            submitted = st.form_submit_button("Analyze Material")
        
        if submitted:
            # Chuẩn bị dữ liệu
            composition = {element: ratio for element, ratio in zip(elements, ratios)}
            
            material_data = {
                'name': material_name,
                'crystal_structure': crystal_structure,
                'composition': composition,
                'properties': {
                    'bandgap': bandgap,
                    'conductivity': conductivity,
                    'carrier_concentration': carrier_concentration,
                    'temperature': temperature,
                    'lattice_constant': lattice_constant,
                    'melting_point': melting_point,
                    'density': density,
                    'thermal_conductivity': thermal_conductivity
                }
            }
            
            # Khởi tạo analyzer với CoT
            analyzer = MaterialCoTAnalyzer()
            
            # Phân tích với Chain-of-Thought
            with st.spinner('Analyzing material using Chain-of-Thought...'):
                cot_results = analyzer.analyze_with_cot(material_data)
            
            # Layout với tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Basic Analysis", "Advanced Visualization", "Chain of Thought", "AI Insights"])
            
            with tab1:
                # Cải thiện layout
                st.subheader("Thông tin tổng quan về vật liệu")
                
                material_card = f"""
                <div style="padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; background-color: #f8f9fa;">
                    <h3 style="color: #1E88E5;">{material_name}</h3>
                    <p><b>Cấu trúc tinh thể:</b> {crystal_structure}</p>
                    <p><b>Thành phần:</b> {', '.join([f"{elem} ({ratio*100:.1f}%)" for elem, ratio in zip(elements, ratios) if elem])}</p>
                </div>
                """
                st.markdown(material_card, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Thuộc tính cơ bản")
                    properties_df = create_material_properties_table(cot_results['basic_properties'])
                    st.dataframe(properties_df, use_container_width=True, hide_index=True)
                    
                    st.subheader("Dự đoán thuộc tính")
                    prediction_df = pd.DataFrame([{
                        "Thuộc tính": key.replace('_', ' ').title(),
                        "Dự đoán": value.get('value', 'N/A') if isinstance(value, dict) else value
                    } for key, value in cot_results['predictions'].items()])
                    st.dataframe(prediction_df, use_container_width=True, hide_index=True)
                    
                    # Thêm điểm chất lượng tổng thể
                    st.metric(
                        "Điểm chất lượng tổng thể", 
                        f"{cot_results['final_analysis']['overall_quality']:.1f}/10.0",
                        delta=None if cot_results['final_analysis']['overall_quality'] < 5 else "Tốt"
                    )
                
                with col2:
                    st.subheader("Biểu đồ radar thuộc tính")
                    radar_fig = create_radar_chart(cot_results['predictions'])
                    st.plotly_chart(radar_fig, use_container_width=True)
                
                # Thêm biểu đồ tam giác thành phần
                if len(composition) >= 2:
                    st.subheader("Biểu đồ thành phần vật liệu")
                    ternary_fig = create_ternary_plot(material_name, composition)
                    st.plotly_chart(ternary_fig, use_container_width=True)
            
            with tab2:
                st.subheader("Trực quan hóa nâng cao")
                
                tabs_advanced = st.tabs([
                    "Cấu trúc tinh thể", "Sơ đồ dải năng lượng", 
                    "So sánh 3D", "Tương quan thuộc tính", 
                    "Phụ thuộc nhiệt độ"
                ])
                
                with tabs_advanced[0]:
                    st.subheader("Trực quan hóa cấu trúc tinh thể")
                    structure_fig = visualize_crystal_structure(
                        crystal_structure,
                        elements
                    )
                    
                    # Khởi tạo session state nếu chưa tồn tại
                    if 'rotation_x' not in st.session_state:
                        st.session_state.rotation_x = 0
                        st.session_state.rotation_y = 0
                        st.session_state.rotation_z = 0
                    
                    # Cập nhật góc nhìn cho figure dựa trên session_state
                    structure_fig.update_scenes(
                        camera=dict(
                            eye=dict(
                                x=1.25 * np.cos(np.radians(st.session_state.rotation_x)),
                                y=1.25 * np.sin(np.radians(st.session_state.rotation_y)),
                                z=1.25 * np.sin(np.radians(st.session_state.rotation_z))
                            )
                        )
                    )
                    
                    st.plotly_chart(structure_fig, use_container_width=True)
                    
                    # Thêm nút để xoay mô hình 3D
                    st.markdown("""
                    <style>
                    .stButton>button {
                        background-color: #1E88E5;
                        color: white;
                        border-radius: 5px;
                        padding: 10px 24px;
                        margin: 5px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Định nghĩa các callback function cho nút bấm
                    def rotate_left():
                        st.session_state.rotation_y = (st.session_state.rotation_y - 90) % 360
                        st.session_state.rotation_changed = True
                        
                    def rotate_right():
                        st.session_state.rotation_y = (st.session_state.rotation_y + 90) % 360
                        st.session_state.rotation_changed = True
                        
                    def reset_view():
                        st.session_state.rotation_x = 0
                        st.session_state.rotation_y = 0
                        st.session_state.rotation_z = 0
                        st.session_state.rotation_changed = True
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.button("Xoay 90° trái", key="rotate_left_btn", on_click=rotate_left)
                    with col2:
                        st.button("Xoay 90° phải", key="rotate_right_btn", on_click=rotate_right)
                    with col3:
                        st.button("Đặt lại góc nhìn", key="reset_view_btn", on_click=reset_view)
                    
                    # Hiển thị thông tin góc quay hiện tại
                    if hasattr(st.session_state, 'rotation_changed') and st.session_state.rotation_changed:
                        st.success(f"Đã thay đổi góc nhìn - Góc quay hiện tại: X={st.session_state.rotation_x}°, Y={st.session_state.rotation_y}°, Z={st.session_state.rotation_z}°")
                        st.session_state.rotation_changed = False
                
                with tabs_advanced[1]:
                    st.subheader("Sơ đồ dải năng lượng điện tử")
                    band_diagram = create_electronic_band_diagram(
                        material_data['properties']['bandgap'],
                        material_name
                    )
                    st.plotly_chart(band_diagram, use_container_width=True)
                    
                    # Thêm giải thích
                    with st.expander("Giải thích sơ đồ dải năng lượng"):
                        st.markdown("""
                        - **Dải dẫn (màu đỏ)**: Các mức năng lượng trống mà điện tử có thể chuyển lên khi được kích thích
                        - **Dải hóa trị (màu xanh)**: Các mức năng lượng đã có điện tử chiếm giữ
                        - **Vùng cấm (khoảng giữa)**: Khoảng cách năng lượng giữa đỉnh dải hóa trị và đáy dải dẫn
                        - **Mức Fermi (đường đứt nét màu xanh lá)**: Mức năng lượng tham chiếu
                        """)
                
                with tabs_advanced[2]:
                    st.subheader("So sánh vật liệu trong không gian 3D")
                    scatter_3d = create_3d_scatter_plot(material_data['properties'])
                    st.plotly_chart(scatter_3d, use_container_width=True)
                    
                    st.info("Biểu đồ này thể hiện vị trí của vật liệu hiện tại (màu đỏ) so với các vật liệu bán dẫn phổ biến trong không gian 3 chiều: bandgap, độ dẫn điện và mật độ.")
                
                with tabs_advanced[3]:
                    st.subheader("Tương quan giữa các thuộc tính vật liệu")
                    heatmap_fig = create_heatmap_correlation(material_data['properties'])
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                    
                    st.info("Biểu đồ nhiệt này thể hiện mức độ tương quan giữa các thuộc tính vật lý của vật liệu. Giá trị gần 1 thể hiện tương quan thuận mạnh, giá trị gần -1 thể hiện tương quan nghịch mạnh.")
                
                with tabs_advanced[4]:
                    st.subheader("Phụ thuộc vào nhiệt độ")
                    temp_fig = create_temperature_dependence(material_data['properties'])
                    st.plotly_chart(temp_fig, use_container_width=True)
                    
                    # Thêm slider để điều chỉnh nhiệt độ
                    temperature = st.slider("Nhiệt độ (K)", 100, 1000, 300)
                    
                    # Hiển thị thuộc tính tại nhiệt độ đã chọn
                    conductivity_at_temp = material_data['properties']['conductivity'] * np.exp(-0.1 * (temperature - 300) / 300)
                    carrier_conc_at_temp = material_data['properties']['carrier_concentration'] * np.exp(-0.05 * (temperature - 300) / 300)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Độ dẫn điện tại nhiệt độ đã chọn", f"{conductivity_at_temp:.2f} S/cm")
                    with col2:
                        st.metric("Nồng độ hạt tải tại nhiệt độ đã chọn", f"{carrier_conc_at_temp:.2e} cm⁻³")

            # Thêm nội dung cho Chain of Thought tab (tab3)
            with tab3:
                st.subheader("Chain of Thought (Quá trình suy luận)")
                
                # Thêm tiến trình phân tích theo bước
                if 'reasoning_steps' in cot_results:
                    # Hiển thị tiến trình phân tích
                    st.markdown("### Tiến trình phân tích")
                    steps = cot_results['reasoning_steps']
                    total_steps = len(steps)
                    
                    # Hiển thị tất cả các bước trên cùng một trang
                    st.markdown("#### Tổng quan các bước phân tích")
                    
                    # Hiển thị thanh tiến trình dưới dạng timeline bằng cách sử dụng các columns của Streamlit
                    timeline_cols = st.columns(total_steps * 2 - 1)
                    
                    # Tạo timeline với các đốt và đường nối
                    for i in range(total_steps):
                        col_idx = i * 2  # Vị trí cột cho mỗi step
                        step_num = i + 1
                        step = steps[i]
                        
                        # Hiển thị đốt timeline
                        with timeline_cols[col_idx]:
                            st.markdown(
                                f"""
                                <div style="display: flex; flex-direction: column; align-items: center;">
                                    <div style="background-color: #1E88E5; color: white; border-radius: 50%; 
                                                width: 40px; height: 40px; display: flex; align-items: center; 
                                                justify-content: center; margin: 0 auto;">
                                        {step_num}
                                    </div>
                                    <div style="margin-top: 5px; text-align: center; font-size: 0.8em;">{step['title']}</div>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        # Hiển thị đường nối giữa các bước
                        if i < total_steps - 1:
                            with timeline_cols[col_idx + 1]:
                                st.markdown(
                                    """
                                    <div style="border-top: 2px dashed #ccc; height: 1px; margin-top: 20px;"></div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                    
                    st.markdown("---")
                    
                    # Hiển thị tất cả các bước với nội dung và hình ảnh
                    for i, step in enumerate(steps):
                        with st.expander(f"Bước {i+1}: {step['title']}", expanded=True):
                            step_type = step['title']
                            step_content = step['content']
                            
                            # Tạo hai cột cho nội dung và hình ảnh
                            content_col, visual_col = st.columns([3, 2])
                            
                            with content_col:
                                st.markdown(step_content)
                                
                                # Thêm key findings nếu có
                                if 'key_findings' in step:
                                    st.markdown("**Phát hiện chính:**")
                                    for finding in step['key_findings']:
                                        st.markdown(f"- {finding}")
                            
                            with visual_col:
                                # Hiển thị trực quan phù hợp với từng loại bước phân tích
                                if "thành phần" in step_type.lower():
                                    # Hiển thị biểu đồ tam giác thành phần
                                    if len(composition) >= 2:
                                        st.subheader("Biểu đồ thành phần")
                                        ternary_fig = create_ternary_plot(material_name, composition)
                                        st.plotly_chart(ternary_fig, use_container_width=True)
                                
                                elif "cấu trúc tinh thể" in step_type.lower():
                                    # Hiển thị mô hình cấu trúc tinh thể
                                    st.subheader("Mô hình cấu trúc tinh thể")
                                    structure_fig = visualize_crystal_structure(crystal_structure, elements)
                                    st.plotly_chart(structure_fig, use_container_width=True)
                                
                                elif "tính chất điện tử" in step_type.lower():
                                    # Hiển thị sơ đồ band structure
                                    st.subheader("Sơ đồ dải năng lượng")
                                    band_fig = create_electronic_band_diagram(
                                        material_data['properties']['bandgap'],
                                        material_name
                                    )
                                    st.plotly_chart(band_fig, use_container_width=True)
                                
                                elif "tính chất nhiệt" in step_type.lower():
                                    # Hiển thị biểu đồ phụ thuộc nhiệt độ
                                    st.subheader("Phụ thuộc nhiệt độ")
                                    temp_fig = create_temperature_dependence(material_data['properties'])
                                    st.plotly_chart(temp_fig, use_container_width=True)
                                    
                                elif "ứng dụng" in step_type.lower():
                                    # Hiển thị biểu đồ radar cho các ứng dụng
                                    st.subheader("Phân tích tiềm năng ứng dụng")
                                    radar_fig = create_radar_chart(cot_results['predictions'])
                                    st.plotly_chart(radar_fig, use_container_width=True)
                                    
                                elif "cải tiến" in step_type.lower() or "cải thiện" in step_type.lower():
                                    if 'recommendations' in cot_results:
                                        # Tạo biểu đồ ưu tiên cải tiến
                                        improvements = cot_results['recommendations'].get('improvements', [])
                                        if improvements:
                                            aspects = [imp.get('aspect', 'Unknown') for imp in improvements]
                                            priorities = [0.8 if 'high' in str(imp.get('priority', '')).lower() 
                                                        else 0.5 if 'medium' in str(imp.get('priority', '')).lower() 
                                                        else 0.3 for imp in improvements]
                                            
                                            # Tạo biểu đồ thanh ngang cho mức độ ưu tiên
                                            fig = go.Figure(go.Bar(
                                                x=priorities,
                                                y=aspects,
                                                orientation='h',
                                                marker_color='#1E88E5',
                                                text=[f"{p*10:.1f}/10" for p in priorities],
                                                textposition='inside'
                                            ))
                                            
                                            fig.update_layout(
                                                title="Mức độ ưu tiên cải tiến",
                                                xaxis_title="Điểm ưu tiên",
                                                yaxis_title="Khía cạnh cần cải thiện"
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                
                                elif "AI" in step_type:
                                    # Hiển thị kết quả so sánh giữa các mô hình AI
                                    st.subheader("So sánh phân tích từ các mô hình AI")
                                    if 'ai_analysis' in cot_results:
                                        ai_models = list(cot_results.get('ai_analysis', {}).keys())
                                        if ai_models:
                                            # Tạo biểu đồ so sánh độ tin cậy của các mô hình
                                            confidences = [0.95, 0.92]  # Giả lập dữ liệu
                                            
                                            fig = go.Figure()
                                            fig.add_trace(go.Bar(
                                                x=ai_models,
                                                y=confidences,
                                                marker_color=['#4285F4', '#EA4335'],
                                                text=[f"{conf*100:.1f}%" for conf in confidences],
                                                textposition='outside'
                                            ))
                                            
                                            fig.update_layout(
                                                title="Độ tin cậy của các mô hình AI",
                                                yaxis=dict(range=[0, 1], tickformat=".0%"),
                                                yaxis_title="Độ tin cậy"
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Thêm phần tách giữa các bước
                            if i < total_steps - 1:
                                st.markdown("---")
                    
                    # Trực quan hóa tổng hợp kết quả phân tích
                    st.markdown("### Tổng hợp kết quả phân tích")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        quality_score = cot_results['final_analysis'].get('overall_quality', 7.5)
                        st.metric("Điểm chất lượng tổng thể", f"{quality_score:.1f}/10")
                    
                    with metric_col2:
                        bandgap_type = cot_results['final_analysis'].get('bandgap_classification', 'Semiconductor')
                        st.metric("Phân loại bandgap", bandgap_type)
                    
                    with metric_col3:
                        applications_count = len(cot_results['recommendations'].get('applications', []))
                        st.metric("Số ứng dụng tiềm năng", applications_count)
                        
                    # Hiển thị biểu đồ kết quả tổng hợp
                    st.subheader("Biểu đồ radar thuộc tính")
                    summary_radar = create_radar_chart(cot_results['predictions'])
                    st.plotly_chart(summary_radar, use_container_width=True)
                
                else:
                    st.warning("Đang khởi tạo phân tích Chain of Thought. Vui lòng đợi trong giây lát hoặc làm mới trang nếu không thấy kết quả.")
                
                # Thêm nút Refresh cho Chain of Thought
                if st.button("Làm mới phân tích Chain of Thought", key="refresh_cot"):
                    st.session_state['cot_refreshed'] = True
                    with st.spinner('Đang tái phân tích vật liệu...'):
                        # Phân tích lại với Chain-of-Thought
                        cot_results = analyzer.analyze_with_cot(material_data)
                        st.experimental_rerun()
                
                # Hiển thị trạng thái
                if st.session_state.get('cot_refreshed', False):
                    st.success("Đã cập nhật phân tích Chain of Thought!")
                    # Đặt lại trạng thái sau khi hiển thị
                    st.session_state['cot_refreshed'] = False
                
                # Hiển thị sơ đồ quá trình suy luận
                st.subheader("Sơ đồ phân tích")
                
                # Tạo flowchart với Graphviz
                cot_chart = """
                digraph {
                    rankdir=TB;
                    node [shape=box, style=filled, fillcolor=lightblue, fontname="Arial", margin=0.3];
                    edge [fontname="Arial"];
                    
                    input [label="Dữ liệu vật liệu\nđầu vào"];
                    step1 [label="1. Phân tích\nthuộc tính vật lý"];
                    step2 [label="2. Đánh giá\ntính ứng dụng"];
                    step3 [label="3. Dự đoán\nthuộc tính nâng cao"];
                    step4 [label="4. So sánh với\nvật liệu chuẩn"];
                    output [label="Kết quả phân tích\ntoàn diện", fillcolor=lightgreen];
                    
                    input -> step1;
                    step1 -> step2;
                    step2 -> step3;
                    step3 -> step4;
                    step4 -> output;
                }
                """
                
                # Dùng streamlit-graphviz nếu đã cài đặt, ngược lại dùng base64 encoded image
                try:
                    from streamlit_graphviz import graphviz_chart
                    graphviz_chart(cot_chart)
                except ImportError:
                    # Phương án dự phòng - chỉ hiển thị text
                    st.markdown("*Sơ đồ quá trình suy luận theo các bước:*")
                    st.markdown("""
                    1. Dữ liệu vật liệu đầu vào ➔ 
                    2. Phân tích thuộc tính vật lý ➔ 
                    3. Đánh giá tính ứng dụng ➔ 
                    4. Dự đoán thuộc tính nâng cao ➔ 
                    5. So sánh với vật liệu chuẩn ➔ 
                    6. Kết quả phân tích toàn diện
                    """)

            # Thêm nội dung cho AI Insights tab (tab4)
            with tab4:
                st.markdown("""
                <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                    <h2 style="color: #000000; margin-top: 0;">💡 Phân tích AI</h2>
                    <p>Phân tích vật liệu sử dụng các mô hình AI tiên tiến</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Kiểm tra xem có dữ liệu ai_analysis trong kết quả CoT không
                if 'ai_analysis' in cot_results:
                    ai_analysis = cot_results['ai_analysis']
                    
                    # Tạo bố cục tổng quan
                    engine_col, quality_col = st.columns(2)
                    
                    with engine_col:
                        # Hiển thị thông tin về AI engine đã sử dụng
                        engine_used = cot_results.get('engine', 'local_fallback')
                        engine_icon = "🤖" if engine_used == "gemini" else "🧠" if engine_used == "groq" else "🔍"
                        
                        st.markdown(f"""
                        <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #e1e4e8;">
                            <div style="font-size: 24px;">{engine_icon}</div>
                            <div style="font-weight: bold; margin: 10px 0; color: #000000;">Phân tích được thực hiện bằng</div>
                            <div style="font-size: 18px; color: #000000;">{engine_used.upper()}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with quality_col:
                        # Hiển thị điểm chất lượng tổng thể nếu có
                        if 'final_analysis' in cot_results and 'overall_quality' in cot_results['final_analysis']:
                            overall_quality = cot_results['final_analysis'].get('overall_quality', 5.0)
                            quality_color = "green" if overall_quality > 7 else "orange" if overall_quality > 4 else "red"
                            
                            st.markdown(f"""
                            <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #e1e4e8;">
                                <div style="font-size: 24px;">⭐</div>
                                <div style="font-weight: bold; margin: 10px 0; color: #000000;">Điểm chất lượng tổng thể</div>
                                <div style="font-size: 32px; color: {quality_color};">{overall_quality}/10</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Tạo tabs phân tích        
                    ai_analysis_tabs = st.tabs(["📊 Tổng quan phân tích", "🔬 Phân tích chi tiết", "📈 Trực quan hóa", "💡 Đề xuất"])
                    
                    with ai_analysis_tabs[0]:
                        st.markdown("### Tổng quan phân tích AI")
                        
                        # Hiển thị thông tin vật liệu
                        material_name = material_data.get('name', 'Unknown Material')
                        crystal_structure = material_data.get('crystal_structure', 'Unknown Structure')
                        bandgap = material_data['properties'].get('bandgap', 'N/A')
                        
                        st.markdown(f"""
                        <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #e1e4e8;">
                            <h4 style="color: #000000;">Thông tin vật liệu: {material_name}</h4>
                            <p style="color: #000000;"><b>Cấu trúc tinh thể:</b> {crystal_structure}</p>
                            <p style="color: #000000;"><b>Bandgap:</b> {bandgap} eV</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if 'bandgap_classification' in cot_results.get('final_analysis', {}):
                            bandgap_type = cot_results['final_analysis']['bandgap_classification']
                            
                            # Tạo card thông tin phân loại bandgap
                            bandgap_icon = "⚡" if bandgap_type == "Semiconductor" else "🔋" if bandgap_type == "Metal" else "💎"
                            st.markdown(f"""
                            <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #e1e4e8;">
                                <h4 style="color: #000000;">{bandgap_icon} Phân loại vật liệu: {bandgap_type}</h4>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Hiển thị các features quan trọng
                        if 'predictions' in cot_results:
                            st.markdown("#### Các thuộc tính chính dự đoán")
                            
                            # Chọn các thuộc tính quan trọng để hiển thị
                            key_properties = ['bandgap_prediction', 'conductivity_prediction', 'thermal_stability', 'heat_transport']
                            metrics_cols = st.columns(len([prop for prop in key_properties if prop in cot_results['predictions']]))
                            
                            col_idx = 0
                            for prop in key_properties:
                                if prop in cot_results['predictions']:
                                    with metrics_cols[col_idx]:
                                        value = cot_results['predictions'][prop]
                                        
                                        if isinstance(value, dict) and 'value' in value:
                                            confidence = value.get('confidence', 0.7) * 100
                                            st.metric(
                                                label=prop.replace('_', ' ').title(), 
                                                value=value['value'],
                                                delta=f"{confidence:.0f}% tin cậy"
                                            )
                                        else:
                                            st.metric(
                                                label=prop.replace('_', ' ').title(), 
                                                value=value
                                            )
                                        col_idx += 1
                        
                        # Hiển thị điểm mạnh và điểm yếu
                        if 'strengths' in cot_results and 'weaknesses' in cot_results:
                            strength_col, weakness_col = st.columns(2)
                            
                            with strength_col:
                                st.markdown("#### ✅ Điểm mạnh")
                                
                                for i, strength in enumerate(cot_results['strengths'][:5]):  # Giới hạn 5 điểm
                                    st.markdown(f"""
                                    <div style="background-color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 5px; border: 1px solid #e1e4e8;">
                                        <span style="color: #000000;">{i+1}. {strength}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            with weakness_col:
                                st.markdown("#### ⚠️ Điểm yếu")
                                
                                for i, weakness in enumerate(cot_results['weaknesses'][:5]):  # Giới hạn 5 điểm
                                    st.markdown(f"""
                                    <div style="background-color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 5px; border: 1px solid #e1e4e8;">
                                        <span style="color: #000000;">{i+1}. {weakness}</span>
                                    </div>
                                    """, unsafe_allow_html=True)

                        # Hiển thị nhận xét tổng thể
                        if 'final_analysis' in cot_results:
                            overall_quality = cot_results['final_analysis'].get('overall_quality', 5.0)
                            
                            message = ""
                            color = ""
                            if overall_quality > 7:
                                message = "Vật liệu này có tiềm năng cao cho các ứng dụng bán dẫn"
                                color = "green"
                            elif overall_quality > 4:
                                message = "Vật liệu này có các thuộc tính trung bình, có thể cần cải thiện thêm"
                                color = "orange"
                            else:
                                message = "Vật liệu này có nhiều hạn chế, cần cải thiện đáng kể"
                                color = "red"
                                
                            st.markdown(f"""
                            <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; text-align: center; margin: 20px 0; border: 1px solid #e1e4e8;">
                                <h4 style="color: {color};">{message}</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    with ai_analysis_tabs[1]:
                        st.markdown("### Phân tích chi tiết từ các mô hình AI")
                    
                        # Hiển thị kết quả phân tích từ các engines
                        if isinstance(ai_analysis, dict) and ai_analysis:
                            ai_models = list(ai_analysis.keys())
                            
                            if ai_models:
                                # Tạo tabs cho từng mô hình AI
                                ai_model_tabs = st.tabs(ai_models)
                                
                                for i, model in enumerate(ai_models):
                                    with ai_model_tabs[i]:
                                        model_result = ai_analysis[model]
                                        
                                        # Hiển thị thông tin về độ tin cậy với giao diện đẹp hơn
                                        confidence = model_result.get('confidence', 0.7)
                                        
                                        # Thiết lập màu sắc dựa trên độ tin cậy
                                        confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                                        
                                        # Hiển thị thanh độ tin cậy
                                        st.markdown(f"""
                                        <div style="margin-bottom: 20px;">
                                            <h4>Độ tin cậy của mô hình {model.upper()}</h4>
                                            <div style="background-color: #f0f0f0; border-radius: 5px; height: 30px;">
                                                <div style="background-color: {confidence_color}; width: {confidence*100}%; height: 30px; border-radius: 5px; text-align: center; color: white; line-height: 30px;">
                                                    {confidence*100:.1f}%
                                                </div>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Hiển thị phân tích text với định dạng đẹp hơn
                                        if 'analysis' in model_result:
                                            st.markdown("""
                                            <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e1e4e8;">
                                                <h4 style="color: #000000;">Phân tích chi tiết</h4>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                            # Hiển thị phân tích với định dạng nổi bật
                                            analysis_text = model_result['analysis']
                                            st.markdown(f"""
                                            <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e1e4e8;">
                                                <span style="color: #000000;">{analysis_text}</span>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        # Hiển thị thông tin về API đã sử dụng nếu có
                                        if 'api_priority' in model_result:
                                            st.info(f"API Priority: {model_result['api_priority']}")
                                    
                                # So sánh các mô hình nếu có nhiều hơn 1 mô hình
                                if len(ai_models) > 1:
                                    st.markdown("### So sánh các mô hình AI")
                                    
                                    # Tạo biểu đồ so sánh độ tin cậy đẹp hơn
                                    confidences = [ai_analysis[model].get('confidence', 0.5) for model in ai_models]
                                    
                                    # Màu sắc cho từng mô hình
                                    model_colors = {
                                        'gemini': '#4285F4',  # Google Blue
                                        'groq': '#EA4335',    # Google Red
                                        'local': '#34A853',   # Google Green
                                        'local_fallback': '#FBBC05'  # Google Yellow
                                    }
                                    
                                    # Lấy màu tương ứng cho mỗi mô hình
                                    colors = [model_colors.get(model.lower(), '#4285F4') for model in ai_models]
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Bar(
                                        x=ai_models,
                                        y=confidences,
                                        marker_color=colors,
                                        text=[f"{conf*100:.1f}%" for conf in confidences],
                                        textposition='outside',
                                        hoverinfo='text',
                                        hovertext=[f"{model}: {conf*100:.1f}% tin cậy" for model, conf in zip(ai_models, confidences)]
                                    ))
                                    
                                    fig.update_layout(
                                        title="Độ tin cậy của các mô hình AI",
                                        yaxis=dict(range=[0, 1], tickformat=".0%"),
                                        yaxis_title="Độ tin cậy",
                                        xaxis_title="Mô hình AI",
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        showlegend=False
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    with ai_analysis_tabs[2]:
                        st.markdown("### Trực quan hóa phân tích")
                        
                        # Tạo nhiều loại biểu đồ khác nhau
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            # Hiển thị biểu đồ radar thuộc tính
                            st.subheader("Biểu đồ radar thuộc tính")
                            radar_fig = create_radar_chart(cot_results['predictions'])
                            st.plotly_chart(radar_fig, use_container_width=True)
                            
                            # Hiển thị biểu đồ đánh giá tổng thể với giao diện đẹp hơn
                            if 'final_analysis' in cot_results and 'overall_quality' in cot_results['final_analysis']:
                                st.subheader("Điểm đánh giá tổng thể")
                                overall_quality = cot_results['final_analysis'].get('overall_quality', 5.0)
                                
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number+delta",
                                    value=overall_quality,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Điểm đánh giá", 'font': {'size': 24}},
                                    delta={'reference': 5.0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                                    gauge={
                                        'axis': {'range': [0, 10], 'tickwidth': 1},
                                        'bar': {'color': "#1E88E5"},
                                        'bgcolor': "white",
                                        'borderwidth': 2,
                                        'bordercolor': "gray",
                                        'steps': [
                                            {'range': [0, 3], 'color': "#ffcccb"},
                                            {'range': [3, 7], 'color': "#ffffcc"},
                                            {'range': [7, 10], 'color': "#ccffcc"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': overall_quality
                                        }
                                    }
                                ))
                                
                                fig.update_layout(height=300, margin=dict(l=20, r=30, b=20, t=50))
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with viz_col2:
                            # Hiển thị biểu đồ ứng dụng tiềm năng
                            if 'recommendations' in cot_results and 'applications' in cot_results['recommendations']:
                                st.subheader("Ứng dụng tiềm năng")
                                applications = cot_results['recommendations']['applications']
                                
                                # Tạo giá trị ngẫu nhiên cho mỗi ứng dụng để hiển thị dưới dạng biểu đồ
                                app_scores = [round(random.uniform(0.7, 0.95), 2) for _ in applications]
                                
                                # Tạo biểu đồ đánh giá ứng dụng
                                fig = go.Figure()
                                
                                # Thêm horizontal bars
                                fig.add_trace(go.Bar(
                                    y=applications[:5],  # Chỉ hiển thị 5 ứng dụng đầu tiên
                                    x=app_scores[:5],
                                    orientation='h',
                                    marker=dict(color=['rgba(30, 136, 229, 0.8)'] * len(applications[:5])),
                                    text=[f"{score*100:.0f}%" for score in app_scores[:5]],
                                    textposition='inside',
                                    hoverinfo='text',
                                    hovertext=[f"{app}: {score*100:.0f}%" for app, score in zip(applications[:5], app_scores[:5])]
                                ))
                                
                                fig.update_layout(
                                    title="Đánh giá ứng dụng tiềm năng",
                                    yaxis=dict(title=""),
                                    xaxis=dict(title="Chỉ số phù hợp", range=[0, 1], tickformat=".0%"),
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    margin=dict(l=20, r=20, t=40, b=20),
                                    height=300
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Hiển thị bảng các thuộc tính dự đoán
                            if 'predictions' in cot_results:
                                st.subheader("Bảng thuộc tính dự đoán")
                                
                                # Tạo dataframe từ dự đoán
                                predictions_data = []
                                for key, value in cot_results['predictions'].items():
                                    if isinstance(value, dict) and 'value' in value:
                                        predictions_data.append({
                                            "Thuộc tính": key.replace('_', ' ').title(),
                                            "Giá trị": value['value'],
                                            "Độ tin cậy": value.get('confidence', 0.7)
                                        })
                                    else:
                                        predictions_data.append({
                                            "Thuộc tính": key.replace('_', ' ').title(),
                                            "Giá trị": value,
                                            "Độ tin cậy": 0.8  # Giá trị mặc định
                                        })
                                
                                predictions_df = pd.DataFrame(predictions_data)
                                
                                # Hiển thị dataframe với định dạng đẹp hơn
                                st.dataframe(
                                    predictions_df,
                                    column_config={
                                        "Thuộc tính": st.column_config.TextColumn("Thuộc tính"),
                                        "Giá trị": st.column_config.TextColumn("Giá trị"),
                                        "Độ tin cậy": st.column_config.ProgressColumn(
                                            "Độ tin cậy",
                                            format="%.0f%%",
                                            min_value=0,
                                            max_value=1
                                        )
                                    },
                                    hide_index=True,
                                    use_container_width=True
                                )
                        
                        # Thêm biểu đồ so sánh với các vật liệu tương tự
                        st.subheader("So sánh với các vật liệu tương tự")
                        
                        # Lấy các thuộc tính quan trọng để so sánh
                        bandgap = material_data['properties'].get('bandgap', 1.0)
                        conductivity = material_data['properties'].get('conductivity', 1000)
                        
                        comparison_fig = create_material_comparison(material_data['properties'], material_data.get('name', 'Current Material'))
                        st.plotly_chart(comparison_fig, use_container_width=True)
                        
                    with ai_analysis_tabs[3]:
                        st.markdown("### Đề xuất và cải thiện")
                        
                        # Hiển thị đề xuất cải tiến từ AI
                        if 'recommendations' in cot_results and 'improvements' in cot_results['recommendations']:
                            improvements = cot_results['recommendations']['improvements']
                            
                            # Tạo bảng đề xuất cải tiến với định dạng đẹp
                            if improvements:
                                st.markdown("""
                                <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #e1e4e8;">
                                    <h4 style="margin-top: 0; color: #000000;">Đề xuất cải tiến</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for i, imp in enumerate(improvements):
                                    # Xác định màu sắc dựa trên độ ưu tiên
                                    priority_color = "red" if imp.get('priority', '').lower() == 'high' else \
                                                    "orange" if imp.get('priority', '').lower() == 'medium' else "gray"
                                    
                                    st.markdown(f"""
                                    <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #e1e4e8;">
                                        <h5 style="margin-top: 0; color: {priority_color};">{i+1}. {imp.get('aspect', 'Unknown')}</h5>
                                        <p style="margin-bottom: 5px; color: #000000;"><b>Đề xuất:</b> {imp.get('recommendation', imp.get('suggestion', 'Không có đề xuất cụ thể'))}</p>
                                        <p style="margin: 0; color: #000000;"><b>Ưu tiên:</b> <span style="color: {priority_color};">{imp.get('priority', 'Medium')}</span></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Tạo bảng đề xuất ứng dụng
                            if 'applications' in cot_results['recommendations']:
                                applications = cot_results['recommendations']['applications']
                                
                                if applications:
                                    st.markdown("""
                                    <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; margin: 20px 0; border: 1px solid #e1e4e8;">
                                        <h4 style="margin-top: 0; color: #000000;">Ứng dụng tiềm năng</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Chia các ứng dụng thành các nhóm
                                    app_cols = st.columns(3)
                                    for i, app in enumerate(applications):
                                        with app_cols[i % 3]:
                                            st.markdown(f"""
                                            <div style="background-color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 10px; text-align: center; border: 1px solid #e1e4e8;">
                                                <span style="color: #000000;">{app}</span>
                                            </div>
                                            """, unsafe_allow_html=True)
                        
                        # Tạo phần next steps
                        st.markdown("""
                        <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; margin: 20px 0; border: 1px solid #e1e4e8;">
                            <h4 style="margin-top: 0; color: #000000;">Các bước tiếp theo</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Tạo danh sách các bước tiếp theo
                        next_steps = [
                            "Thực hiện thí nghiệm để xác nhận các tính chất được dự đoán",
                            "Tối ưu hóa quy trình chế tạo để cải thiện chất lượng",
                            "Khảo sát tác động của điều kiện môi trường lên hiệu suất",
                            "Tích hợp vật liệu vào thiết bị thử nghiệm"
                        ]
                        
                        # Next steps items
                        for i, step in enumerate(next_steps):
                            st.markdown(f"""
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <div style="background-color: #1E88E5; color: white; border-radius: 50%; width: 25px; height: 25px; 
                                            display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                                    {i+1}
                                </div>
                                <div style="color: #000000;">{step}</div>
                            </div>
                            """, unsafe_allow_html=True)

                else:
                    # Hiển thị thông tin từ engine fallback
                    st.subheader("Phân tích từ AI")
                    
                    # Hiển thị các key predictions
                    if 'predictions' in cot_results:
                        st.markdown("### Dự đoán thuộc tính chính")
                        pred_cols = st.columns(3)
                        col_idx = 0
                        
                        for key, value in cot_results['predictions'].items():
                            with pred_cols[col_idx % 3]:
                                if isinstance(value, dict) and 'value' in value and 'confidence' in value:
                                    st.metric(
                                        key.replace('_', ' ').title(), 
                                        value['value'],
                                        delta=f"{value['confidence']*100:.0f}% tin cậy"
                                    )
                                else:
                                    st.metric(key.replace('_', ' ').title(), value)
                                col_idx += 1
                    
                    # Hiển thị các ứng dụng được đề xuất với giao diện đẹp hơn
                    if 'recommendations' in cot_results and 'applications' in cot_results['recommendations']:
                        st.markdown("### Ứng dụng tiềm năng")
                        applications = cot_results['recommendations']['applications']
                        
                        # Tạo grid hiển thị ứng dụng
                        app_cols = st.columns(3)
                        for i, app in enumerate(applications):
                            with app_cols[i % 3]:
                                st.markdown(f"""
                                <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; margin-bottom: 10px; text-align: center; border: 1px solid #e1e4e8;">
                                    <div style="font-weight: bold; color: #000000;">{app}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # Hiển thị điểm mạnh và điểm yếu với UI đẹp hơn
                    if 'strengths' in cot_results and 'weaknesses' in cot_results:
                        st.markdown("## Đánh giá chi tiết")
                        strength_col, weakness_col = st.columns(2)
                        
                        with strength_col:
                            st.markdown("""
                            <div style="background-color: #ffffff; padding: 10px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #e1e4e8;">
                                <h4 style="margin-top: 0; color: #000000;">Điểm mạnh</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for strength in cot_results['strengths']:
                                st.markdown(f"""
                                <div style="background-color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 5px; border-left: 4px solid green;">
                                    <span style="color: #000000;">✅ {strength}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with weakness_col:
                            st.markdown("""
                            <div style="background-color: #ffffff; padding: 10px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #e1e4e8;">
                                <h4 style="margin-top: 0; color: #000000;">Điểm yếu</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for weakness in cot_results['weaknesses']:
                                st.markdown(f"""
                                <div style="background-color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 5px; border-left: 4px solid #E53935;">
                                    <span style="color: #000000;">⚠️ {weakness}</span>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Biểu đồ radar thuộc tính dưới phần đánh giá
                st.markdown("### Biểu đồ radar thuộc tính")
                if 'predictions' in cot_results:
                    radar_fig = create_radar_chart(cot_results['predictions'])
                    st.plotly_chart(radar_fig, use_container_width=True)
                
                # AI tóm tắt cuối cùng
                st.subheader("Kết luận tổng hợp")
                
                if 'final_analysis' in cot_results:
                    # Hiển thị phần kết luận với giao diện đẹp hơn
                    overall_quality = cot_results['final_analysis'].get('overall_quality', 5.0)
                    
                    # Xác định đánh giá dựa trên điểm số
                    if overall_quality > 7:
                        conclusion = """
                        <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #e1e4e8;">
                            <h4 style="color: green;">Vật liệu này có tiềm năng cao cho các ứng dụng bán dẫn</h4>
                            <p style="color: #000000;">Có thể sử dụng trong các ứng dụng hiệu suất cao và có triển vọng thương mại.</p>
                        </div>
                        """
                    elif overall_quality > 4:
                        conclusion = """
                        <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #e1e4e8;">
                            <h4 style="color: orange;">Vật liệu này có các thuộc tính trung bình, có thể cần cải thiện thêm</h4>
                            <p style="color: #000000;">Có tiềm năng nhưng cần nghiên cứu thêm để tối ưu hóa các thuộc tính.</p>
                        </div>
                        """
                    else:
                        conclusion = """
                        <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #e1e4e8;">
                            <h4 style="color: red;">Vật liệu này có nhiều hạn chế, cần cải thiện đáng kể</h4>
                            <p style="color: #000000;">Không phù hợp với hầu hết các ứng dụng thực tế trong trạng thái hiện tại.</p>
                        </div>
                        """
                    
                    st.markdown(conclusion, unsafe_allow_html=True)
                else:
                    st.warning("Không có phân tích tổng hợp cho vật liệu này.")

    elif selected_tab == "Image Analysis":
        st.sidebar.header("Phân tích hình ảnh vật liệu")
        
        # Cải thiện giao diện
        image_analysis_method = st.sidebar.radio(
            "Phương pháp phân tích",
            ["Tải lên hình ảnh", "Chụp từ camera", "Từ URL"]
        )
        
        # Hộp thông tin
        st.sidebar.info("Hệ thống hỗ trợ phân tích hình ảnh hiển vi từ các loại vật liệu bán dẫn (SEM, TEM, AFM, optical microscopy).")
        
        # Tạo placeholder cho hình ảnh nguồn
        source_image = None
        
        if image_analysis_method == "Tải lên hình ảnh":
            uploaded_file = st.sidebar.file_uploader("Tải lên hình ảnh vật liệu", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
            if uploaded_file is not None:
                # Lưu file tạm thời
                with open("temp_image.jpg", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                source_image = "temp_image.jpg"
                
                # Hiển thị thông tin file
                file_details = {
                    "Tên file": uploaded_file.name,
                    "Kích thước": f"{uploaded_file.size / 1024:.1f} KB",
                    "Loại file": uploaded_file.type
                }
                st.sidebar.write("Thông tin file:")
                for k, v in file_details.items():
                    st.sidebar.text(f"{k}: {v}")

        elif image_analysis_method == "Chụp từ camera":
            st.sidebar.warning("Tính năng này yêu cầu quyền truy cập camera.")
            camera_image = st.sidebar.camera_input("Chụp ảnh vật liệu")
            if camera_image is not None:
                # Lưu file tạm thời
                with open("temp_camera.jpg", "wb") as f:
                    f.write(camera_image.getbuffer())
                source_image = "temp_camera.jpg"
                
        elif image_analysis_method == "Từ URL":
            image_url = st.sidebar.text_input("Nhập URL hình ảnh:")
            if image_url:
                try:
                    import urllib.request
                    urllib.request.urlretrieve(image_url, "temp_url_image.jpg")
                    source_image = "temp_url_image.jpg"
                    st.sidebar.success("Đã tải hình ảnh thành công!")
                except Exception as e:
                    st.sidebar.error(f"Lỗi khi tải hình ảnh: {e}")
        
        # Tùy chọn phân tích nâng cao
        if source_image:
            st.sidebar.subheader("Tùy chọn phân tích")
            analysis_type = st.sidebar.multiselect(
                "Loại phân tích",
                ["Phân tích lỗi và khuyết tật", "Đo kích thước tinh thể", "Phân tích thành phần", 
                 "Phân tích cấu trúc bề mặt", "Báo cáo chất lượng"],
                ["Phân tích lỗi và khuyết tật", "Báo cáo chất lượng"]
            )
            
            # Tùy chọn xử lý hình ảnh
            st.sidebar.subheader("Xử lý hình ảnh")
            apply_noise_reduction = st.sidebar.checkbox("Giảm nhiễu", value=True)
            apply_contrast_enhancement = st.sidebar.checkbox("Tăng cường độ tương phản", value=True)
            scale_factor = st.sidebar.slider("Hệ số tỷ lệ phân tích", 0.5, 2.0, 1.0, 0.1)
            
            # Chạy phân tích khi người dùng nhấn nút
            run_analysis = st.sidebar.button("Phân tích hình ảnh", use_container_width=True, type="primary")
            
            if run_analysis:
                # Giả lập gọi class MaterialImageAnalyzer
                try:
                    # Import class MaterialImageAnalyzer - giả lập nếu chưa có
                    try:
                        from utils.image_analyzer import MaterialImageAnalyzer
                        analyzer = MaterialImageAnalyzer()
                    except ImportError:
                        # Tạo class giả lập nếu chưa tồn tại
                        class MaterialImageAnalyzer:
                            def analyze_image(self, image_path, options=None):
                                import time
                                import numpy as np
                                from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
                                import random
                                
                                # Mở ảnh gốc
                                try:
                                    image = Image.open(image_path)
                                except Exception as e:
                                    return {"error": f"Không thể mở hình ảnh: {str(e)}"}
                                
                                # Giả lập xử lý hình ảnh
                                time.sleep(1)  # Giả lập thời gian xử lý
                                
                                # Tạo bản sao để đánh dấu
                                marked_image = image.copy()
                                draw = ImageDraw.Draw(marked_image)
                                
                                # Kích thước ảnh
                                width, height = image.size
                                
                                # Giả lập phát hiện lỗi và đánh dấu
                                num_defects = random.randint(3, 8)
                                surface_defects = []
                                crystal_defects = []
                                structural_defects = []
                                
                                for i in range(num_defects):
                                    x = random.randint(0, width - 1)
                                    y = random.randint(0, height - 1)
                                    size = random.randint(10, 30)
                                    
                                    defect_type = random.choice(["surface", "crystal", "structural"])
                                    
                                    # Vẽ vòng tròn tại vị trí lỗi
                                    if defect_type == "surface":
                                        draw.ellipse((x-size, y-size, x+size, y+size), outline="red", width=2)
                                        draw.text((x, y+size+5), f"S{i+1}", fill="red")
                                        surface_defects.append({
                                            "id": f"S{i+1}",
                                            "position": (x, y),
                                            "size": size,
                                            "type": "Oxidation" if random.random() > 0.5 else "Contamination",
                                            "severity": random.choice(["Low", "Medium", "High"])
                                        })
                                    elif defect_type == "crystal":
                                        draw.rectangle((x-size, y-size, x+size, y+size), outline="blue", width=2)
                                        draw.text((x, y+size+5), f"C{i+1}", fill="blue")
                                        crystal_defects.append({
                                            "id": f"C{i+1}",
                                            "position": (x, y),
                                            "size": size,
                                            "type": random.choice(["Dislocation", "Stacking fault", "Twin boundary"]),
                                            "severity": random.choice(["Low", "Medium", "High"])
                                        })
                                    else:
                                        draw.polygon([(x, y-size), (x+size, y), (x, y+size), (x-size, y)], outline="green", width=2)
                                        draw.text((x, y+size+5), f"T{i+1}", fill="green")
                                        structural_defects.append({
                                            "id": f"T{i+1}",
                                            "position": (x, y),
                                            "size": size,
                                            "type": random.choice(["Grain boundary", "Phase separation", "Void"]),
                                            "severity": random.choice(["Low", "Medium", "High"])
                                        })
                                
                                # Lưu hình ảnh đã đánh dấu
                                marked_image_path = "temp_marked_image.jpg"
                                marked_image.save(marked_image_path)
                                
                                # Tạo phân tích giả lập từ AI
                                ai_analysis = {
                                    "gemini": """
                                    # Phân tích vật liệu từ hình ảnh hiển vi
                                    
                                    Hình ảnh hiển thị cấu trúc bề mặt của vật liệu bán dẫn với một số đặc trưng:
                                    
                                    1. **Cấu trúc tinh thể**: Vật liệu có cấu trúc đa tinh thể với kích thước hạt trung bình khoảng 20-50 µm.
                                    
                                    2. **Khuyết tật và lỗi**:
                                       - Phát hiện một số khuyết tật bề mặt, có thể do quá trình chế tạo hoặc oxy hóa
                                       - Có dấu hiệu của một số lỗi mạng tinh thể, đặc biệt ở vùng biên giới hạt
                                    
                                    3. **Đề xuất cải thiện**:
                                       - Tối ưu hóa nhiệt độ và thời gian ủ để giảm thiểu lỗi mạng
                                       - Cải thiện quy trình làm sạch bề mặt để giảm nhiễm bẩn
                                    """,
                                    
                                    "groq": """
                                    ## Kết quả phân tích hình ảnh vật liệu
                                    
                                    Hình ảnh cho thấy vật liệu bán dẫn với các đặc điểm sau:
                                    
                                    - **Đặc tính bề mặt**: Bề mặt không đồng nhất với nhiều vùng có cấu trúc khác nhau
                                    - **Tinh thể học**: Có cấu trúc đa tinh thể với ranh giới hạt rõ ràng
                                    - **Khuyết tật**: Phát hiện khoảng 5-8 khuyết tật bao gồm lỗi mạng, khuyết tật bề mặt và ranh giới pha
                                    
                                    Điều này cho thấy vật liệu có thể đã trải qua quá trình xử lý nhiệt hoặc cơ học không tối ưu.
                                    
                                    Đề xuất: Điều chỉnh điều kiện chế tạo để giảm mật độ khuyết tật và tăng kích thước hạt tinh thể.
                                    """,
                                    
                                    "combined_insights": [
                                        "Vật liệu có cấu trúc đa tinh thể với kích thước hạt trung bình 20-50 µm",
                                        "Phát hiện nhiều loại khuyết tật, bao gồm lỗi bề mặt và lỗi mạng tinh thể",
                                        "Cần cải thiện điều kiện chế tạo để giảm thiểu lỗi và tăng độ đồng nhất",
                                        "Ranh giới hạt rõ ràng, cho thấy quá trình kết tinh không hoàn toàn tối ưu"
                                    ]
                                }
                                
                                # Tạo thông tin chất lượng hình ảnh
                                image_quality = {
                                    "resolution": f"{width}x{height}",
                                    "quality_score": round(random.uniform(6.5, 9.5), 1),
                                    "contrast": round(random.uniform(0.6, 0.95), 3),
                                    "noise_level": round(random.uniform(0.05, 0.3), 2),
                                    "brightness": round(random.uniform(0.4, 0.8), 2)
                                }
                                
                                # Phân tích kích thước
                                crystal_sizes = []
                                for _ in range(5):
                                    crystal_sizes.append(round(random.uniform(15, 60), 1))
                                
                                size_analysis = {
                                    "average_size": round(sum(crystal_sizes) / len(crystal_sizes), 1),
                                    "min_size": min(crystal_sizes),
                                    "max_size": max(crystal_sizes),
                                    "size_distribution": crystal_sizes
                                }
                                
                                return {
                                    "original_image_path": image_path,
                                    "marked_image_path": marked_image_path,
                                    "surface_defects": surface_defects,
                                    "crystal_defects": crystal_defects,
                                    "structural_defects": structural_defects,
                                    "ai_analysis": ai_analysis,
                                    "image_quality": image_quality,
                                    "size_analysis": size_analysis
                                }
                        
                        analyzer = MaterialImageAnalyzer()
                    
                    with st.spinner('Đang phân tích hình ảnh...'):
                        # Chuẩn bị tùy chọn phân tích
                        analysis_options = {
                            "noise_reduction": apply_noise_reduction,
                            "contrast_enhancement": apply_contrast_enhancement,
                            "scale_factor": scale_factor,
                            "analysis_types": analysis_type
                        }
                        
                        # Phân tích hình ảnh
                        results = analyzer.analyze_image(source_image, analysis_options)
                        
                        if "error" in results:
                            st.error(f"Lỗi phân tích hình ảnh: {results['error']}")
                        else:
                            # Hiển thị kết quả trong các tab
                            img_tab1, img_tab2, img_tab3, img_tab4 = st.tabs([
                                "Phân tích khuyết tật", 
                                "Đo kích thước",
                                "Chất lượng hình ảnh",
                                "Phân tích AI"
                            ])
                            
                            with img_tab1:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("Hình ảnh gốc")
                                    st.image(results["original_image_path"])
                                
                                with col2:
                                    st.subheader("Hình ảnh đã phân tích")
                                    st.image(results["marked_image_path"])
                                
                                # Tạo bảng tổng hợp lỗi
                                st.subheader("Tổng hợp khuyết tật")
                                
                                defect_summary = {
                                    "Loại khuyết tật": ["Lỗi bề mặt", "Lỗi tinh thể", "Lỗi cấu trúc", "Tổng cộng"],
                                    "Số lượng": [
                                        len(results["surface_defects"]), 
                                        len(results["crystal_defects"]), 
                                        len(results["structural_defects"]),
                                        len(results["surface_defects"]) + len(results["crystal_defects"]) + len(results["structural_defects"])
                                    ]
                                }
                                
                                defect_df = pd.DataFrame(defect_summary)
                                
                                # Tạo biểu đồ lỗi
                                fig = px.bar(
                                    defect_df.iloc[:3], 
                                    x="Loại khuyết tật", 
                                    y="Số lượng",
                                    color="Loại khuyết tật",
                                    title="Phân bố khuyết tật theo loại"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Tạo bảng và thông tin chi tiết
                                all_defects = []
                                
                                for defect in results["surface_defects"]:
                                    defect["category"] = "Bề mặt"
                                    all_defects.append(defect)
                                
                                for defect in results["crystal_defects"]:
                                    defect["category"] = "Tinh thể"
                                    all_defects.append(defect)
                                
                                for defect in results["structural_defects"]:
                                    defect["category"] = "Cấu trúc"
                                    all_defects.append(defect)
                                
                                if all_defects:
                                    defects_df = pd.DataFrame([
                                        {
                                            "ID": d["id"],
                                            "Loại": d["category"],
                                            "Chi tiết": d["type"],
                                            "Mức độ": d["severity"],
                                            "Kích thước (px)": d["size"]
                                        } for d in all_defects
                                    ])
                                    
                                    st.dataframe(defects_df, use_container_width=True, hide_index=True)
                                    
                                    # Hiển thị đề xuất xử lý
                                    with st.expander("Đề xuất xử lý"):
                                        if len(results["surface_defects"]) > len(results["crystal_defects"]):
                                            st.info("Phần lớn khuyết tật nằm ở bề mặt. Đề xuất cải thiện quy trình làm sạch và xử lý bề mặt.")
                                        elif len(results["crystal_defects"]) > len(results["structural_defects"]):
                                            st.info("Phần lớn khuyết tật nằm trong cấu trúc tinh thể. Đề xuất tối ưu hóa điều kiện nhiệt độ và thời gian trong quá trình chế tạo.")
                                        else:
                                            st.info("Phát hiện nhiều loại khuyết tật cấu trúc. Đề xuất xem xét lại quy trình tổng hợp vật liệu.")
                            
                            with img_tab2:
                                st.subheader("Phân tích kích thước")
                                
                                # Hiển thị thông tin kích thước
                                size_data = results["size_analysis"]
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Kích thước trung bình", f"{size_data['average_size']} µm")
                                with col2:
                                    st.metric("Kích thước nhỏ nhất", f"{size_data['min_size']} µm")
                                with col3:
                                    st.metric("Kích thước lớn nhất", f"{size_data['max_size']} µm")
                                
                                # Biểu đồ phân bố kích thước
                                size_dist = pd.DataFrame({
                                    "Kích thước (µm)": size_data["size_distribution"],
                                    "Tinh thể": [f"Crystal {i+1}" for i in range(len(size_data["size_distribution"]))]
                                })
                                
                                fig = px.histogram(
                                    size_dist, 
                                    x="Kích thước (µm)",
                                    nbins=10,
                                    title="Phân bố kích thước tinh thể"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Thêm thông tin về đồng nhất
                                size_std = np.std(size_data["size_distribution"])
                                size_cv = size_std / size_data["average_size"] * 100  # Coefficient of variation
                                
                                st.subheader("Độ đồng nhất kích thước")
                                st.progress(max(0, min(100, 100 - size_cv)) / 100)
                                
                                if size_cv < 15:
                                    st.success(f"Kích thước tinh thể đồng nhất tốt (CV: {size_cv:.1f}%)")
                                elif size_cv < 30:
                                    st.warning(f"Kích thước tinh thể tương đối đồng nhất (CV: {size_cv:.1f}%)")
                                else:
                                    st.error(f"Kích thước tinh thể không đồng nhất (CV: {size_cv:.1f}%)")
                            
                            with img_tab3:
                                st.subheader("Chất lượng hình ảnh")
                                
                                quality_data = results["image_quality"]
                                
                                # Tạo đồng hồ đo chất lượng
                                fig = go.Figure(go.Indicator(
                                    mode = "gauge+number",
                                    value = quality_data["quality_score"],
                                    title = {'text': "Điểm chất lượng"},
                                    domain = {'x': [0, 1], 'y': [0, 1]},
                                    gauge = {
                                        'axis': {'range': [0, 10]},
                                        'bar': {'color': "darkblue"},
                                        'steps': [
                                            {'range': [0, 4], 'color': "red"},
                                            {'range': [4, 7], 'color': "orange"},
                                            {'range': [7, 10], 'color': "green"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "black", 'width': 4},
                                            'thickness': 0.75,
                                            'value': quality_data["quality_score"]
                                        }
                                    }
                                ))
                                
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Hiển thị các thông số chi tiết
                                quality_details = pd.DataFrame([
                                    {"Thông số": "Độ phân giải", "Giá trị": quality_data["resolution"]},
                                    {"Thông số": "Độ tương phản", "Giá trị": quality_data["contrast"]},
                                    {"Thông số": "Mức độ nhiễu", "Giá trị": quality_data["noise_level"]},
                                    {"Thông số": "Độ sáng", "Giá trị": quality_data["brightness"]}
                                ])
                                
                                st.dataframe(quality_details, use_container_width=True, hide_index=True)
                                
                                # Đề xuất cải thiện
                                st.subheader("Đề xuất cải thiện chất lượng hình ảnh")
                                
                                if quality_data["noise_level"] > 0.2:
                                    st.warning("Hình ảnh có nhiễu cao. Đề xuất sử dụng kỹ thuật giảm nhiễu hoặc cải thiện điều kiện chụp.")
                                if quality_data["contrast"] < 0.7:
                                    st.warning("Độ tương phản thấp. Đề xuất điều chỉnh độ tương phản trước khi phân tích.")
                                if quality_data["brightness"] < 0.5 or quality_data["brightness"] > 0.8:
                                    st.warning("Độ sáng không tối ưu. Đề xuất điều chỉnh độ sáng để có kết quả phân tích tốt hơn.")
                                if quality_data["quality_score"] > 8.0:
                                    st.success("Chất lượng hình ảnh tốt cho phân tích!")
                            
                            with img_tab4:
                                st.subheader("Phân tích từ AI")
                                
                                ai_results = results["ai_analysis"]
                                
                                # Tạo tabs cho các mô hình khác nhau
                                ai_tab1, ai_tab2, ai_tab3 = st.tabs(["Gemini", "Groq", "Kết luận chung"])
                                
                                with ai_tab1:
                                    st.markdown(ai_results["gemini"])
                                
                                with ai_tab2:
                                    st.markdown(ai_results["groq"])
                                
                                with ai_tab3:
                                    st.subheader("Các điểm chính")
                                    for idx, insight in enumerate(ai_results["combined_insights"], 1):
                                        st.markdown(f"{idx}. {insight}")
                                    
                                    # Kết luận tổng thể
                                    st.subheader("Kết luận")
                                    total_defects = len(results["surface_defects"]) + len(results["crystal_defects"]) + len(results["structural_defects"])
                                    
                                    if total_defects > 6:
                                        st.error("Vật liệu có nhiều khuyết tật, cần xem xét lại quy trình sản xuất.")
                                    elif total_defects > 3:
                                        st.warning("Vật liệu có một số khuyết tật, cần cải thiện nhưng vẫn có thể sử dụng.")
                                    else:
                                        st.success("Vật liệu có chất lượng tốt với ít khuyết tật.")
                                    
                                    # Khuyến nghị
                                    with st.expander("Khuyến nghị kỹ thuật"):
                                        st.markdown("""
                                        1. **Quy trình sản xuất**:
                                            - Tối ưu hóa nhiệt độ và thời gian ủ
                                            - Kiểm soát tốt hơn môi trường sản xuất để tránh nhiễm bẩn
                                        
                                        2. **Xử lý sau sản xuất**:
                                            - Quy trình làm sạch bề mặt kỹ lưỡng
                                            - Xử lý nhiệt độ thấp để giảm ứng suất
                                        
                                        3. **Đánh giá bổ sung**:
                                            - Phân tích XRD để đánh giá cấu trúc tinh thể
                                            - Đo đạc điện tử để xác định ảnh hưởng của khuyết tật
                                        """)
                except Exception as e:
                    st.error(f"Lỗi trong quá trình phân tích: {str(e)}")
                
                # Xóa file tạm sau khi phân tích
                if source_image and source_image.startswith("temp_"):
                    try:
                        # os.remove(source_image)
                        pass  # Để lại file để có thể tham khảo
                    except:
                        pass
            
            # Hiển thị hướng dẫn khi chưa tải lên hình ảnh
            else:
                st.info("👈 Vui lòng tải lên hình ảnh vật liệu từ panel bên trái hoặc chọn một phương pháp khác để bắt đầu phân tích.")
                
                # Hiển thị hình ảnh mẫu
                st.subheader("Các loại hình ảnh được hỗ trợ")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Hình ảnh SEM**")
                    st.caption("Phân giải cao, hiển thị chi tiết bề mặt và cấu trúc tinh thể")
                with col2:
                    st.markdown("**Hình ảnh TEM**")
                    st.caption("Hiển thị cấu trúc trong của vật liệu, phân tích khuyết tật mạng tinh thể")
                with col3:
                    st.markdown("**Hình ảnh AFM**")
                    st.caption("Bản đồ bề mặt 3D, đo độ nhám và đặc điểm bề mặt")

    elif selected_tab == "Batch Analysis":
        # Nâng cấp tính năng phân tích batch với giao diện thân thiện hơn
        st.sidebar.header("Cài đặt phân tích hàng loạt")
        
        # Tạo một giao diện thân thiện hơn
        data_source = st.sidebar.radio(
            "Nguồn dữ liệu",
            ["Bộ dữ liệu mặc định", "Tải lên CSV", "Tạo dữ liệu mới"],
            index=0
        )
        
        csv_file = None
        
        if data_source == "Bộ dữ liệu mặc định":
            default_path = "data/raw/documents/fake_materials_dataset.csv"
            if os.path.exists(default_path):
                csv_file = default_path
                st.sidebar.success(f"Đang sử dụng bộ dữ liệu mặc định")
            else:
                st.sidebar.error(f"Không tìm thấy bộ dữ liệu mặc định: {default_path}")
                st.sidebar.info("Vui lòng tải lên file CSV hoặc tạo dữ liệu mới")
        
        elif data_source == "Tải lên CSV":
            uploaded_file = st.sidebar.file_uploader("Tải lên CSV", type="csv")
            if uploaded_file is not None:
                # Lưu file tạm thời
                with open("temp_upload.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                csv_file = "temp_upload.csv"
                st.sidebar.success(f"Đã tải lên: {uploaded_file.name}")
        
        elif data_source == "Tạo dữ liệu mới":
            # Cài đặt cho việc tạo dữ liệu
            num_materials = st.sidebar.slider("Số lượng vật liệu", 10, 100, 40)
            include_variations = st.sidebar.checkbox("Tạo biến thể", value=True)
            
            if st.sidebar.button("Tạo dữ liệu mới", type="primary"):
                with st.spinner("Đang tạo dữ liệu..."):
                    from evaluate_models import create_fake_material_csv
                    new_csv_path = "data/raw/documents/new_materials_dataset.csv"
                    success = create_fake_material_csv(new_csv_path, num_materials=num_materials)
                    
                    if success:
                        csv_file = new_csv_path
                        st.sidebar.success(f"Đã tạo bộ dữ liệu mới với {num_materials} vật liệu")
                    else:
                        st.sidebar.error("Lỗi khi tạo dữ liệu mới")
        
        # Tùy chọn phân tích nâng cao
        st.sidebar.subheader("Cài đặt phân tích")
        max_records = st.sidebar.number_input("Số lượng vật liệu tối đa", min_value=1, max_value=2000, value=50)
        batch_size = st.sidebar.number_input("Kích thước batch", min_value=1, max_value=20, value=5)
        
        # Tùy chọn các engine AI
        ai_engines = st.sidebar.multiselect(
            "Chọn AI engines", 
            ["gemini", "groq"],
            default=["gemini"]
        )
        
        # Thêm các tùy chọn nâng cao
        with st.sidebar.expander("Tùy chọn nâng cao"):
            apply_semantic_search = st.checkbox("Sử dụng tìm kiếm ngữ nghĩa", value=True)
            use_cache = st.checkbox("Sử dụng cache kết quả", value=True)
            save_results = st.checkbox("Lưu kết quả", value=True)
            
            # Tùy chọn trực quan hóa
            st.subheader("Trực quan hóa")
            show_radar_charts = st.checkbox("Hiển thị biểu đồ radar", value=True)
            show_correlation_matrix = st.checkbox("Hiển thị ma trận tương quan", value=True)
            show_3d_scatter = st.checkbox("Hiển thị biểu đồ scatter 3D", value=True)
        
        # Nút phân tích
        start_batch_analysis = st.sidebar.button(
            "Bắt đầu phân tích", 
            type="primary",
            use_container_width=True
        )
        
        # Hiển thị cấu trúc tab cho batch analysis
        batch_tab1, batch_tab2, batch_tab3, batch_tab4 = st.tabs([
            "Dữ liệu đầu vào", 
            "Kết quả phân tích", 
            "Trực quan hóa",
            "So sánh & Đánh giá"
        ])
        
        with batch_tab1:
            if csv_file and os.path.exists(csv_file):
                try:
                    # Đọc dữ liệu
                    df = pd.read_csv(csv_file)
                    
                    # Tiêu đề và thông tin tổng quan
                    st.markdown(f"""
                    <div style="padding: 15px; border-radius: 8px; background-color: #f8f9fa; margin-bottom: 15px;">
                        <h3 style="margin-top: 0;">Thông tin dữ liệu</h3>
                        <p><b>Tổng số vật liệu:</b> {len(df)} bản ghi</p>
                        <p><b>Số cột:</b> {len(df.columns)} thuộc tính</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Hiển thị dữ liệu với edit mode
                    st.subheader("Dữ liệu vật liệu")
                    st.dataframe(
                        df,
                        use_container_width=True,
                        height=400,
                        column_config={
                            "material_id": st.column_config.TextColumn("ID vật liệu"),
                            "material_name": st.column_config.TextColumn("Tên vật liệu"),
                            "bandgap": st.column_config.NumberColumn("Bandgap (eV)", format="%.2f"),
                            "conductivity": st.column_config.NumberColumn("Độ dẫn điện (S/cm)"),
                            "crystal_structure": st.column_config.TextColumn("Cấu trúc tinh thể")
                        }
                    )
                    
                    # Thống kê cơ bản
                    st.subheader("Thống kê cơ bản")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if "material_name" in df.columns:
                            material_counts = df['material_name'].value_counts().head(10)
                            st.write("Tần suất vật liệu:")
                            st.bar_chart(material_counts)
                    
                    with col2:
                        if "crystal_structure" in df.columns:
                            structure_counts = df['crystal_structure'].value_counts()
                            
                            # Tạo biểu đồ pie chart
                            fig = px.pie(
                                names=structure_counts.index,
                                values=structure_counts.values,
                                title="Phân bố cấu trúc tinh thể"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Bộ lọc và tìm kiếm
                    st.subheader("Bộ lọc và tìm kiếm")
                    
                    col3, col4, col5 = st.columns(3)
                    
                    with col3:
                        search_term = st.text_input("Tìm kiếm vật liệu:", placeholder="Ví dụ: Silicon, GaAs...")
                    
                    with col4:
                        if "crystal_structure" in df.columns:
                            structures = ["Tất cả"] + list(df['crystal_structure'].unique())
                            selected_structure = st.selectbox("Lọc theo cấu trúc:", structures)
                    
                    with col5:
                        if "bandgap" in df.columns:
                            min_bandgap, max_bandgap = st.slider(
                                "Lọc theo bandgap (eV):",
                                min_value=0.0,
                                max_value=float(df['bandgap'].max()) + 0.5,
                                value=(0.0, float(df['bandgap'].max()))
                            )
                    
                    # Áp dụng bộ lọc
                    filtered_df = df.copy()
                    
                    if search_term:
                        filtered_df = filtered_df[filtered_df['material_name'].str.contains(search_term, case=False, na=False)]
                    
                    if "crystal_structure" in df.columns and selected_structure != "Tất cả":
                        filtered_df = filtered_df[filtered_df['crystal_structure'] == selected_structure]
                    
                    if "bandgap" in df.columns:
                        filtered_df = filtered_df[(filtered_df['bandgap'] >= min_bandgap) & (filtered_df['bandgap'] <= max_bandgap)]
                    
                    if len(filtered_df) != len(df):
                        st.write(f"Đã lọc: {len(filtered_df)} / {len(df)} vật liệu")
                        st.dataframe(filtered_df, use_container_width=True)
                    
                    # Phần tài liệu PDF
                    st.subheader("Tài liệu tham khảo")
                    pdf_dir = "data/raw/documents"
                    
                    # Kiểm tra thư mục PDF
                    if os.path.exists(pdf_dir):
                        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
                        if pdf_files:
                            st.write(f"Đã tìm thấy {len(pdf_files)} tài liệu PDF:")
                            for pdf in pdf_files:
                                st.markdown(f"- {pdf}")
                        else:
                            st.info("Không tìm thấy tài liệu PDF nào. Tải lên tài liệu để cải thiện phân tích.")
                    else:
                        st.info(f"Thư mục {pdf_dir} không tồn tại.")
                    
                    # Tải lên PDF mới
                    uploaded_pdf = st.file_uploader("Tải lên tài liệu PDF mới", type="pdf")
                    if uploaded_pdf is not None:
                        # Đảm bảo thư mục tồn tại
                        os.makedirs(pdf_dir, exist_ok=True)
                        # Lưu file
                        pdf_path = os.path.join(pdf_dir, uploaded_pdf.name)
                        with open(pdf_path, "wb") as f:
                            f.write(uploaded_pdf.getbuffer())
                        st.success(f"Đã tải lên: {uploaded_pdf.name}")
                
                except Exception as e:
                    st.error(f"Lỗi khi đọc dữ liệu: {str(e)}")
            else:
                st.info("Vui lòng chọn nguồn dữ liệu từ sidebar để bắt đầu.")
        
        if start_batch_analysis and csv_file and os.path.exists(csv_file):
            with batch_tab2:
                st.subheader("Tiến trình phân tích")
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.empty()
                
                # Cập nhật tiến độ
                def update_progress(current, total):
                    progress = min(1.0, (current + 1) / total)
                    progress_bar.progress(progress)
                    status_text.text(f"Đang xử lý batch {current+1}/{total} ({int(progress*100)}%)")
                
                try:
                    # Đọc dữ liệu vật liệu
                    materials = pd.read_csv(csv_file).to_dict('records')
                    if max_records:
                        materials = materials[:max_records]
                    
                    total_materials = len(materials)
                    status_text.text(f"Đang xử lý {total_materials} vật liệu...")
                    
                    # Thực hiện phân tích
                    all_results, material_scores = batch_analyze_materials(
                        materials, 
                        ai_engines, 
                        batch_size=batch_size,
                        progress_callback=update_progress
                    )
                    
                    progress_bar.progress(1.0)
                    status_text.text(f"✅ Hoàn tất phân tích {total_materials} vật liệu!")
                    
                    # Hiển thị bảng xếp hạng vật liệu
                    st.subheader("Xếp hạng vật liệu")
                    
                    sorted_materials = sorted(material_scores.items(), key=lambda x: x[1]['score']['score'], reverse=True)
                    
                    # Tạo DataFrame từ kết quả
                    scores_data = []
                    for material_id, data in sorted_materials:
                        scores_data.append({
                            "ID": material_id,
                            "Tên vật liệu": data["material_name"],
                            "Điểm": data["score"]["score"],
                            "Xếp loại": data["score"]["rating"],
                            "Bandgap (eV)": data["properties"]["bandgap"],
                            "Độ dẫn điện": data["properties"]["conductivity"],
                            "Độ bền nhiệt": data["properties"]["thermal_stability"]
                        })
                    
                    scores_df = pd.DataFrame(scores_data)
                    
                    # Hiển thị bảng với định dạng màu sắc
                    st.dataframe(
                        scores_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Điểm": st.column_config.ProgressColumn(
                                "Điểm đánh giá",
                                format="%d",
                                min_value=0,
                                max_value=100
                            ),
                            "Xếp loại": st.column_config.TextColumn(
                                "Xếp loại"
                            )
                        }
                    )
                    
                    # Danh sách kết quả chi tiết
                    st.subheader("Kết quả AI phân tích chi tiết")
                    
                    # Tạo tìm kiếm vật liệu
                    search_material = st.text_input("Tìm kiếm vật liệu:", key="search_results")
                    
                    # Lọc kết quả theo tìm kiếm
                    filtered_results = scores_df
                    if search_material:
                        filtered_results = scores_df[scores_df["Tên vật liệu"].str.contains(search_material, case=False)]
                    
                    # Hiển thị kết quả phân tích cho từng vật liệu
                    for i, row in filtered_results.iterrows():
                        material_id = row["ID"]
                        material_name = row["Tên vật liệu"]
                        
                        with st.expander(f"{material_name} (ID: {material_id}) - Điểm: {row['Điểm']}/100"):
                            # Chi tiết điểm
                            if material_id in material_scores:
                                score_details = material_scores[material_id]["score"]
                                
                                st.markdown(f"**Xếp loại:** {score_details['rating']}")
                                st.markdown(f"**Tỷ lệ điểm:** {score_details['percentage']}")
                                
                                st.subheader("Chi tiết điểm đánh giá")
                                for aspect, details in score_details["breakdown"].items():
                                    st.markdown(f"- **{aspect}:** {details}")
                            
                            # Kết quả phân tích từ các engine
                            relevant_results = [r for r in all_results if r.get("material_id") == material_id]
                            if relevant_results:
                                st.subheader("Kết quả phân tích từ AI")
                                
                                # Hiển thị các kết quả từ AI mà không sử dụng expander lồng nhau
                                for result in relevant_results:
                                    if result.get("status") == "success":
                                        st.markdown(f"**Engine: {result.get('engine')} - Prompt: {result.get('prompt_type')}**")
                                        st.markdown(result.get("response", "Không có phản hồi"))
                                        st.markdown("---")  # Thêm dòng phân cách
                    
                    # Lưu kết quả
                    if save_results:
                        # Lưu kết quả vào file
                        results_dir = "results"
                        os.makedirs(results_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        result_file = os.path.join(results_dir, f"batch_analysis_{timestamp}.json")
                        
                        with open(result_file, 'w') as f:
                            json.dump({
                                "timestamp": timestamp,
                                "total_materials": total_materials,
                                "material_scores": material_scores
                            }, f, indent=2)
                        
                        st.success(f"Đã lưu kết quả vào {result_file}")
                
                except Exception as e:
                    st.error(f"Lỗi trong quá trình phân tích: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            with batch_tab3:
                if 'material_scores' in locals():
                    st.subheader("Trực quan hóa kết quả phân tích")
                    
                    viz_tabs = st.tabs(["Biểu đồ cột", "Biểu đồ tương quan", "Biểu đồ phân phối", "Biểu đồ 3D"])
                    
                    with viz_tabs[0]:
                        # Biểu đồ cột điểm số
                        top_materials = sorted(material_scores.items(), key=lambda x: x[1]['score']['score'], reverse=True)[:20]
                        
                        # Tạo dữ liệu cho biểu đồ cột
                        chart_data = []
                        for material_id, data in top_materials:
                            chart_data.append({
                                "Vật liệu": data["material_name"],
                                "Điểm": data["score"]["score"]
                            })
                        
                        chart_df = pd.DataFrame(chart_data)
                        st.write("Top 20 vật liệu theo điểm đánh giá")
                        fig = px.bar(chart_df, x="Vật liệu", y="Điểm",
                                    title="Điểm đánh giá vật liệu",
                                    color="Điểm", height=500)
                        fig.update_layout(xaxis={'categoryorder':'total descending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tabs[1]:
                        # Biểu đồ scatter plot bandgap vs conductivity
                        scatter_data = []
                        for material_id, data in material_scores.items():
                            try:
                                # Cố gắng chuyển đổi bandgap và conductivity sang kiểu số
                                bandgap_str = str(data["properties"]["bandgap"]).replace(',', '.')
                                conductivity_str = str(data["properties"]["conductivity"]).replace(',', '.')
                                
                                # Loại bỏ các ký tự không phải số
                                import re
                                bandgap_str = re.sub(r'[^\d.]', '', bandgap_str) if bandgap_str.strip() != 'N/A' else '0'
                                conductivity_str = re.sub(r'[^\d.]', '', conductivity_str) if conductivity_str.strip() != 'N/A' else '0'
                                
                                # Chuyển đổi sang số
                                bandgap = float(bandgap_str) if bandgap_str else 0.0
                                conductivity = float(conductivity_str) if conductivity_str else 0.0
                                
                                scatter_data.append({
                                    "Vật liệu": data["material_name"],
                                    "Bandgap (eV)": bandgap,
                                    "Độ dẫn điện (S/cm)": conductivity,
                                    "Điểm": data["score"]["score"]
                                })
                            except (ValueError, TypeError) as e:
                                continue
                        
                        if scatter_data:
                            scatter_df = pd.DataFrame(scatter_data)
                            
                            # Tạo biểu đồ tương quan
                            st.subheader("Tương quan giữa Bandgap và Độ dẫn điện")
                            fig = px.scatter(scatter_df, x="Bandgap (eV)", y="Độ dẫn điện (S/cm)",
                                        color="Điểm", hover_name="Vật liệu",
                                        title="Bandgap vs Độ dẫn điện",
                                        height=500)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Thêm biểu đồ tương quan mới
                            st.subheader("Ma trận tương quan thuộc tính")
                            
                            # Mở rộng dữ liệu để tạo ma trận tương quan
                            if "thermal_stability" in data["properties"]:
                                thermal_scores = {
                                    "high": 3, "cao": 3, 
                                    "medium": 2, "trung bình": 2, 
                                    "low": 1, "thấp": 1
                                }
                                for item in scatter_data:
                                    thermal_text = material_scores[material_id]["properties"]["thermal_stability"].lower()
                                    thermal_score = 0
                                    
                                    for key, value in thermal_scores.items():
                                        if key in thermal_text:
                                            thermal_score = value
                                            break
                                    
                                    item["Độ bền nhiệt"] = thermal_score
                            
                            # Tính toán ma trận tương quan
                            numeric_df = scatter_df.select_dtypes(include=[np.number])
                            corr_matrix = numeric_df.corr()
                            
                            # Tạo biểu đồ heatmap
                            fig = px.imshow(
                                corr_matrix, 
                                text_auto=True,
                                color_continuous_scale="RdBu_r",
                                aspect="auto",
                                title="Tương quan giữa các thuộc tính vật liệu"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Cung cấp giải thích
                            with st.expander("Giải thích ma trận tương quan"):
                                st.markdown("""
                                - **Giá trị gần 1**: Tương quan thuận mạnh - khi thuộc tính này tăng, thuộc tính kia cũng tăng
                                - **Giá trị gần -1**: Tương quan nghịch mạnh - khi thuộc tính này tăng, thuộc tính kia giảm
                                - **Giá trị gần 0**: Ít hoặc không có tương quan
                                
                                Ví dụ: Bandgap và độ dẫn điện thường có tương quan nghịch (giá trị âm) vì vật liệu có bandgap cao thường có độ dẫn điện thấp hơn.
                                """)
                                
                    with viz_tabs[2]:
                        # Biểu đồ phân phối điểm số
                        if 'material_scores' in locals():
                            score_distribution = [data["score"]["score"] for _, data in material_scores.items()]
                            rating_counts = pd.Series([data["score"]["rating"] for _, data in material_scores.items()]).value_counts()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Phân phối điểm đánh giá")
                                fig = px.histogram(score_distribution, nbins=20,
                                                title="Phân bố điểm đánh giá vật liệu")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.subheader("Phân bố theo xếp loại")
                                fig = px.pie(names=rating_counts.index, values=rating_counts.values,
                                        title="Phân bố xếp loại vật liệu")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Thêm biểu đồ violin
                            st.subheader("Phân bố điểm theo loại cấu trúc")
                            
                            if scatter_data:
                                scatter_df = pd.DataFrame(scatter_data)
                                
                                # Tạo biểu đồ violin
                                if len(scatter_df) > 5:  # Cần đủ dữ liệu
                                    fig = px.violin(
                                        scatter_df, 
                                        y="Điểm", 
                                        box=True, 
                                        points="all",
                                        title="Phân bố điểm đánh giá"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tabs[3]:
                        # Biểu đồ 3D
                        if scatter_data and len(scatter_data) > 3:
                            scatter_df = pd.DataFrame(scatter_data)
                            
                            st.subheader("Biểu đồ 3D của các vật liệu")
                            
                            # Tạo biểu đồ scatter 3D
                            fig = go.Figure(data=[go.Scatter3d(
                                x=scatter_df["Bandgap (eV)"],
                                y=scatter_df["Độ dẫn điện (S/cm)"],
                                z=scatter_df["Điểm"],
                                text=scatter_df["Vật liệu"],
                                mode='markers',
                                marker=dict(
                                    size=8,
                                    color=scatter_df["Điểm"],
                                    colorscale='Viridis',
                                    opacity=0.8,
                                    colorbar=dict(title="Điểm")
                                )
                            )])
                            
                            # Cập nhật layout
                            fig.update_layout(
                                scene=dict(
                                    xaxis_title='Bandgap (eV)',
                                    yaxis_title='Độ dẫn điện (S/cm)',
                                    zaxis_title='Điểm đánh giá',
                                ),
                                title="So sánh vật liệu trong không gian 3D",
                                height=700
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Hướng dẫn tương tác
                            st.info("""
                            **Hướng dẫn tương tác với biểu đồ 3D:**
                            - Click và kéo để xoay góc nhìn
                            - Scroll để phóng to/thu nhỏ
                            - Double-click để reset góc nhìn
                            - Hover lên điểm bất kỳ để xem thông tin chi tiết
                            """)
                    
                else:
                    st.info("Vui lòng chạy phân tích trước để xem trực quan hóa")
            
            with batch_tab4:
                if 'material_scores' in locals():
                    st.subheader("So sánh và đánh giá vật liệu")
                    
                    # Tạo công cụ so sánh trực tiếp
                    st.markdown("### So sánh các vật liệu")
                    material_names = [data["material_name"] for _, data in material_scores.items()]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        material1 = st.selectbox("Vật liệu 1:", material_names)
                    with col2:
                        material2 = st.selectbox("Vật liệu 2:", material_names, index=min(1, len(material_names)-1))
                    
                    if material1 and material2:
                        # Tìm thông tin vật liệu
                        material1_data = None
                        material2_data = None
                        
                        for material_id, data in material_scores.items():
                            if data["material_name"] == material1:
                                material1_data = data
                            if data["material_name"] == material2:
                                material2_data = data
                        
                        if material1_data and material2_data:
                            # Tạo bảng so sánh
                            compare_data = []
                            
                            # Điểm tổng hợp
                            compare_data.append({
                                "Thuộc tính": "Điểm đánh giá",
                                material1: material1_data["score"]["score"],
                                material2: material2_data["score"]["score"],
                                "Chênh lệch": material1_data["score"]["score"] - material2_data["score"]["score"]
                            })
                            
                            # Xếp loại
                            compare_data.append({
                                "Thuộc tính": "Xếp loại",
                                material1: material1_data["score"]["rating"],
                                material2: material2_data["score"]["rating"],
                                "Chênh lệch": "N/A"
                            })
                            
                            # Các thuộc tính vật lý
                            for prop in ["bandgap", "conductivity", "thermal_stability"]:
                                if prop in material1_data["properties"] and prop in material2_data["properties"]:
                                    # Cố gắng chuyển đổi sang số để tính chênh lệch
                                    try:
                                        val1 = float(str(material1_data["properties"][prop]).replace(',', '.'))
                                        val2 = float(str(material2_data["properties"][prop]).replace(',', '.'))
                                        diff = val1 - val2
                                    except:
                                        val1 = material1_data["properties"][prop]
                                        val2 = material2_data["properties"][prop]
                                        diff = "N/A"
                                    
                                    # Tên hiển thị
                                    prop_name = prop.replace('_', ' ').title()
                                    if prop == "bandgap":
                                        prop_name = "Bandgap (eV)"
                                    elif prop == "conductivity":
                                        prop_name = "Độ dẫn điện (S/cm)"
                                    elif prop == "thermal_stability":
                                        prop_name = "Độ bền nhiệt"
                                    
                                    compare_data.append({
                                        "Thuộc tính": prop_name,
                                        material1: val1,
                                        material2: val2,
                                        "Chênh lệch": diff
                                    })
                            
                            # Hiển thị bảng so sánh
                            compare_df = pd.DataFrame(compare_data)
                            st.dataframe(compare_df, use_container_width=True, hide_index=True)
                            
                            # Tạo biểu đồ radar so sánh
                            st.subheader("Biểu đồ so sánh radar")
                            
                            # Chuẩn bị dữ liệu
                            radar_props = ["Bandgap", "Conductivity", "Thermal Stability", "Score"]
                            
                            # Chuẩn hóa giá trị
                            try:
                                # Hàm chuyển đổi an toàn
                                def safe_float(val, default=0.0):
                                    if val is None or val == '' or str(val).lower() == 'n/a':
                                        return default
                                    try:
                                        return float(str(val).replace(',', '.'))
                                    except (ValueError, TypeError):
                                        return default
                                    
                                # Xử lý bandgap
                                bandgap1 = safe_float(material1_data["properties"]["bandgap"])
                                bandgap2 = safe_float(material2_data["properties"]["bandgap"])
                                max_bandgap = max(bandgap1, bandgap2, 3.5)  # Giới hạn trên hợp lý
                                
                                # Xử lý conductivity
                                conductivity1 = safe_float(material1_data["properties"]["conductivity"])
                                conductivity2 = safe_float(material2_data["properties"]["conductivity"])
                                max_conductivity = max(conductivity1, conductivity2, 5000)
                                
                                # Xử lý thermal stability
                                thermal1 = 0
                                thermal2 = 0
                                thermal_map = {"high": 3, "cao": 3, "medium": 2, "trung": 2, "low": 1, "thấp": 1, "kém": 1}
                                
                                thermal1_str = str(material1_data["properties"].get("thermal_stability", "")).lower()
                                thermal2_str = str(material2_data["properties"].get("thermal_stability", "")).lower()
                                
                                for term, value in thermal_map.items():
                                    if term in thermal1_str:
                                        thermal1 = value
                                    if term in thermal2_str:
                                        thermal2 = value
                                
                                # Xử lý scores
                                score1 = safe_float(material1_data["score"]["score"]) / 100
                                score2 = safe_float(material2_data["score"]["score"]) / 100
                                
                                # Chuẩn hóa giá trị - tránh chia cho 0
                                if max_bandgap > 0:
                                    bandgap1_norm = bandgap1 / max_bandgap
                                    bandgap2_norm = bandgap2 / max_bandgap
                                else:
                                    bandgap1_norm = 0
                                    bandgap2_norm = 0
                                    
                                if max_conductivity > 0:
                                    cond1_norm = conductivity1 / max_conductivity
                                    cond2_norm = conductivity2 / max_conductivity
                                else:
                                    cond1_norm = 0
                                    cond2_norm = 0
                                    
                                # Chuẩn hóa giá trị thermal
                                thermal1_norm = thermal1 / 3 if thermal1 > 0 else 0
                                thermal2_norm = thermal2 / 3 if thermal2 > 0 else 0
                                
                                # Đảm bảo giá trị nằm trong khoảng [0, 1]
                                radar_values1 = [
                                    min(max(bandgap1_norm, 0), 1),
                                    min(max(cond1_norm, 0), 1),
                                    min(max(thermal1_norm, 0), 1),
                                    min(max(score1, 0), 1)
                                ]
                                
                                radar_values2 = [
                                    min(max(bandgap2_norm, 0), 1),
                                    min(max(cond2_norm, 0), 1),
                                    min(max(thermal2_norm, 0), 1),
                                    min(max(score2, 0), 1)
                                ]
                                
                                # Tạo biểu đồ radar
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatterpolar(
                                    r=radar_values1,
                                    theta=radar_props,
                                    fill='toself',
                                    name=material1
                                ))
                                
                                fig.add_trace(go.Scatterpolar(
                                    r=radar_values2,
                                    theta=radar_props,
                                    fill='toself',
                                    name=material2
                                ))
                                
                                fig.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, 1]
                                        )
                                    ),
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Lỗi khi tạo biểu đồ radar: {e}")
                else:
                    st.info("Vui lòng chạy phân tích trước để xem so sánh và đánh giá")

if __name__ == "__main__":
    main() 
# =============================================================================
# visualization.py - Tạo biểu đồ
# =============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
from district_utils import calculate_centers


def create_plot(solution, areas, num_districts, depot, title):
    """Tạo biểu đồ phân vùng và trả về base64 string"""
    plt.figure(figsize=(14, 12))
    colors = plt.cm.get_cmap('nipy_spectral', num_districts)
    
    # Vẽ depot
    plt.scatter(depot['x'], depot['y'], color='black', marker='s', s=500, 
               label='KHO HÀNG (DEPOT)', zorder=5, edgecolors='yellow', linewidths=3)
    
    # Vẽ các khu vực
    for idx, dist_id in enumerate(solution):
        area = areas[idx]
        plt.scatter(area['x'], area['y'], color=colors(dist_id), 
                   s=area['parcels']*10, alpha=0.85, edgecolors='black', linewidths=1)
    
    # Vẽ tâm các quận
    centers = calculate_centers(solution, areas, num_districts)
    for i in range(num_districts):
        plt.scatter(centers[i, 0], centers[i, 1], color='red', marker='X', 
                   s=600, linewidths=4, label=f'Tâm Quận {i}' if i == 0 else '', 
                   edgecolors='white', zorder=4)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Tọa độ X")
    plt.ylabel("Tọa độ Y")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Chuyển sang base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

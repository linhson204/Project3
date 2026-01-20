# =============================================================================
# generate_test_cases.py - Tạo và lưu Test Cases vào JSON
# =============================================================================
"""
Script để sinh dữ liệu test cases và lưu vào file JSON.
Chạy 1 lần để tạo dữ liệu cố định.

Usage: python generate_test_cases.py
"""

import os
import json
import math
from data_generator import generate_areas, create_drivers, create_vehicles

# =============================================================================
# CÔNG THỨC TÍNH TOÁN
# =============================================================================

def calculate_driver_count(num_areas: int) -> int:
    """Tính số tài xế dựa trên số khu vực"""
    if num_areas <= 45:
        return 1
    elif num_areas <= 70:
        return 2
    elif num_areas <= 100:
        return 3
    elif num_areas == 150:
        return 5  # Special case: 150 areas = 5 drivers
    elif num_areas <= 200:
        return (num_areas + 29) // 30 - 1
    elif num_areas <= 500:
        return (num_areas + 29) // 30
    elif num_areas <= 700:
        return (num_areas + 29) // 30 + 1
    else:
        return (num_areas + 29) // 30 + 2


def calculate_test_params(num_areas: int) -> dict:
    """Tính toán tất cả parameters cho một test case"""
    map_size = math.ceil(math.sqrt(num_areas / 7) * 10)
    total_drivers = calculate_driver_count(num_areas)
    fulltime = math.ceil(total_drivers / 2)
    parttime = total_drivers - fulltime
    
    large_vehicles = math.ceil(total_drivers * 0.6)
    medium_vehicles = math.ceil(total_drivers * 0.2) if total_drivers >= 3 else 0
    small_vehicles = max(0, total_drivers - large_vehicles - medium_vehicles)
    
    num_districts = total_drivers
    
    return {
        'num_areas': num_areas,
        'map_size': map_size,
        'num_districts': num_districts,
        'fulltime': fulltime,
        'parttime': parttime,
        'large_vehicles': large_vehicles,
        'medium_vehicles': medium_vehicles,
        'small_vehicles': small_vehicles,
        'total_drivers': total_drivers
    }


def generate_and_save_test_case(num_areas: int, output_dir: str = "test_cases"):
    """
    Tạo một test case và lưu vào file JSON.
    """
    params = calculate_test_params(num_areas)
    
    # Sinh dữ liệu
    areas = generate_areas(
        num_areas=params['num_areas'],
        map_size=params['map_size'],
        num_sample_days=5
    )
    
    drivers = create_drivers(
        num_fulltime=params['fulltime'],
        num_parttime=params['parttime']
    )
    
    vehicles = create_vehicles(
        num_large=params['large_vehicles'],
        num_medium=params['medium_vehicles'],
        num_small=params['small_vehicles']
    )
    
    depot = {
        'x': params['map_size'] // 2,
        'y': params['map_size'] // 2
    }
    
    test_case = {
        'name': f"TestCase_{num_areas}_areas",
        'params': params,
        'areas': areas,
        'drivers': drivers,
        'vehicles': vehicles,
        'depot': depot,
        'num_districts': params['num_districts'],
        'map_size': params['map_size']
    }
    
    # Tạo thư mục nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu vào file JSON
    filepath = os.path.join(output_dir, f"case_{num_areas}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(test_case, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created: {filepath}")
    return filepath


def main():
    """Tạo tất cả test cases"""
    TEST_SIZES = [50, 100, 150, 300, 500, 700, 800, 1000, 1200, 1500]
    
    print("=" * 60)
    print("GENERATING FIXED TEST CASES (JSON)")
    print("=" * 60)
    
    # In bảng tóm tắt
    print(f"\n{'Areas':<8} {'Map':<6} {'Districts':<10} {'Drivers':<10} {'Vehicles':<15}")
    print(f"{'':8} {'Size':<6} {'':10} {'FT/PT':<10} {'L/M/S':<15}")
    print("-" * 60)
    
    for size in TEST_SIZES:
        params = calculate_test_params(size)
        drivers_str = f"{params['fulltime']}/{params['parttime']}"
        vehicles_str = f"{params['large_vehicles']}/{params['medium_vehicles']}/{params['small_vehicles']}"
        print(f"{size:<8} {params['map_size']:<6} {params['num_districts']:<10} {drivers_str:<10} {vehicles_str:<15}")
    
    print("\n" + "-" * 60)
    print("Generating JSON files...")
    print("-" * 60)
    
    for size in TEST_SIZES:
        generate_and_save_test_case(size)
    
    print("\n" + "=" * 60)
    print("✓ All test cases generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

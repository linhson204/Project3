# =============================================================================
# test_cases.py - Loader cho Test Cases từ JSON
# =============================================================================
"""
Utility để load các bộ test cases cố định từ file JSON.

Usage:
    from test_cases import get_test_case, get_test_cases, list_available_cases
    
    # Lấy một test case cụ thể
    case = get_test_case(100)
    
    # Lấy tất cả test cases
    cases = get_test_cases()
"""

import os
import json

# Thư mục chứa test cases
TEST_CASES_DIR = os.path.join(os.path.dirname(__file__), "test_cases")

# Các test sizes có sẵn
AVAILABLE_SIZES = [50, 100, 150, 300, 500, 700, 800, 1000, 1200, 1500]

# Cache để tránh đọc file nhiều lần
_cache = {}


def get_test_case(num_areas: int) -> dict:
    """ Load một test case từ file JSON. """
    if num_areas not in AVAILABLE_SIZES:
        raise ValueError(f"Test case for {num_areas} areas not found. Available: {AVAILABLE_SIZES}")
    
    # Kiểm tra cache
    if num_areas in _cache:
        return _cache[num_areas]
    
    # Đọc từ file
    filepath = os.path.join(TEST_CASES_DIR, f"case_{num_areas}.json")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Test case file not found: {filepath}\n"
            f"Run 'python generate_test_cases.py' to generate test cases."
        )
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Cache kết quả
    _cache[num_areas] = data
    return data


def get_test_cases() -> dict:
    """ Load tất cả test cases có sẵn. """
    result = {}
    for size in AVAILABLE_SIZES:
        try:
            result[size] = get_test_case(size)
        except FileNotFoundError:
            pass  # Skip nếu file không tồn tại
    return result


def list_available_cases() -> list:
    """ Liệt kê các test cases có sẵn trong thư mục. """
    available = []
    for size in AVAILABLE_SIZES:
        filepath = os.path.join(TEST_CASES_DIR, f"case_{size}.json")
        if os.path.exists(filepath):
            available.append(size)
    return available


def clear_cache():
    """ Xóa cache để đọc lại từ file """
    global _cache
    _cache = {}


def print_summary():
    """ In bảng tóm tắt các test cases có sẵn """
    cases = get_test_cases()
    
    if not cases:
        print("No test cases found. Run 'python generate_test_cases.py' first.")
        return
    
    print("\n" + "=" * 70)
    print("AVAILABLE TEST CASES")
    print("=" * 70)
    print(f"{'Areas':<8} {'Map':<6} {'Districts':<10} {'Drivers':<12} {'Vehicles':<12} {'Depot':<10}")
    print("-" * 70)
    
    for size in sorted(cases.keys()):
        case = cases[size]
        p = case['params']
        drivers_str = f"{p['fulltime']}FT + {p['parttime']}PT"
        vehicles_str = f"{p['large_vehicles']}L/{p['medium_vehicles']}M/{p['small_vehicles']}S"
        depot_str = f"({case['depot']['x']}, {case['depot']['y']})"
        print(f"{size:<8} {p['map_size']:<6} {p['num_districts']:<10} {drivers_str:<12} {vehicles_str:<12} {depot_str:<10}")
    
    print("=" * 70)


if __name__ == "__main__":
    print_summary()
    
    print("\nLoading test cases...")
    cases = get_test_cases()
    
    for size, case in cases.items():
        print(f"\n[{case['name']}]")
        print(f"  - Areas: {len(case['areas'])}")
        print(f"  - Sample area: {case['areas'][0]}")
    
    print("\n✓ All test cases loaded from JSON successfully!")

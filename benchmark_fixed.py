# =============================================================================
# benchmark_fixed.py - Benchmark sử dụng Test Cases cố định từ JSON
# =============================================================================
"""
Chạy benchmark với các bộ test cases cố định để đảm bảo kết quả tái lập.

Usage: 
    python benchmark_fixed.py              # Chạy tất cả sizes
    python benchmark_fixed.py 50 100       # Chạy sizes cụ thể
"""

import time
import json
import sys

from test_cases import get_test_case, AVAILABLE_SIZES
from algorithms import (
    generate_initial_solution, calculate_objective,
    local_search, local_search_multi_swap,
    vns, vns_multi_swap, vns_overtime_priority
)


def run_benchmark(num_areas: int) -> dict:
    """Chạy benchmark cho một test case cố định"""
    
    # Load test case từ JSON
    case = get_test_case(num_areas)
    
    areas = case['areas']
    drivers = case['drivers']
    vehicles = case['vehicles']
    depot = case['depot']
    num_districts = case['num_districts']
    map_size = case['map_size']
    params = case['params']
    
    print(f"\n{'='*60}")
    print(f"[{case['name']}]")
    print(f"Districts: {num_districts}, Drivers: {params['fulltime']}FT+{params['parttime']}PT")
    print(f"Vehicles: {params['large_vehicles']}L/{params['medium_vehicles']}M/{params['small_vehicles']}S")
    print(f"Map: {map_size}x{map_size}, Depot: ({depot['x']}, {depot['y']})")
    print(f"{'='*60}")
    
    # Tạo lời giải ban đầu
    initial = generate_initial_solution(num_areas, num_districts, areas)
    initial_score = calculate_objective(initial, areas, num_districts, drivers, vehicles, depot, map_size)
    print(f"Initial score: {initial_score:.4f}")
    
    results = {
        'num_areas': num_areas,
        'num_districts': num_districts,
        'map_size': map_size,
        'drivers': {'fulltime': params['fulltime'], 'parttime': params['parttime']},
        'vehicles': {'large': params['large_vehicles'], 'medium': params['medium_vehicles'], 'small': params['small_vehicles']},
        'initial_score': float(initial_score),
        'algorithms': {}
    }
    
    # Định nghĩa các thuật toán test
    algorithms = [
        ('local_search', lambda: local_search(initial, areas, num_districts, drivers, vehicles, depot, map_size)),
        ('ls_multi_swap', lambda: local_search_multi_swap(initial, areas, num_districts, drivers, vehicles, depot, map_size)),
        ('vns_multi_swap', lambda: vns_multi_swap(initial, areas, num_districts, drivers, vehicles, depot, map_size)),
        ('vns_overtime', lambda: vns_overtime_priority(initial, areas, num_districts, drivers, vehicles, depot, map_size)),
    ]
    
    # Thêm VNS thường cho sizes nhỏ
    if num_areas <= 500:
        algorithms.insert(2, ('vns', lambda: vns(initial, areas, num_districts, drivers, vehicles, depot, map_size)))
    
    print(f"\n{'Algorithm':<20} {'Score':<12} {'Improvement':<12} {'Time':<10}")
    print("-" * 54)
    
    for name, algo_func in algorithms:
        start = time.time()
        solution, score = algo_func()
        elapsed = time.time() - start
        
        improvement = (initial_score - score) / initial_score * 100
        print(f"{name:<20} {score:<12.4f} {improvement:>10.1f}% {elapsed:>8.1f}s")
        
        results['algorithms'][name] = {
            'score': float(score),
            'time': round(elapsed, 2),
            'improvement': round(improvement, 1)
        }
    
    return results


def main():
    # Xử lý arguments
    if len(sys.argv) > 1:
        sizes = [int(s) for s in sys.argv[1:]]
        # Validate
        for s in sizes:
            if s not in AVAILABLE_SIZES:
                print(f"Error: Size {s} not available. Choose from: {AVAILABLE_SIZES}")
                return
    else:
        sizes = AVAILABLE_SIZES
    
    print("=" * 60)
    print("BENCHMARK WITH FIXED TEST CASES (JSON)")
    print(f"Sizes to test: {sizes}")
    print("=" * 60)
    
    all_results = []
    
    for size in sizes:
        try:
            result = run_benchmark(size)
            all_results.append(result)
        except Exception as e:
            print(f"Error testing {size}: {e}")
            import traceback
            traceback.print_exc()
    
    # Lưu kết quả
    output_file = 'benchmark_results_fixed.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # In summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Areas':<8} {'Initial':<10} {'Best Score':<12} {'Best Algo':<18} {'Improvement':<12}")
    print("-" * 60)
    
    for r in all_results:
        best_algo = min(r['algorithms'].items(), key=lambda x: x[1]['score'])
        print(f"{r['num_areas']:<8} {r['initial_score']:<10.4f} {best_algo[1]['score']:<12.4f} {best_algo[0]:<18} {best_algo[1]['improvement']:>10.1f}%")
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()

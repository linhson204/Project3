# =============================================================================
# algorithms.py - Thuật toán tối ưu: Local Search, VNS (OPTIMIZED)
# =============================================================================

import random
import numpy as np
import math
from models import T_MAX, TRAVEL_SPEED, RELOAD_TIME
from district_utils import (
    build_neighbor_graph, calculate_centers, is_contiguous, 
    repair_contiguity, assign_resources
)

# =============================================================================
# ADAPTIVE PARAMETERS - Tự điều chỉnh theo kích thước bài toán
# =============================================================================

def get_adaptive_params(num_areas):
    if num_areas <= 100:
        return {'max_iter': 5, 'k_max': 6, 'ls_iter': 100, 'boundary_limit': 100}
    elif num_areas <= 200:
        return {'max_iter': 4, 'k_max': 5, 'ls_iter': 120, 'boundary_limit': 120}
    elif num_areas <= 300:
        return {'max_iter': 3, 'k_max': 5, 'ls_iter': 140, 'boundary_limit': 140}
    elif num_areas <= 500:
        return {'max_iter': 3, 'k_max': 5, 'ls_iter': 160, 'boundary_limit': 160}
    elif num_areas <= 800:
        return {'max_iter': 2, 'k_max': 5, 'ls_iter': 160, 'boundary_limit': 160}
    elif num_areas <= 1000:
        return {'max_iter': 2, 'k_max': 5, 'ls_iter': 170, 'boundary_limit': 170}
    else:  # > 1000
        return {'max_iter': 2, 'k_max': 5, 'ls_iter': 180, 'boundary_limit': 180}

# =============================================================================
# OPTIMIZED CACHING 
# =============================================================================

def precompute_area_positions(areas):
    return np.array([[a['x'], a['y']] for a in areas])

# =============================================================================
# OBJECTIVE FUNCTION 
# =============================================================================

def calculate_district_times(solution, areas, assignments, num_districts, depot, area_positions=None):
    centers = calculate_centers(solution, areas, num_districts)
    times = {}
    depot_pos = np.array([depot['x'], depot['y']])
    
    # Pre-group areas by district
    district_areas = {d: [] for d in range(num_districts)}
    for idx, d in enumerate(solution):
        district_areas[d].append(idx)
    
    for district_id in range(num_districts):
        weight = assignments[district_id]['weight']
        capacity = assignments[district_id]['vehicle']['capacity']
        num_trips = math.ceil(weight / capacity) if capacity > 0 else 1
        
        center = centers[district_id]
        dist_to_depot = np.linalg.norm(depot_pos - center)
        
        # Vectorized distance calculation
        indices = district_areas[district_id]
        if indices:
            if area_positions is not None:
                positions = area_positions[indices]
            else:
                positions = np.array([[areas[i]['x'], areas[i]['y']] for i in indices])
            dist_within = np.sum(np.linalg.norm(positions - center, axis=1))
            service_time = sum(areas[i]['service_time'] for i in indices)
        else:
            dist_within = 0
            service_time = 0
        
        total_dist = dist_to_depot + dist_within
        if num_trips > 1:
            total_dist += dist_to_depot * (num_trips - 1) * 2
        
        travel_time = total_dist / TRAVEL_SPEED
        reload_time = (num_trips - 1) * RELOAD_TIME if num_trips > 1 else 0
        
        times[district_id] = travel_time + reload_time + service_time
    
    return times


def calculate_objective(solution, areas, num_districts, drivers, vehicles, depot, map_size, area_positions=None):
    stats = [{'service_time': 0, 'parcels': 0, 'weight': 0, 'id': i, 'area_count': 0} 
             for i in range(num_districts)]
    
    for idx, dist_id in enumerate(solution):
        stats[dist_id]['service_time'] += areas[idx]['service_time']
        stats[dist_id]['parcels'] += areas[idx]['parcels']
        stats[dist_id]['weight'] += areas[idx]['weight']
        stats[dist_id]['area_count'] += 1
    
    assignments, valid = assign_resources(stats, drivers, vehicles)
    if not valid:
        return 1000000
    
    centers = calculate_centers(solution, areas, num_districts)
    times = calculate_district_times(solution, areas, assignments, num_districts, depot, area_positions)
    
    # Balance score
    util_rates = [
        stats[i]['service_time'] / assignments[i]['driver']['work_time']
        for i in range(num_districts) if assignments[i]['driver']['work_time'] > 0
    ]
    mean_util = np.mean(util_rates) if util_rates else 0
    balance = sum((r - mean_util) ** 2 for r in util_rates) / num_districts if num_districts > 0 else 0
    
    # Compactness score - VECTORIZED
    if area_positions is not None:
        district_centers = np.array([centers[solution[i]] for i in range(len(areas))])
        compactness = np.sum(np.linalg.norm(area_positions - district_centers, axis=1))
    else:
        compactness = sum(
            np.linalg.norm(np.array([areas[idx]['x'], areas[idx]['y']]) - centers[dist_id])
            for idx, dist_id in enumerate(solution)
        )
    
    # Time penalty
    time_penalty = 0
    time_util = []
    for i in range(num_districts):
        limit = assignments[i]['driver']['work_time']
        actual = times[i]
        time_util.append(actual / limit if limit > 0 else 0)
        
        if actual > limit:
            excess = (actual - limit) / limit
            time_penalty += excess * 10000
    
    mean_time_util = np.mean(time_util) if time_util else 0
    balance_time = sum((r - mean_time_util) ** 2 for r in time_util) / num_districts * 10 if num_districts > 0 else 0
    
    # Normalize
    mean_work = np.mean([d['work_time'] for d in drivers]) if drivers else 1
    norm_balance = balance / (mean_work ** 2) if mean_work > 0 else 0
    norm_compact = compactness / (len(areas) * map_size) if len(areas) * map_size > 0 else 0
    
    return 0.3 * norm_balance + 0.3 * balance_time + 0.3 * norm_compact + 0.1 * time_penalty


# =============================================================================
# SOLUTION GENERATORS
# =============================================================================

def generate_initial_solution(num_areas, num_districts, areas):
    """Tạo lời giải ban đầu theo grid"""
    sorted_idx = sorted(range(num_areas), key=lambda i: (areas[i]['x'] // 10, areas[i]['y'] // 10))
    solution = [-1] * num_areas
    per_district = num_areas // num_districts
    extra = num_areas % num_districts
    
    curr = 0
    for d in range(num_districts):
        count = per_district + (1 if d < extra else 0)
        for _ in range(count):
            solution[sorted_idx[curr]] = d
            curr += 1
    
    return solution


# =============================================================================
# LOCAL SEARCH 
# =============================================================================

def local_search(solution, areas, num_districts, drivers, vehicles, depot, map_size, 
                 use_contiguity=True, graph=None, area_positions=None, max_iter=None):
    """Local Search tập trung vào boundary areas"""
    num_areas = len(areas)
    params = get_adaptive_params(num_areas)
    
    if max_iter is None:
        max_iter = params['ls_iter']
    
    # Pre-compute
    if area_positions is None:
        area_positions = precompute_area_positions(areas)
    if graph is None and use_contiguity:
        graph = build_neighbor_graph(areas)
    
    current = list(solution)
    current_score = calculate_objective(current, areas, num_districts, drivers, vehicles, depot, map_size, area_positions)
    best, best_score = list(current), current_score
    
    boundary_limit = params['boundary_limit']
    
    for iteration in range(max_iter):
        improved = False
        
        # Tìm boundary areas 
        boundary = []
        if graph:
            for idx in range(num_areas):
                for n in graph[idx]:
                    if current[n] != current[idx]:
                        boundary.append(idx)
                        break
                if len(boundary) >= boundary_limit:
                    break
        
        # Shuffle để đa dạng hóa
        random.shuffle(boundary)
        
        # Thử di chuyển (giới hạn số lần thử)
        max_tries = min(len(boundary), boundary_limit // 2)
        for idx in boundary[:max_tries]:
            curr_dist = current[idx]
            neighbor_dists = {current[n] for n in graph[idx]} - {curr_dist} if graph else set()
            
            for new_dist in neighbor_dists:
                test = list(current)
                test[idx] = new_dist
                
                if use_contiguity and graph:
                    if not is_contiguous(test, graph, curr_dist) or not is_contiguous(test, graph, new_dist):
                        continue
                
                score = calculate_objective(test, areas, num_districts, drivers, vehicles, depot, map_size, area_positions)
                if score < current_score:
                    current, current_score = test, score
                    improved = True
                    break
            
            if improved:
                break
        
        if current_score < best_score:
            best, best_score = list(current), current_score
        
        if not improved:
            break
    
    # Repair contiguity nếu cần
    if use_contiguity and graph:
        repaired = repair_contiguity(best, graph, num_districts)
        repaired_score = calculate_objective(repaired, areas, num_districts, drivers, vehicles, depot, map_size, area_positions)
        if repaired_score <= best_score:
            best, best_score = repaired, repaired_score
    
    return best, best_score


def local_search_multi_swap(solution, areas, num_districts, drivers, vehicles, depot, map_size, 
                            use_contiguity=True, swap_size=2):
    """Local Search với Multi-Swap strategy - hoán đổi nhiều khu vực giữa 2 quận"""
    num_areas = len(areas)
    params = get_adaptive_params(num_areas)
    
    # Pre-compute
    area_positions = precompute_area_positions(areas)
    graph = build_neighbor_graph(areas) if use_contiguity else None
    
    current = list(solution)
    current_score = calculate_objective(current, areas, num_districts, drivers, vehicles, depot, map_size, area_positions)
    initial_score = current_score
    best, best_score = list(current), current_score
    
    max_iter = params['ls_iter']
    
    # Số lần thử scale theo số khu vực
    single_move_tries = max(250, num_areas // 3)  # Tối thiểu 50, tăng theo num_areas
    swap_pair_tries = max(250, num_areas // 3)    # Số cặp quận thử swap
    
    for iteration in range(max_iter):
        improved = False
        
        # Single move strategy - số lần thử scale theo num_areas
        for _ in range(single_move_tries):
            area_idx = random.randint(0, num_areas - 1)
            curr_dist = current[area_idx]
            new_dist = random.randint(0, num_districts - 1)
            
            if new_dist == curr_dist:
                continue
            
            test = list(current)
            test[area_idx] = new_dist
            
            if use_contiguity and graph:
                if not is_contiguous(test, graph, curr_dist) or not is_contiguous(test, graph, new_dist):
                    continue
            
            score = calculate_objective(test, areas, num_districts, drivers, vehicles, depot, map_size, area_positions)
            
            if score < current_score:
                current, current_score = test, score
                improved = True
                break
        
        if improved:
            if current_score < best_score:
                best, best_score = list(current), current_score
            continue
        
        # Multi-Swap strategy - số cặp thử scale theo num_areas
        district_pairs = [(a, b) for a in range(num_districts) for b in range(a + 1, num_districts)]
        random.shuffle(district_pairs)
        
        for district_a, district_b in district_pairs[:swap_pair_tries]:
            areas_a = [i for i, d in enumerate(current) if d == district_a]
            areas_b = [i for i, d in enumerate(current) if d == district_b]
            
            if not areas_a or not areas_b:
                continue
            
            for _ in range(3):
                sample_size = min(swap_size, len(areas_a), len(areas_b))
                sample_a = random.sample(areas_a, sample_size)
                sample_b = random.sample(areas_b, sample_size)
                
                test = list(current)
                for idx in sample_a:
                    test[idx] = district_b
                for idx in sample_b:
                    test[idx] = district_a
                
                if use_contiguity and graph:
                    if not is_contiguous(test, graph, district_a) or not is_contiguous(test, graph, district_b):
                        continue
                
                score = calculate_objective(test, areas, num_districts, drivers, vehicles, depot, map_size, area_positions)
                
                if score < current_score:
                    current, current_score = test, score
                    improved = True
                    break
            
            if improved:
                break
        
        if current_score < best_score:
            best, best_score = list(current), current_score
        
        if not improved:
            break
    
    # Repair contiguity
    if use_contiguity and graph:
        repaired = repair_contiguity(best, graph, num_districts)
        repaired_score = calculate_objective(repaired, areas, num_districts, drivers, vehicles, depot, map_size, area_positions)
        if repaired_score <= best_score:
            best, best_score = repaired, repaired_score
    
    if best_score > initial_score:
        return list(solution), initial_score
    
    return best, best_score


# =============================================================================
# VNS (Variable Neighborhood Search) 
# =============================================================================

def shake(solution, k, num_districts):
    """Shake solution by moving random areas"""
    new_sol = list(solution)
    
    for _ in range(k):
        district = random.randint(0, num_districts - 1)
        areas_in_dist = [i for i, d in enumerate(new_sol) if d == district]
        
        if not areas_in_dist:
            continue
        
        num_move = max(1, int(len(areas_in_dist) * 0.15)) 
        to_move = random.sample(areas_in_dist, min(num_move, len(areas_in_dist)))
        
        for idx in to_move:
            new_dist = random.randint(0, num_districts - 1)
            while new_dist == district and num_districts > 1:
                new_dist = random.randint(0, num_districts - 1)
            new_sol[idx] = new_dist
    
    return new_sol


def vns(solution, areas, num_districts, drivers, vehicles, depot, map_size, max_iter=None, k_max=None, use_contiguity=True):
    """Variable Neighborhood Search"""
    params = get_adaptive_params(len(areas))
    if max_iter is None:
        max_iter = params['max_iter']
    if k_max is None:
        k_max = params['k_max']
    
    # Pre-compute
    area_positions = precompute_area_positions(areas)
    graph = build_neighbor_graph(areas) if use_contiguity else None
    
    best, best_score = local_search(solution, areas, num_districts, drivers, vehicles, depot, map_size, 
                                     use_contiguity, graph, area_positions)
    
    for _ in range(max_iter):
        k = 1
        while k <= k_max:
            shaken = shake(best, k, num_districts)
            
            if use_contiguity and graph:
                shaken = repair_contiguity(shaken, graph, num_districts)
            
            improved, score = local_search(shaken, areas, num_districts, drivers, vehicles, depot, map_size, 
                                           use_contiguity, graph, area_positions)
            
            if score < best_score:
                best, best_score = improved, score
                k = 1
            else:
                k += 1
    
    return best, best_score


def vns_multi_swap(solution, areas, num_districts, drivers, vehicles, depot, map_size, max_iter=None, k_max=None, use_contiguity=True):
    """VNS với Local Search Multi-Swap"""
    params = get_adaptive_params(len(areas))
    if max_iter is None:
        max_iter = params['max_iter']
    if k_max is None:
        k_max = params['k_max']
    
    # Pre-compute
    area_positions = precompute_area_positions(areas)
    graph = build_neighbor_graph(areas) if use_contiguity else None
    
    best, best_score = local_search_multi_swap(solution, areas, num_districts, drivers, vehicles, depot, map_size, use_contiguity)
    
    for _ in range(max_iter):
        k = 1
        while k <= k_max:
            shaken = shake(best, k, num_districts)
            
            if use_contiguity and graph:
                shaken = repair_contiguity(shaken, graph, num_districts)
            
            improved, score = local_search_multi_swap(shaken, areas, num_districts, drivers, vehicles, depot, map_size, use_contiguity)
            
            if score < best_score:
                best, best_score = improved, score
                k = 1
            else:
                k += 1
    
    return best, best_score


def vns_overtime_priority(solution, areas, num_districts, drivers, vehicles, depot, map_size, max_iter=None, k_max=None, use_contiguity=True):
    """VNS với ưu tiên chuyển từ overtime districts sang undertime districts - OPTIMIZED"""
    params = get_adaptive_params(len(areas))
    if max_iter is None:
        max_iter = params['max_iter']
    if k_max is None:
        k_max = params['k_max']
    
    # Pre-compute
    area_positions = precompute_area_positions(areas)
    graph = build_neighbor_graph(areas) if use_contiguity else None
    
    best, best_score = local_search_multi_swap(solution, areas, num_districts, drivers, vehicles, depot, map_size, 
                                     use_contiguity)
    
    for _ in range(max_iter):
        k = 1
        while k <= k_max:
            # Identify overtime/undertime districts
            stats = [{'service_time': 0, 'parcels': 0, 'weight': 0, 'id': i} for i in range(num_districts)]
            for idx, d in enumerate(best):
                stats[d]['service_time'] += areas[idx]['service_time']
                stats[d]['parcels'] += areas[idx]['parcels']
                stats[d]['weight'] += areas[idx]['weight']
            
            assignments, valid = assign_resources(stats, drivers, vehicles)
            if not valid:
                shaken = shake(best, k, num_districts)
            else:
                times = calculate_district_times(best, areas, assignments, num_districts, depot, area_positions)
                overtime = [d for d in range(num_districts) if times[d] > T_MAX]
                undertime = {d for d in range(num_districts) if times[d] <= T_MAX}
                
                if overtime and undertime:
                    shaken = shake_overtime(best, k, areas, overtime, undertime, graph, times)
                else:
                    shaken = shake(best, k, num_districts)
            
            if use_contiguity and graph:
                shaken = repair_contiguity(shaken, graph, num_districts)
            
            improved, score = local_search_multi_swap(shaken, areas, num_districts, drivers, vehicles, depot, map_size, 
                                           use_contiguity)
            
            if score < best_score:
                best, best_score = improved, score
                k = 1
            else:
                k += 1
    
    return best, best_score


def shake_overtime(solution, k, areas, overtime_dists, undertime_dists, graph, times):
    """Shake ưu tiên chuyển từ overtime sang undertime districts lân cận"""
    new_sol = list(solution)
    targets = random.sample(overtime_dists, min(k, len(overtime_dists)))

    for dist in targets:
        dist_areas = [i for i, d in enumerate(solution) if d == dist]
        if not dist_areas:
            continue
        
        # Tính num_move dựa trên mức độ overtime
        actual_time = times[dist]
        overtime_ratio = (actual_time - T_MAX) / actual_time if actual_time > T_MAX else 0
        
        # Số areas cần chuyển tỷ lệ với overtime_ratio
        # Tối thiểu 1, tối đa 50% số areas của district
        num_move = max(1, min(
            math.ceil(len(dist_areas) * overtime_ratio * 1.15),  # Nhân 1.15 để chắc chắn giảm đủ
            len(dist_areas) // 2  # Không chuyển quá nửa
        ))
        
        sorted_areas = sorted(dist_areas, key=lambda i: areas[i]['service_time'], reverse=True)
        
        for idx in sorted_areas[:num_move]:
            if graph:
                neighbors = {solution[n] for n in graph[idx] if solution[n] != dist}
                valid = [d for d in neighbors if d in undertime_dists]
                
                if valid:
                    valid.sort(key=lambda d: times[d])
                    new_sol[idx] = valid[0]
                    times[valid[0]] += areas[idx]['service_time']
                    if times[valid[0]] > T_MAX:
                        undertime_dists.discard(valid[0])
    
    return new_sol

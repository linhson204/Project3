# =============================================================================
# district_utils.py - Utility functions cho districts
# =============================================================================

import numpy as np
from collections import Counter

def build_neighbor_graph(areas: list, threshold: float = 7.5) -> dict:
    """Xây dựng đồ thị lân cận NG dựa trên khoảng cách"""
    n = len(areas)
    graph = {i: [] for i in range(n)}
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((areas[i]['x'] - areas[j]['x'])**2 + 
                          (areas[i]['y'] - areas[j]['y'])**2)
            if dist <= threshold:
                graph[i].append(j)
                graph[j].append(i)
    
    return graph


def calculate_centers(solution: list, areas: list, num_districts: int) -> np.ndarray:
    """Tính tâm của mỗi district"""
    centers = np.zeros((num_districts, 2))
    counts = np.zeros(num_districts)
    
    for idx, district_id in enumerate(solution):
        centers[district_id] += [areas[idx]['x'], areas[idx]['y']]
        counts[district_id] += 1
    
    counts[counts == 0] = 1
    return centers / counts[:, np.newaxis]


def is_contiguous(solution: list, neighbor_graph: dict, district_id: int) -> bool:
    """Kiểm tra district có liền mạch không (BFS)"""
    district_areas = [i for i, d in enumerate(solution) if d == district_id]
    if len(district_areas) <= 1:
        return True
    
    visited = {district_areas[0]}
    queue = [district_areas[0]]
    
    while queue:
        current = queue.pop(0)
        for neighbor in neighbor_graph[current]:
            if neighbor in district_areas and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited) == len(district_areas)


def repair_contiguity(solution: list, neighbor_graph: dict, num_districts: int, max_iter: int = 100) -> list:
    """Sửa chữa các district không liền mạch"""
    solution = list(solution)
    
    for _ in range(max_iter):
        all_ok = True
        
        for district_id in range(num_districts):
            if is_contiguous(solution, neighbor_graph, district_id):
                continue
            
            all_ok = False
            district_areas = [i for i, d in enumerate(solution) if d == district_id]
            if not district_areas:
                continue
            
            # BFS từ area đầu tiên
            visited = {district_areas[0]}
            queue = [district_areas[0]]
            while queue:
                current = queue.pop(0)
                for neighbor in neighbor_graph[current]:
                    if neighbor in district_areas and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Di chuyển các area bị cô lập sang district lân cận
            for area in district_areas:
                if area not in visited:
                    neighbors = [solution[n] for n in neighbor_graph[area] if solution[n] != district_id]
                    if neighbors:
                        solution[area] = Counter(neighbors).most_common(1)[0][0]
        
        if all_ok:
            break
    
    return solution


def assign_resources(district_stats: list, drivers: list, vehicles: list) -> tuple:
    """
    Gán tài nguyên với strict priority constraint.
    Driver/Vehicle type d chỉ được dùng khi tất cả type d' < d đã dùng hết.
    """
    sorted_districts = sorted(district_stats, key=lambda x: x['service_time'], reverse=True)
    sorted_drivers = sorted(drivers, key=lambda x: (x['type_id'], -x['work_time']))
    sorted_vehicles = sorted(vehicles, key=lambda x: (x['type_id'], -x['capacity']))
    
    # Đếm số lượng theo type
    driver_count = {}
    for d in drivers:
        driver_count[d['type_id']] = driver_count.get(d['type_id'], 0) + 1
    
    vehicle_count = {}
    for v in vehicles:
        vehicle_count[v['type_id']] = vehicle_count.get(v['type_id'], 0) + 1
    
    assignments = {}
    used_drivers, used_vehicles = [], []
    used_driver_by_type, used_vehicle_by_type = {}, {}
    
    for district in sorted_districts:
        dist_id = district['id']
        
        # Tìm driver với strict priority
        driver = None
        for d in sorted_drivers:
            if d['id'] in used_drivers:
                continue
            
            can_use = all(
                used_driver_by_type.get(t, 0) >= driver_count.get(t, 0)
                for t in range(1, d['type_id'])
            )
            
            if can_use:
                driver = d
                used_drivers.append(d['id'])
                used_driver_by_type[d['type_id']] = used_driver_by_type.get(d['type_id'], 0) + 1
                break
        
        # Tìm vehicle với strict priority
        vehicle = None
        for v in sorted_vehicles:
            if v['id'] in used_vehicles:
                continue
            
            can_use = all(
                used_vehicle_by_type.get(t, 0) >= vehicle_count.get(t, 0)
                for t in range(1, v['type_id'])
            )
            
            if can_use:
                vehicle = v
                used_vehicles.append(v['id'])
                used_vehicle_by_type[v['type_id']] = used_vehicle_by_type.get(v['type_id'], 0) + 1
                break
        
        if driver is None or vehicle is None:
            return None, False
        
        assignments[dist_id] = {
            'driver': driver,
            'vehicle': vehicle,
            'service_time': float(district['service_time']),
            'parcels': int(district['parcels']),
            'weight': float(district['weight'])
        }
    
    return assignments, True

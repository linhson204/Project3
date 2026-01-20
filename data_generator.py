# =============================================================================
# data_generator.py - Tạo dữ liệu Areas, Drivers, Vehicles
# =============================================================================

import random
from models import T_MAX, NUM_SAMPLE_DAYS, SERVICE_TIME_PER_PARCEL

def generate_areas(num_areas: int, map_size: int, num_sample_days: int = NUM_SAMPLE_DAYS) -> list:
    """ Tạo danh sách basic areas B với demand data cho tập sample days T. """
    areas = []
    random.seed(42)
    
    for i in range(num_areas):
        x, y = random.randint(0, map_size), random.randint(0, map_size)
        base_parcels = random.randint(5, 10)
        weight_per_parcel = round(random.uniform(0.2, 0.5), 3)
        
        # Daily demand với fluctuation ±30%
        daily_demand = {}
        for day in range(num_sample_days):
            fluctuation = random.uniform(0.7, 1.3)
            parcels = max(1, int(base_parcels * fluctuation))
            daily_demand[day] = {
                'parcels': parcels,
                'weight': round(parcels * weight_per_parcel, 2),
                'service_time': round(parcels * SERVICE_TIME_PER_PARCEL, 3)
            }
        
        # Aggregate values
        avg = lambda key: sum(d[key] for d in daily_demand.values()) / num_sample_days
        max_val = lambda key: max(d[key] for d in daily_demand.values())
        
        areas.append({
            'id': i, 'x': x, 'y': y,
            'daily_demand': daily_demand,
            'parcels': int(round(avg('parcels'))),
            'weight': round(avg('weight'), 2),
            'service_time': round(avg('service_time'), 3),
            'max_parcels': max_val('parcels'),
            'max_weight': round(max_val('weight'), 2),
            'max_service_time': round(max_val('service_time'), 3)
        })
    
    random.seed()
    return areas


def create_drivers(num_fulltime: int, num_parttime: int) -> list:
    """Tạo danh sách drivers với priority theo đề bài (d1 < d2 = d1 ưu tiên hơn)"""
    drivers = []
    driver_id = 0
    prefixes = ['Anh', 'Chị']
    
    # Full-time drivers (type_id = 1, highest priority)
    for i in range(num_fulltime):
        drivers.append({
            'id': driver_id,
            'name': f'{prefixes[i % 2]} {driver_id + 1}',
            'type': 'Full-time',
            'type_id': 1,
            'r_d': 1.0,
            'work_time': T_MAX
        })
        driver_id += 1
    
    # Part-time drivers (type_id = 2+, lower priority)
    ratios = [0.75, 0.625, 0.5]
    for i in range(num_parttime):
        ratio = ratios[i % len(ratios)]
        drivers.append({
            'id': driver_id,
            'name': f'{prefixes[i % 2]} {driver_id + 1}',
            'type': f'Part-time ({int(ratio*100)}%)',
            'type_id': 2 + (i % len(ratios)),
            'r_d': ratio,
            'work_time': T_MAX * ratio
        })
        driver_id += 1
    
    return drivers


def create_vehicles(num_large: int, num_medium: int, num_small: int) -> list:
    """Tạo danh sách vehicles với priority (type_id nhỏ = ưu tiên cao)"""
    vehicles = []
    vehicle_id = 0
    
    configs = [
        (num_large, 'Xe máy to', 1, 20),
        (num_medium, 'Xe máy vừa', 2, 15),
        (num_small, 'Xe máy nhỏ', 3, 10)
    ]
    
    for count, vtype, type_id, capacity in configs:
        for _ in range(count):
            vehicles.append({
                'id': vehicle_id,
                'type': vtype,
                'type_id': type_id,
                'capacity': capacity
            })
            vehicle_id += 1
    
    return vehicles

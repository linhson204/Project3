# =============================================================================
# api.py - FastAPI Application (Refactored)
# =============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
import math
import warnings
warnings.filterwarnings('ignore')

# Import từ các modules
from models import OptimizationRequest, OptimizationResponse
from data_generator import generate_areas, create_drivers, create_vehicles
from district_utils import assign_resources
from algorithms import (
    generate_initial_solution, calculate_objective, calculate_district_times,
    local_search, local_search_multi_swap, vns, vns_multi_swap, vns_overtime_priority
)
from visualization import create_plot

# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(title="Delivery Route Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def read_root():
    """Serve index.html"""
    index_path = BASE_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/test")
async def test_server():
    """Health check endpoint"""
    return {
        "status": "success",
        "message": "Server đang hoạt động!",
        "endpoints": ["GET /", "GET /test", "POST /optimize", "GET /docs"]
    }


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize(request: OptimizationRequest):
    """Main optimization endpoint"""
    try:
        # Chuẩn bị dữ liệu
        depot = {'x': request.map_size // 2, 'y': request.map_size // 2}
        areas = generate_areas(request.num_areas, request.map_size, request.num_sample_days)
        drivers = create_drivers(request.num_fulltime_drivers, request.num_parttime_drivers)
        vehicles = create_vehicles(request.num_large_vehicles, request.num_medium_vehicles, request.num_small_vehicles)
        
        total_drivers = request.num_fulltime_drivers + request.num_parttime_drivers
        total_vehicles = request.num_large_vehicles + request.num_medium_vehicles + request.num_small_vehicles
        num_districts = min(request.num_districts, total_drivers, total_vehicles)
        
        # Tạo lời giải ban đầu
        initial = generate_initial_solution(request.num_areas, num_districts, areas)
        initial_score = calculate_objective(initial, areas, num_districts, drivers, vehicles, depot, request.map_size)
        
        # Chạy thuật toán
        algorithms = {
            "random": lambda: (initial, initial_score),
            "ls": lambda: local_search(initial, areas, num_districts, drivers, vehicles, depot, request.map_size),
            "ls_multi_swap": lambda: local_search_multi_swap(initial, areas, num_districts, drivers, vehicles, depot, request.map_size),
            "vns": lambda: vns(initial, areas, num_districts, drivers, vehicles, depot, request.map_size),
            "vns_multi_swap": lambda: vns_multi_swap(initial, areas, num_districts, drivers, vehicles, depot, request.map_size),
            "vns_overtime": lambda: vns_overtime_priority(initial, areas, num_districts, drivers, vehicles, depot, request.map_size),
        }
        
        algo_names = {
            "random": "Random",
            "ls": "Local Search", 
            "ls_multi_swap": "Local Search Multi-Swap",
            "vns": "VNS",
            "vns_multi_swap": "VNS Multi-Swap",
            "vns_overtime": "VNS Overtime Priority"
        }
        
        if request.algorithm not in algorithms:
            raise HTTPException(status_code=400, detail=f"Invalid algorithm: {request.algorithm}")
        
        best, best_score = algorithms[request.algorithm]()
        algo_name = algo_names[request.algorithm]
        
        # Thống kê districts
        stats = [{'service_time': 0, 'parcels': 0, 'weight': 0, 'id': i, 'area_count': 0} 
                 for i in range(num_districts)]
        
        for idx, dist_id in enumerate(best):
            stats[dist_id]['service_time'] += areas[idx]['service_time']
            stats[dist_id]['parcels'] += areas[idx]['parcels']
            stats[dist_id]['weight'] += areas[idx]['weight']
            stats[dist_id]['area_count'] += 1
        
        assignments, valid = assign_resources(stats, drivers, vehicles)
        if not valid:
            raise HTTPException(status_code=400, detail="Không đủ tài nguyên")
        
        times = calculate_district_times(best, areas, assignments, num_districts, depot)
        plot = create_plot(best, areas, num_districts, depot, f"{algo_name} - Score: {best_score:.4f}")
        
        # Tạo kết quả
        assignment_list = []
        for d_id in range(num_districts):
            assign = assignments[d_id]
            driver = assign['driver']
            vehicle = assign['vehicle']
            total_time = float(times[d_id])  # Convert numpy to Python float
            
            num_trips = math.ceil(assign['weight'] / vehicle['capacity']) if vehicle['capacity'] > 0 else 1
            parcels_per_hour = assign['parcels'] / total_time if total_time > 0 else 0
            util_rate = (total_time / driver['work_time'] * 100) if driver['work_time'] > 0 else 0
            
            assignment_list.append({
                'district': d_id,
                'driver_name': driver['name'],
                'driver_type': driver['type'],
                'vehicle_type': vehicle['type'],
                'area_count': int(stats[d_id]['area_count']),
                'total_time': round(float(total_time), 2),
                'work_time': float(driver['work_time']),
                'parcels': int(assign['parcels']),
                'parcels_per_hour': round(float(parcels_per_hour), 1),
                'weight': round(float(assign['weight']), 1),
                'vehicle_capacity': int(vehicle['capacity']),
                'num_trips': int(num_trips),
                'utilization_rate': round(float(util_rate), 1),
                'overtime': bool(total_time > driver['work_time'])  # Convert numpy.bool to Python bool
            })
        
        # Summary
        total_service_time = sum(a['service_time'] for a in assignments.values())
        total_parcels = sum(a['parcels'] for a in assignments.values())
        total_weight = sum(a['weight'] for a in assignments.values())
        total_time_all = sum(times.values())
        
        improvement = ((initial_score - best_score) / initial_score * 100) if initial_score > 0 else 0
        
        return OptimizationResponse(
            success=True,
            algorithm_name=algo_name,
            score=round(best_score, 4),
            improvement_percent=round(improvement, 2),
            plot_base64=plot,
            assignments=assignment_list,
            summary={
                'total_service_time': round(total_service_time, 2),
                'total_parcels': total_parcels,
                'total_weight': round(total_weight, 1),
                'total_time': round(total_time_all, 2),
                'parcels_per_hour': round(total_parcels / total_time_all if total_time_all > 0 else 0, 1)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# =============================================================================
# models.py - Data Models & Constants cho Tactical Planning
# =============================================================================

from pydantic import BaseModel
from typing import List, Dict, Any

# =============================================================================
# CONSTANTS (Theo đề bài Tactical Planning)
# =============================================================================

TRAVEL_SPEED = 600  # 60km/h (đơn vị: 100m/h, map_size tính theo 100m)
RELOAD_TIME = 0.2  # t_reload: Thời gian reload tại depot (giờ)
T_MAX = 8.0  # t_max: Thời gian làm việc tối đa của full-time driver (giờ)
NUM_SAMPLE_DAYS = 5  # |T|: Số ngày mẫu mặc định
SERVICE_TIME_PER_PARCEL = 0.02  # s_o: Thời gian phục vụ trung bình (giờ) = 1.2 phút/bưu kiện

# =============================================================================
# API MODELS
# =============================================================================

class OptimizationRequest(BaseModel):
    """Request model cho API /optimize"""
    num_areas: int = 175
    map_size: int = 50
    num_fulltime_drivers: int = 3
    num_parttime_drivers: int = 2
    num_large_vehicles: int = 3
    num_medium_vehicles: int = 1
    num_small_vehicles: int = 1
    num_districts: int = 6
    num_sample_days: int = 5
    algorithm: str = "vns_overtime"

class OptimizationResponse(BaseModel):
    """Response model cho API /optimize"""
    success: bool
    algorithm_name: str
    score: float
    improvement_percent: float
    plot_base64: str
    assignments: List[Dict[str, Any]]
    summary: Dict[str, Any]

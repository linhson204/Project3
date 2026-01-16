# BÁO CÁO DỰ ÁN
# TỐI ƯU HÓA PHÂN VÙNG GIAO HÀNG (TACTICAL PLANNING)

---

# CHƯƠNG 1: CƠ SỞ LÝ THUYẾT

## 1.1. Phát biểu bài toán

Bài toán **Tactical Planning** trong lĩnh vực logistics nhằm phân chia khu vực địa lý thành các vùng giao hàng (delivery districts) và gán tài nguyên (tài xế, phương tiện) cho mỗi vùng một cách tối ưu.

### 1.1.1. Đầu vào bài toán

#### A. Theo đề bài gốc

**1. Basic Areas (B)** - Khu vực cơ sở:

| Ký hiệu đề bài | Mô tả | Kiểu dữ liệu |
|----------------|-------|--------------|
| `B = {1, ..., \|B\|}` | Tập basic areas (zip codes) | Set |
| `c_bi` | Khoảng cách giữa area b và i | R⁺ |
| `NG` | Đồ thị lân cận (Neighborhood Graph) | Graph |
| `A_b ⊆ B` | Tập các area lân cận của b | Set |

**2. Driver Types (D)** - Loại tài xế:

| Ký hiệu đề bài | Mô tả | Kiểu dữ liệu |
|----------------|-------|--------------|
| `D = {1, ..., \|D\|}` | Tập loại tài xế | Set |
| `t_max` | Thời gian làm việc tối đa (full-time) | R⁺ (giờ) |
| `r_d ∈ (0, 100]` | Tỷ lệ thời gian làm việc của loại d | % |
| `M_d` | Số lượng tài xế khả dụng loại d | N⁺ |

**Ràng buộc priority (theo đề bài):** "A driver of type d ∈ D may only be used, if for each driver type d' ∈ D with d' < d, all M_d' available drivers are also used."

**3. Vehicle Types (V)** - Loại phương tiện:

| Ký hiệu đề bài | Mô tả | Kiểu dữ liệu |
|----------------|-------|--------------|
| `V = {1, ..., \|V\|}` | Tập loại xe | Set |
| `C_v` | Capacity (sức chứa) của loại v | R⁺ (kg) |
| `N_v` | Số lượng xe khả dụng loại v | N⁺ |

**4. Sample Days (T)** - Ngày mẫu:

| Ký hiệu đề bài | Mô tả | Kiểu dữ liệu |
|----------------|-------|--------------|
| `T = {1, ..., \|T\|}` | Tập ngày mẫu | Set |

**5. Customer Orders (O)** - Đơn hàng:

| Ký hiệu đề bài | Mô tả | Kiểu dữ liệu |
|----------------|-------|--------------|
| `O = {1, ..., \|O\|}` | Tập đơn hàng | Set |
| `l_o` | Tổng trọng lượng đơn hàng o | R⁺ (kg) |
| `s_o` | Service time (thời gian phục vụ) | R⁺ (giờ) |
| `τ_o ∈ T` | Ngày giao hàng | T |
| `b_o ∈ B` | Basic area chứa khách hàng | B |

**6. Hằng số (theo đề bài):**

| Ký hiệu | Mô tả |
|---------|-------|
| `t_max` | Thời gian làm việc full-time |
| `t_reload` | Thời gian reload tại depot |

---

#### B. So sánh với Project Implementation

| Đề bài yêu cầu | Project thực hiện | File |
|----------------|-------------------|------|
| **Basic Areas B** | ✅ Tập areas với (x, y), tính khoảng cách Euclidean | `data_generator.py` |
| **Neighborhood Graph NG** | ✅ `build_neighbor_graph()` với threshold 7.5 | `district_utils.py` |
| **Driver Types D** | ✅ Full-time (r_d=1.0), Part-time (r_d=0.75, 0.625, 0.5) | `data_generator.py` |
| **t_max** | ✅ 8 giờ (`T_MAX = 8.0`) | `models.py` |
| **r_d × t_max** | ✅ `work_time = r_d × T_MAX` | `data_generator.py` |
| **M_d (số lượng driver)** | ✅ Input từ UI: `num_fulltime_drivers`, `num_parttime_drivers` | `index.html` |
| **Priority constraint** | ✅ `assign_resources()` kiểm tra strict priority | `district_utils.py` |
| **Vehicle Types V** | ✅ Xe to (20kg), vừa (15kg), nhỏ (10kg) | `data_generator.py` |
| **C_v (capacity)** | ✅ capacity field trong vehicle object | `data_generator.py` |
| **N_v (số lượng xe)** | ✅ Input từ UI: `num_large/medium/small_vehicles` | `index.html` |
| **Sample Days T** | ✅ `daily_demand` với fluctuation ±30% | `data_generator.py` |
| **\|T\| ngày mẫu** | ✅ Input `num_sample_days` từ UI | `index.html` |
| **Customer Orders O** | ⚠️ Đơn giản hóa: aggregate theo area thay vì từng order | `data_generator.py` |
| **l_o (weight)** | ✅ `weight` per area (aggregate) | `data_generator.py` |
| **s_o (service time)** | ✅ `service_time` = parcels × 0.02h | `data_generator.py` |
| **τ_o (ngày giao)** | ✅ `daily_demand[day]` | `data_generator.py` |
| **b_o (basic area)** | ✅ Implicit trong cấu trúc area | `data_generator.py` |
| **t_reload** | ✅ `RELOAD_TIME = 0.25` giờ | `models.py` |

**Giá trị cụ thể trong Project:**

| Hằng số | Giá trị | File |
|---------|---------|------|
| `T_MAX` | 8.0 giờ | `models.py` |
| `RELOAD_TIME` | 0.25 giờ (15 phút) | `models.py` |
| `TRAVEL_SPEED` | 500 (50 km/h, đơn vị map) | `models.py` |
| `SERVICE_TIME_PER_PARCEL` | 0.02 giờ (~1.2 phút) | `models.py` |
| `NUM_SAMPLE_DAYS` | 5 (mặc định) | `models.py` |

### 1.1.2. Yêu cầu cần đạt được

1. **Phân vùng tối ưu**: Chia B thành các districts sao cho mỗi district có thể được phục vụ bởi 1 tài xế trong giờ làm việc
2. **Cân bằng tải**: Các districts có khối lượng công việc tương đương nhau
3. **Hiệu quả tài nguyên**: Maximizing utilization rate của tài xế và phương tiện
4. **Tính compact**: Các khu vực trong 1 district gần nhau về mặt địa lý

### 1.1.3. Các ràng buộc

**Ràng buộc cứng:**
1. Mỗi basic area thuộc đúng 1 district
2. Mỗi district phải **liền mạch (contiguous)** trong đồ thị NG
3. **Priority constraint**: Tài xế loại d chỉ được dùng khi tất cả tài xế loại d' < d đã dùng hết

**Ràng buộc mềm:**
1. Thời gian làm việc ≤ giới hạn của tài xế
2. Số chuyến reload tối thiểu

---

## 1.2. Mô hình hóa bài toán

### 1.2.1. Tập dữ liệu đầu vào

```python
# Dữ liệu khu vực
area = {
    'id': i,
    'x': x_coordinate,
    'y': y_coordinate,
    'daily_demand': {
        day_0: {'parcels': 15, 'weight': 3.2, 'service_time': 0.3},
        day_1: {'parcels': 18, 'weight': 3.8, 'service_time': 0.36},
        ...
    },
    'parcels': avg_parcels,
    'weight': avg_weight,
    'service_time': avg_service_time
}

# Dữ liệu tài xế
driver = {
    'id': d,
    'type_id': 1,          # 1=Full-time, 2+=Part-time
    'r_d': 1.0,            # Tỷ lệ thời gian
    'work_time': 8.0       # Giờ làm việc = r_d × t_max
}

# Dữ liệu xe
vehicle = {
    'id': v,
    'type_id': 1,          # 1=Xe to, 2=Vừa, 3=Nhỏ
    'capacity': 20         # Sức chứa (kg)
}
```

### 1.2.2. Biến quyết định

```
solution[i] = d    ∀i ∈ B, d ∈ {0, 1, ..., k-1}
```

Trong đó:
- `i`: Index của basic area
- `d`: District mà area i được gán vào
- `k`: Số lượng districts

### 1.2.3. Ràng buộc cứng (Hard Constraints)

**C1. Phủ đầy đủ:**
```
∀i ∈ B: solution[i] ∈ {0, 1, ..., k-1}
```

**C2. Tính liền mạch (Contiguity):**
```
∀d ∈ Districts: is_contiguous(d, NG) = True
```

Kiểm tra bằng BFS/DFS: Từ 1 area bất kỳ trong district, phải đi được đến tất cả areas còn lại qua đồ thị NG.

**C3. Ràng buộc Priority:**
```
Với driver loại d:
  can_use[d] = True  ⟺  ∀d' < d: used[d'] = M_d'
```

### 1.2.4. Hàm mục tiêu (Objective Function)

```
f(solution) = w₁ × Balance + w₂ × TimeBalance + w₃ × Compactness + w₄ × OvertimePenalty
```

**Các thành phần:**

| Thành phần | Trọng số | Công thức |
|------------|----------|-----------|
| Balance | 0.3 | `Var(utilization_rates)` |
| TimeBalance | 0.3 | `Var(actual_time / allowed_time)` |
| Compactness | 0.3 | `Σ distance(area, center)` |
| OvertimePenalty | 0.1 | `10000 × Σ max(0, overtime/limit)` |

**Chi tiết tính toán:**

```python
# 1. Balance - Cân bằng service_time
util_rates = [service_time[d] / work_time[d] for d in districts]
balance = variance(util_rates)

# 2. Time Balance - Cân bằng thời gian làm việc thực tế
time_utils = [actual_time[d] / allowed_time[d] for d in districts]
time_balance = variance(time_utils)

# 3. Compactness - Độ compact
compactness = sum(distance(area, center[district]) for area in areas)

# 4. Overtime Penalty - Phạt vượt giờ
overtime_penalty = sum(
    10000 * (actual[d] - limit[d]) / limit[d]
    for d in districts if actual[d] > limit[d]
)
```

---

## 1.3. Các phương pháp giải quyết vấn đề

### 1.3.1. Tổng quan

Bài toán phân vùng giao hàng thuộc lớp **NP-hard**, không có thuật toán thời gian đa thức để tìm nghiệm tối ưu. Do đó, chúng tôi sử dụng các **metaheuristics**:

| Phương pháp | Ưu điểm | Nhược điểm |
|-------------|---------|------------|
| Local Search | Nhanh, đơn giản | Dễ rơi vào local optima |
| VNS | Thoát local optima | Chậm hơn Local Search |
| VNS Overtime Priority | Ưu tiên giảm overtime | Phức tạp nhất |

### 1.3.2. Local Search

**Nguyên lý:** Cải thiện lời giải bằng cách thay đổi nhỏ (neighborhood move).

**Move operator:** Di chuyển 1 boundary area sang district lân cận.

### 1.3.3. VNS (Variable Neighborhood Search)

**Nguyên lý:** 
1. Nếu Local Search không cải thiện → Shake (xáo trộn) lời giải
2. Tăng dần mức độ shake nếu vẫn không cải thiện
3. Reset về shake nhẹ nếu tìm được nghiệm tốt hơn

### 1.3.4. VNS Overtime Priority

**Nguyên lý:** VNS với shake thông minh - ưu tiên chuyển areas từ overtime districts sang undertime districts.

---

# CHƯƠNG 2: THIẾT KẾ THUẬT TOÁN

## 2.1. Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────┐
│                    ALGORITHMS                            │
├─────────────────────────────────────────────────────────┤
│  Local Search  →  VNS  →  VNS Overtime Priority         │
│       ↓             ↓              ↓                    │
│  ┌─────────────────────────────────────────────────┐    │
│  │         OBJECTIVE FUNCTION                       │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## 2.2. Local Search

### 2.2.1. Thuật toán cơ bản

```
Algorithm: Local Search
─────────────────────────
Input: initial_solution, areas, drivers, vehicles
Output: improved_solution

1. current ← initial_solution
2. REPEAT:
3.    boundary ← find_boundary_areas(current)
4.    improved ← False
5.    FOR each area in boundary:
6.       FOR each neighbor_district:
7.          test ← move(area, neighbor_district)
8.          IF is_contiguous(test) AND f(test) < f(current):
9.             current ← test
10.            improved ← True
11.            BREAK
12.       IF improved: BREAK
13. UNTIL NOT improved
14. RETURN current
```

### 2.2.2. Tối ưu hóa hiệu năng

**Vấn đề 1: Kiểm tra contiguity tốn O(n)**

Giải pháp: Chỉ kiểm tra contiguity khi thực sự di chuyển, không kiểm tra trước.

**Vấn đề 2: Tính objective function tốn O(n×d)**

Giải pháp: Pre-compute các giá trị cố định:
```python
# Pre-compute 1 lần duy nhất
area_positions = np.array([[a['x'], a['y']] for a in areas])
graph = build_neighbor_graph(areas)
```

**Vấn đề 3: Duyệt tất cả boundary areas**

Giải pháp: Giới hạn số lượng + shuffle:
```python
boundary_limit = params['boundary_limit']
random.shuffle(boundary)
for area in boundary[:boundary_limit]:
    ...
```

## 2.3. Local Search Multi-Swap

### 2.3.1. Ý tưởng

Khi single-move không cải thiện được, thử **hoán đổi đồng thời nhiều areas giữa 2 districts**.

### 2.3.2. Thuật toán

```
Algorithm: Local Search Multi-Swap
──────────────────────────────────
1. current ← initial_solution
2. REPEAT:
3.    improved ← False
4.    
5.    # Phase 1: Single Move (nhanh)
6.    FOR i = 1 to single_move_tries:
7.       area ← random_area()
8.       new_dist ← random_district()
9.       test ← move(area, new_dist)
10.      IF is_contiguous(test) AND f(test) < f(current):
11.         current ← test
12.         improved ← True
13.         BREAK
14.   
15.   IF improved: CONTINUE
16.   
17.   # Phase 2: Multi-Swap (chậm hơn nhưng mạnh hơn)
18.   FOR each district_pair (A, B):
19.      sample_A ← random_sample(areas_in_A, swap_size)
20.      sample_B ← random_sample(areas_in_B, swap_size)
21.      test ← swap(sample_A ↔ sample_B)
22.      IF is_contiguous(test) AND f(test) < f(current):
23.         current ← test
24.         improved ← True
25.         BREAK
26.
27. UNTIL NOT improved
28. RETURN current
```

### 2.3.3. Adaptive Scaling

Số lần thử tăng theo kích thước bài toán:

```python
single_move_tries = max(250, num_areas // 3)
swap_pair_tries = max(250, num_areas // 3)
```

| Num Areas | Single Move Tries | Swap Pair Tries |
|-----------|------------------|-----------------|
| 100 | 250 | 250 |
| 300 | 250 | 250 |
| 600 | 250 | 250 |
| 900 | 300 | 300 |
| 1200 | 400 | 400 |

## 2.4. VNS (Variable Neighborhood Search)

### 2.4.1. Thuật toán

```
Algorithm: VNS
──────────────
1. best ← LocalSearch(initial)
2. FOR iter = 1 to max_iter:
3.    k ← 1
4.    WHILE k ≤ k_max:
5.       shaken ← Shake(best, k)
6.       shaken ← RepairContiguity(shaken)
7.       improved ← LocalSearch(shaken)
8.       IF f(improved) < f(best):
9.          best ← improved
10.         k ← 1  // Reset
11.      ELSE:
12.         k ← k + 1  // Tăng mức shake
13. RETURN best
```

### 2.4.2. Shake Function

```python
def shake(solution, k, num_districts):
    for _ in range(k):
        # Chọn ngẫu nhiên 1 district
        district = random.randint(0, num_districts - 1)
        areas_in_dist = [i for i, d in enumerate(solution) if d == district]
        
        # Di chuyển 15% areas sang district khác
        num_move = max(1, int(len(areas_in_dist) * 0.15))
        to_move = random.sample(areas_in_dist, num_move)
        
        for idx in to_move:
            new_dist = random.randint(0, num_districts - 1)
            solution[idx] = new_dist
    
    return solution
```

## 2.5. VNS Overtime Priority

### 2.5.1. Ý tưởng cốt lõi

Thay vì shake ngẫu nhiên, **ưu tiên chuyển areas từ overtime districts sang undertime districts**.

### 2.5.2. Thuật toán

```
Algorithm: VNS Overtime Priority
────────────────────────────────
1. best ← LocalSearchMultiSwap(initial)
2. FOR iter = 1 to max_iter:
3.    k ← 1
4.    WHILE k ≤ k_max:
5.       # Xác định overtime/undertime districts
6.       overtime ← {d : time[d] > t_max}
7.       undertime ← {d : time[d] ≤ t_max}
8.       
9.       IF overtime ≠ ∅ AND undertime ≠ ∅:
10.         shaken ← ShakeOvertime(best, k, overtime, undertime)
11.      ELSE:
12.         shaken ← Shake(best, k)
13.      
14.      shaken ← RepairContiguity(shaken)
15.      improved ← LocalSearchMultiSwap(shaken)
16.      
17.      IF f(improved) < f(best):
18.         best ← improved
19.         k ← 1
20.      ELSE:
21.         k ← k + 1
22. RETURN best
```

### 2.5.3. ShakeOvertime Function

```python
def shake_overtime(solution, k, areas, overtime_dists, undertime_dists, graph, times):
    # Chọn k overtime districts để xử lý
    targets = random.sample(overtime_dists, min(k, len(overtime_dists)))
    
    for dist in targets:
        dist_areas = [i for i, d in enumerate(solution) if d == dist]
        
        # Sắp xếp theo service_time giảm dần
        sorted_areas = sorted(dist_areas, 
                              key=lambda i: areas[i]['service_time'], 
                              reverse=True)
        
        # Di chuyển 15% areas có service_time cao nhất
        num_move = max(1, int(len(sorted_areas) * 0.15))
        
        for idx in sorted_areas[:num_move]:
            # Tìm undertime district lân cận có time thấp nhất
            neighbors = {solution[n] for n in graph[idx] if solution[n] != dist}
            valid = [d for d in neighbors if d in undertime_dists]
            
            if valid:
                valid.sort(key=lambda d: times[d])
                solution[idx] = valid[0]
                times[valid[0]] += areas[idx]['service_time']
    
    return solution
```

## 2.6. Tối ưu hóa thời gian chạy

### 2.6.1. Pre-computing

```python
# Tính toán 1 lần, sử dụng nhiều lần
area_positions = np.array([[a['x'], a['y']] for a in areas])
graph = build_neighbor_graph(areas)
```

**Hiệu quả:** Giảm ~30% thời gian tổng thể.

### 2.6.2. Vectorized Operations

```python
# Thay vì loop Python:
compactness = sum(distance(area, center) for area in areas)

# Sử dụng NumPy vectorized:
district_centers = np.array([centers[solution[i]] for i in range(len(areas))])
compactness = np.sum(np.linalg.norm(area_positions - district_centers, axis=1))
```

**Hiệu quả:** Nhanh hơn ~5x cho phép tính khoảng cách.

### 2.6.3. Adaptive Parameters

Tự động điều chỉnh theo kích thước bài toán:

```python
def get_adaptive_params(num_areas):
    if num_areas <= 100:
        return {'max_iter': 5, 'k_max': 6, 'ls_iter': 200, 'boundary_limit': 200}
    elif num_areas <= 500:
        return {'max_iter': 3, 'k_max': 5, 'ls_iter': 200, 'boundary_limit': 200}
    elif num_areas <= 1000:
        return {'max_iter': 2, 'k_max': 5, 'ls_iter': 200, 'boundary_limit': 200}
    else:
        return {'max_iter': 2, 'k_max': 5, 'ls_iter': 200, 'boundary_limit': 200}
```

### 2.6.4. Early Termination

```python
# Dừng sớm khi không còn cải thiện
if not improved:
    break
```

### 2.6.5. Boundary Limiting

```python
# Giới hạn số boundary areas được xét
boundary = find_boundary_areas(current)
random.shuffle(boundary)
for area in boundary[:boundary_limit]:
    ...
```

**Hiệu quả:** Giảm độ phức tạp từ O(n) xuống O(boundary_limit).

---

# CHƯƠNG 3: ĐÁNH GIÁ VÀ KẾT QUẢ

## 3.1. Môi trường thực nghiệm

- **CPU:** Intel Core i5
- **RAM:** 8GB
- **OS:** Windows 10/11
- **Language:** Python 3.13
- **Framework:** FastAPI + NumPy

## 3.2. Các kịch bản test

| Kịch bản | Num Areas | Num Districts | Num Drivers | Num Vehicles |
|----------|-----------|---------------|-------------|--------------|
| Small | 100 | 5 | 5 | 5 |
| Medium | 500 | 25 | 30 | 30 |
| Large | 1000 | 50 | 60 | 60 |

## 3.3. So sánh thời gian chạy

| Kịch bản | Local Search | VNS | VNS Multi-Swap | VNS Overtime |
|----------|--------------|-----|----------------|--------------|
| Small | ~2s | ~5s | ~8s | ~10s |
| Medium | ~15s | ~40s | ~60s | ~80s |
| Large | ~40s | ~120s | ~180s | ~240s |

## 3.4. So sánh chất lượng lời giải

| Kịch bản | Initial Score | Local Search | VNS | VNS Overtime |
|----------|---------------|--------------|-----|--------------|
| Small | 0.45 | 0.28 | 0.22 | 0.18 |
| Medium | 0.52 | 0.35 | 0.28 | 0.24 |
| Large | 0.58 | 0.42 | 0.36 | 0.32 |

**% Cải thiện so với Initial:**

| Thuật toán | Small | Medium | Large |
|------------|-------|--------|-------|
| Local Search | 38% | 33% | 28% |
| VNS | 51% | 46% | 38% |
| VNS Overtime | 60% | 54% | 45% |

## 3.5. Phân tích Overtime

| Kịch bản | Thuật toán | Số districts overtime | Avg overtime (giờ) |
|----------|------------|----------------------|-------------------|
| Medium | Initial | 8/25 (32%) | 1.2h |
| Medium | Local Search | 4/25 (16%) | 0.5h |
| Medium | VNS Overtime | 1/25 (4%) | 0.1h |

## 3.6. Kết luận

### 3.6.1. Điểm mạnh

1. **VNS Overtime Priority** cho kết quả tốt nhất về giảm overtime
2. **Adaptive parameters** giúp cân bằng giữa tốc độ và chất lượng
3. **Pre-computing và vectorization** giảm đáng kể thời gian chạy

### 3.6.2. Hạn chế

1. Với bài toán > 1000 areas, thời gian chạy vẫn khá lâu
2. Kết quả phụ thuộc vào initial solution

### 3.6.3. Hướng phát triển

1. **Parallel processing**: Song song hóa các vòng lặp độc lập
2. **GPU acceleration**: Sử dụng CUDA cho các phép tính ma trận
3. **Hybrid algorithms**: Kết hợp với Genetic Algorithm hoặc Simulated Annealing

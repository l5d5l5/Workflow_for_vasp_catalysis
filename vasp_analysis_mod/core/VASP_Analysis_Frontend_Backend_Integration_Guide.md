# VASP 分析模块前后端对接文档（最终版）

> 适配版本：`main.py`（Dispatcher 统一调度）+ `Analysis.py`（含 `VaspAnalysisDispatcher`）  
> 文档目标：前端/后端联调、接口维护、功能验收

---

## 1. 文档目的

本文档定义 VASP 分析模块的最终对接规范，覆盖：

- 统一请求/响应格式
- Dispatcher 任务映射
- DOS / Relax / COHP（Summary/Curves/Export）接口协议
- 安全规则（路径白名单、CORS）
- 联调顺序与错误码约定

---

## 2. 核心架构与任务分发

后端路由统一调用：

- `VaspAnalysisDispatcher.dispatch(task_type, work_dir, **kwargs)`

### 2.1 API 与 task_type 对照

| API 路由 | task_type | 对应分析能力 |
|---|---|---|
| `POST /api/vasp/structure` | `structure_info` | 结构信息提取 |
| `POST /api/vasp/dos` | `dos` | DOS 多曲线分析 |
| `POST /api/vasp/relax` | `relax` | 结构优化分析 |
| `POST /api/vasp/cohp/summary` | `cohp_summary` | ICOHP 摘要 + 筛选 |
| `POST /api/vasp/cohp/curves` | `cohp_curves` | 指定键 COHP 曲线 |
| `POST /api/vasp/cohp/export` | `cohp_export` | 单键分轨道导出 |

---

## 3. 全局数据协议

### 3.1 请求格式

- `Content-Type: application/json`

### 3.2 统一响应格式

```json
{
  "success": true,
  "code": 200,
  "message": "Success",
  "data": {}
}
```

字段说明：

- `success`：业务成功标识
- `code`：业务码（非 HTTP 状态码）
- `message`：提示信息
- `data`：业务数据体

---

## 4. 公共输入字段（所有 POST 接口）

```json
{
  "workDir": "/data/vasp/job_001",
  "inputType": "path",
  "inputValue": "/data/vasp/job_001",
  "save_data": false
}
```

规则：

- `workDir` 与 `inputValue` 至少提供一个（推荐 `workDir`）
- `inputType` 当前仅支持 `"path"`
- `save_data` 透传给分析器，控制是否保存中间/结果文件

---

## 5. 安全与访问控制

### 5.1 目录白名单

后端仅允许访问 `ALLOWED_BASE_DIRS` 环境变量配置目录及其子目录。  
若未配置，默认只允许服务启动目录 `Path.cwd()`。

### 5.2 CORS

允许来源由 `ALLOWED_ORIGINS` 控制。生产环境请配置固定域名。

### 5.3 推荐环境变量

```bash
export ALLOWED_BASE_DIRS="/data/vasp,/home/ubuntu/vasp_jobs"
export ALLOWED_ORIGINS="http://localhost:5173,https://your-ui-domain.com"
```

---

## 6. 健康检查

## 6.1 `GET /healthz`

功能：检查服务可用性与目录策略。

响应示例：

```json
{
  "success": true,
  "code": 200,
  "message": "ok",
  "data": {
    "service": "VASP Analysis API",
    "dispatcher": "VaspAnalysisDispatcher",
    "allowedBaseDirs": ["/data/vasp"]
  }
}
```

---

## 7. 结构分析接口

## 7.1 `POST /api/vasp/structure`

功能：提取结构信息（优先 `CONTCAR`，否则 `POSCAR`）。

请求示例：

```json
{
  "workDir": "/data/vasp/job_001"
}
```

响应示例：

```json
{
  "success": true,
  "code": 200,
  "message": "Structure info extracted successfully",
  "data": {
    "formula": "Pd4CO2",
    "elements": ["Pd", "C", "O"],
    "totalAtoms": 7,
    "volume": 123.456,
    "lattice": {
      "a": 8.5,
      "b": 8.5,
      "c": 20.0,
      "alpha": 90.0,
      "beta": 90.0,
      "gamma": 120.0
    },
    "vasp_text": "...原始结构文本..."
  }
}
```

前端用途：

- `elements`：DOS/COHP 元素选择
- `totalAtoms`：DOS site 索引上限（1-indexed）

---

## 8. DOS 接口（支持多曲线一次请求）

## 8.1 `POST /api/vasp/dos`

功能：批量计算 DOS 曲线与统计描述符。

请求示例：

```json
{
  "workDir": "/data/vasp/job_001",
  "erange": [-10, 5],
  "show_tdos": true,
  "curves": [
    {
      "mode": "element",
      "element": "Pd",
      "orbital": "d"
    },
    {
      "mode": "site",
      "site": 1,
      "orbital": "p"
    }
  ]
}
```

参数说明：

- `curves[]`：多曲线数组（无需额外“多曲线路由”）
- `mode="element"`：按元素汇总
- `mode="site"`：按位点（前端传 1-indexed）
- `erange=[emin, emax]`，需满足 `emin < emax`
- `show_tdos=true` 时返回总态密度

响应示例（节选）：

```json
{
  "success": true,
  "code": 200,
  "message": "DOS analysis complete",
  "data": {
    "energy": [-9.9998, -9.9387, 4.9999],
    "ispin": 2,
    "tdos": {
      "up": [0.11, 0.09],
      "down": [0.10, 0.08]
    },
    "curves": [
      {
        "id": "unknown",
        "label": "element-d",
        "color": "#333",
        "dos_up": [0.12, 0.15],
        "dos_down": [0.11, 0.14],
        "stats": {
          "center": -1.2345,
          "width": 0.5678,
          "skewness": 0.0123,
          "kurtosis": 3.4567,
          "filling": 0.6789,
          "filled_states": 12.3456,
          "center_down": -1.3456,
          "width_down": 0.4321,
          "skewness_down": 0.0234,
          "kurtosis_down": 2.9876,
          "filling_down": 0.6543,
          "filled_states_down": 11.2345,
          "magnetic_moment": 0.1234
        }
      }
    ]
  }
}
```

---

## 9. Relax 接口

## 9.1 `POST /api/vasp/relax`

功能：结构优化收敛分析，支持位点分波磁矩解析。

请求示例：

```json
{
  "workDir": "/data/vasp/job_001",
  "get_site_mag": true
}
```

> 字段名为 `get_site_mag`（不是 `get_site_magment`）

响应示例（节选）：

```json
{
  "success": true,
  "code": 200,
  "message": "Optimization converged",
  "data": {
    "converged": true,
    "final_energy_eV": -148.9125,
    "fermi_level_eV": -2.4512,
    "ionic_steps": 42,
    "total_electrons": 48.0,
    "total_magnetization": 1.852,
    "energy_history": [-150.1, -149.3, -148.9],
    "de_history": [-0.81, -0.45, -0.01],
    "force_history": [3.50, 2.85, 0.02],
    "final_force": 0.0185,
    "site_magnetization": [
      {"atom_index": 1, "s": 0.00, "p": 0.00, "d": 0.12, "tot": 0.12}
    ],
    "warnings": [],
    "initial_structure": { "...": "..." },
    "final_structure": { "...": "..." }
  }
}
```

---

## 10. COHP 接口（按 UI 交互拆分）

COHP 已拆分为三条接口，匹配前端操作流：

1. 摘要列表（含筛选）  
2. 指定键曲线  
3. 单键导出

---

## 10.1 `POST /api/vasp/cohp/summary`

功能：获取 ICOHP 摘要，支持筛选。

请求模型关键字段：

- `n_top_bonds: int`（默认 20）
- `filter_type: "none" | "element_pair" | "bond_index"`（默认 `"none"`）
- `filter_value`：
  - `element_pair` 时示例：`["Fe", "O"]`
  - `bond_index` 时示例：`["1", "2"]` 或 `[1, 2]`

请求示例 1（前20个键）：

```json
{
  "workDir": "/data/vasp/lobster_job",
  "n_top_bonds": 20
}
```

请求示例 2（元素对筛选）：

```json
{
  "workDir": "/data/vasp/lobster_job",
  "filter_type": "element_pair",
  "filter_value": ["Fe", "O"]
}
```

请求示例 3（键编号筛选）：

```json
{
  "workDir": "/data/vasp/lobster_job",
  "filter_type": "bond_index",
  "filter_value": ["1", "2", "15"]
}
```

响应示例（节选）：

```json
{
  "success": true,
  "code": 200,
  "message": "COHP summary complete",
  "data": {
    "ispin": 2,
    "icohp_summary": [
      {
        "bond_label": "1",
        "atom1": "Fe1",
        "atom2": "O2",
        "length_Ang": 1.85,
        "icohp_up": -2.10,
        "icohp_down": -2.00,
        "icohp_total": -2.05,
        "orbitals": {}
      }
    ],
    "n_bonds": 20
  }
}
```

---

## 10.2 `POST /api/vasp/cohp/curves`

功能：获取指定键 COHP 曲线，支持是否包含分轨道曲线列。

请求字段：

- `bond_labels: [str]`（可选；不传则由后端默认策略决定）
- `erange: [emin, emax]`（可选）
- `include_orbitals: bool`（默认 `false`）

请求示例：

```json
{
  "workDir": "/data/vasp/lobster_job",
  "bond_labels": ["1", "2"],
  "erange": [-10, 5],
  "include_orbitals": false
}
```

响应示例（节选）：

```json
{
  "success": true,
  "code": 200,
  "message": "COHP curves complete",
  "data": {
    "cohp_curves": [
      {
        "energy_eV": -10.0,
        "1_up": -0.12,
        "1_down": -0.10,
        "2_up": -0.08,
        "2_down": -0.07
      }
    ]
  }
}
```

---

## 10.3 `POST /api/vasp/cohp/export`

功能：导出单个 bond 的分轨道数据（下载场景）。

请求字段：

- `export_type: "single"`（当前支持单键导出）
- `bond_label: str`（必填）
- `erange`（可选）
- `include_orbitals: bool`（建议 `true`）

请求示例：

```json
{
  "workDir": "/data/vasp/lobster_job",
  "export_type": "single",
  "bond_label": "1",
  "erange": [-8, 2],
  "include_orbitals": true
}
```

响应示例：

```json
{
  "success": true,
  "code": 200,
  "message": "Successfully exported bond 1",
  "data": {
    "energy_eV": [-8.0, -7.95, 2.0],
    "1_up": [-0.10, -0.12, 0.02],
    "1_down": [-0.08, -0.10, 0.03],
    "1_d_xz-p_y_up": [-0.02, -0.03, 0.01],
    "1_d_xz-p_y_down": [-0.01, -0.02, 0.01]
  }
}
```

---

## 11. 前端联调建议流程

1. `GET /healthz`：确认服务状态与目录白名单  
2. `POST /api/vasp/structure`：初始化元素/原子范围  
3. `POST /api/vasp/dos`：按 UI 选择批量拉取 DOS 曲线  
4. `POST /api/vasp/relax`：按需传 `get_site_mag=true`  
5. `POST /api/vasp/cohp/summary`：加载摘要并做筛选  
6. `POST /api/vasp/cohp/curves`：按选中键拉曲线  
7. `POST /api/vasp/cohp/export`：导出单键分轨道数据

---

## 12. 业务错误码约定

| code | 含义 | 常见原因 |
|---|---|---|
| 200 | 成功 | 正常返回 |
| 400 | 参数错误 | 缺字段、类型不匹配、`erange` 非法 |
| 403 | 权限错误 | 目录越权（不在 `ALLOWED_BASE_DIRS`） |
| 404 | 资源不存在 | `workDir` 不存在 |
| 500 | 服务内部错误 | 解析失败、调度异常、未知错误 |

---

## 13. 功能状态总览（当前）

| 能力 | 状态 | 说明 |
|---|---|---|
| Dispatcher 统一调度 | ✅ | 已统一 |
| 结构信息预分析 | ✅ | `/structure` |
| DOS 多曲线批量分析 | ✅ | `/dos` 单请求多曲线 |
| Relax 位点磁矩开关 | ✅ | `get_site_mag` |
| COHP 摘要接口 | ✅ | `/cohp/summary` |
| COHP 元素对筛选 | ✅ | `filter_type=element_pair` |
| COHP 键编号筛选 | ✅ | `filter_type=bond_index` |
| COHP 曲线接口 | ✅ | `/cohp/curves` |
| COHP 单键分轨道导出 | ✅ | `/cohp/export` |

---

## 14. 快速启动

在后端目录执行：

```bash
uvicorn main:app --reload --port 8000
```

---

## 15. 维护同步要求

当 `Analysis.py` 调度任务或参数签名变化时，需同步更新：

1. `main.py` 的 Pydantic 请求模型
2. 路由内 `run_task(..., **kwargs)` 参数透传
3. 本文档的请求示例与字段定义
4. 前端请求构造与字段命名

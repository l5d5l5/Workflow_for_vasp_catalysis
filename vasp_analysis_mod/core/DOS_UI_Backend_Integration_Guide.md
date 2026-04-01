# VASP DOS 前端 / 后端 对接文档

## 1. 现有实现概览

当前文件夹中包含：
- `main.py`: FastAPI 后端服务入口，提供 DOS 相关 API
- `Analysis.py`: 核心分析逻辑，包含 `DosAnalysis`、`RelaxAnalysis`、`StructureAnalysis`、`CohpAnalysis`
- `parse.py`: 高速解析器，包含 `FastCohpcar` 和 `DoscarParser`
- `最最终版-VASP_DOS_Analysis_UI_(Strict_Validation_+_Stats).html`: DOS 前端 UI 设计页面
- `最终版本VASP_Relax_Analysis_UI_(With_Force_&_Full_Params).html`: Relax 前端 UI 设计页面

## 2. 后端功能实现情况

### 2.1 已实现

- `main.py` 提供三个接口：
  - `POST /api/vasp/structure`
  - `POST /api/vasp/dos`
  - `POST /api/vasp/relax`
- `Analysis.py` 中 `DosAnalysis` 已实现：
  - 读取 `work_dir/DOSCAR`
  - 读取 `work_dir/POSCAR` 或 `work_dir/OUTCAR` 获取元素列表
  - 通过 `show_tdos` 返回 TDOS 数据
  - 根据 `curves` 配置按元素或按位点提取 PDOS
  - 支持 `s` / `p` / `d` 轨道投影
  - 计算描述符（d 带中心、宽度、偏度、峰度、填充、filled_states）
- `parse.py` 提供高性能 DOSCAR 解析器

### 2.2 目前未完全对外暴露 / 需补充

- `StructureAnalysis`、`CohpAnalysis` 已在 `Analysis.py` 中实现，但当前 `main.py` 未将它们注册为 API 路由
- 前端 HTML 页面当前只使用模拟数据，不存在真实的后端调用
- `inputValue` 在后端中直接作为工作目录路径使用；如果 UI 传入的是“队列 ID”，则需要前端或后端增加映射逻辑

## 3. 后端 API 说明

### 3.1 POST /api/vasp/structure

请求示例：
```json
{
  "inputType": "path",
  "inputValue": "/path/to/vasp/workdir"
}
```

响应示例：
```json
{
  "success": true,
  "code": 200,
  "message": "Success",
  "data": {
    "totalAtoms": 32,
    "elements": ["Pd", "C", "O"]
  }
}
```

说明：
- 后端从 `POSCAR` 或 `OUTCAR` 中提取元素列表
- `inputValue` 目前必须是一个真实可访问的目录路径

### 3.2 POST /api/vasp/dos

请求示例：
```json
{
  "inputType": "path",
  "inputValue": "/path/to/vasp/workdir",
  "erange": [-10, 5],
  "show_tdos": true,
  "curves": [
    {
      "mode": "element",
      "element": "Pd",
      "orbital": "d",
      "id": "curve-1",
      "label": "Pd (All) - d",
      "color": "#3b82f6"
    }
  ]
}
```

响应结构：
- `energy`: 能量数组（已减去 `efermi`）
- `ispin`: 自旋数
- `tdos`: TDOS 数据，若 `show_tdos=true`
- `curves`: 每条曲线的 PDOS 数据和统计结果

曲线对象示例：
```json
{
  "id": "curve-1",
  "label": "Pd (All) - d",
  "color": "#3b82f6",
  "dos_up": [...],
  "dos_down": [...],
  "stats": {
    "center": -1.2345,
    "width": 0.4567,
    "skewness": 0.1234,
    "kurtosis": 3.5678,
    "filling": 0.6789,
    "filled_states": 12.3456,
    "magnetic_moment": 0.1234
  }
}
```

## 3.3 POST /api/vasp/relax

请求示例：
```json
{
  "inputType": "path",
  "inputValue": "/path/to/vasp/workdir",
  "get_site_mag": true
}
```

响应示例：
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
    "energy_history": [...],
    "de_history": [...],
    "force_history": [...],
    "final_force": 0.0185,
    "site_magnetization": [...],
    "warnings": [...],
    "initial_structure": {...},
    "final_structure": {...}
  }
}
```

说明：
- `get_site_mag` 为可选参数，若为 `true` 则会解析并返回每个原子的分波磁矩数据
- 该接口依赖 `OUTCAR`、`OSZICAR`、`POSCAR`/`CONTCAR`

## 4. 前端对接建议

### 4.1 需要实现的前端请求逻辑

#### 4.1.1 结构加载

`最最终版-VASP_DOS_Analysis_UI_(Strict_Validation_+_Stats).html` 中的 `loadStructure()` 应改为：
- 发起 `POST /api/vasp/structure`
- 参数为 `inputValue` 的目录路径
- 读取返回数据并填充：
  - `structure.totalAtoms`
  - `structure.elements`

#### 4.1.2 DOS 分析

`analyze()` 应改为：
- 发起 `POST /api/vasp/dos`
- 传入当前 `curves`、`erange`、`show_tdos`
- 解析返回结果并渲染图表
- 对 `ispin` 为 2 的情况，使用 `dos_down` 数据

### 4.2 变量映射

前端与后端字段关系：
- `eMin`, `eMax` -> `erange`
- `showTDOS` -> `show_tdos`
- `curves[].payload.type` -> `curves[].mode`
- `curves[].payload.element` -> `curves[].element`
- `curves[].payload.site` -> `curves[].site`
- `curves[].payload.orbital` -> `curves[].orbital`

### 4.3 需注意的点

- `site` 值应当为 1-indexed，后端会减 1 处理
- 后端 `DosAnalysis` 只支持实际文件目录工作区，前端如果使用“队列 ID”，必须先转换为物理路径
- 当前 UI 的 `addCurve()` 生成的 `payload` 中已有 `type`、`element`、`site`、`orbital`
  - 只需映射到后端 `curves` 参数即可

## 5. 当前集成状态评估

### 已实现
- 后端 `main.py` 已经实现 `structure`、`dos` 和 `relax` 三个核心 API
- 后端 `DosAnalysis` 已经具备真实 DOSCAR/PDOS 解析与轨道投影分析能力
- 后端 `parse.py` 已实现高性能 DOSCAR 解析器

### 未完成
- `最最终版-VASP_DOS_Analysis_UI_(Strict_Validation_+_Stats).html` 目前仍然使用前端模拟数据，不会调用后端接口
- Relax UI 页面目前是静态模拟演示，尚未调用新的 `POST /api/vasp/relax` 后端接口
- `StructureAnalysis`、`CohpAnalysis` 功能仍然没有对应的后端路由

## 6. 推荐后续开发步骤

1. 启动后端服务：
   ```bash
   uvicorn main:app --reload --port 8000
   ```
2. 将 DOS 前端页面的 `loadStructure()`、`analyze()` 等逻辑改为 `fetch` 调用后端
3. 为 Relax UI 新增前端调用，使用 `POST /api/vasp/relax`；如需结构信息可继续补充 `StructureAnalysis.get_info()` 路由
4. 如果前端仍使用“队列 ID”，则补充一个 ID -> 工作目录路径的映射层
5. 统一前端返回数据结构，避免模拟数据和真实数据字段不一致

## 7. 关键文件说明

- `main.py`: FastAPI 后端入口，负责 CORS 和路由注册
- `Analysis.py`: 核心业务逻辑实现
- `parse.py`: VASP 文件高性能解析器
- `最最终版-VASP_DOS_Analysis_UI_(Strict_Validation_+_Stats).html`: DOS 前端 UI 模板
- `最终版本VASP_Relax_Analysis_UI_(With_Force_&_Full_Params).html`: Relax 前端 UI 模板
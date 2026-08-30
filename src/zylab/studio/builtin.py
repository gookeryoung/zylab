"""内置分析模板（随包发布的预连节点图；用户模板另存于 data_dir/templates/）."""

from __future__ import annotations

from .template import Template

__all__ = ["BUILTIN_TEMPLATES"]

#: 悬臂梁模型的常用参数分组（几何/材料/载荷）
_BEAM_GROUPS = (
    {"title": "几何与网格", "params": ["model.length", "model.height", "model.nx", "model.ny"]},
    {"title": "材料", "params": ["model.e_modulus", "model.poisson", "model.thickness", "model.density"]},
    {"title": "载荷", "params": ["model.tip_load"]},
)

BUILTIN_TEMPLATES: tuple[Template, ...] = tuple(
    Template.from_dict(raw)
    for raw in (
        {
            "id": "structural.cantilever_static",
            "name": "悬臂梁静力分析",
            "discipline": "structural",
            "description": "Q4 平面应力悬臂梁端部受载，输出变形云图与应变能。",
            "tags": ["入门"],
            "nodes": [
                {"id": "model", "type": "example.cantilever_q4"},
                {"id": "solve", "type": "analysis.static", "inputs": {"model": "model.model"}},
            ],
            "ui": {"param_groups": _BEAM_GROUPS, "results": ["solve"]},
        },
        {
            "id": "structural.cantilever_modal",
            "name": "悬臂梁模态分析",
            "discipline": "structural",
            "description": "Q4 平面应力悬臂梁自由振动，输出频率表与各阶振型云图。",
            "tags": ["振动"],
            "nodes": [
                {"id": "model", "type": "example.cantilever_q4", "params": {"tip_load": 0.0}},
                {"id": "solve", "type": "analysis.modal", "inputs": {"model": "model.model"}},
            ],
            "ui": {
                "param_groups": [*_BEAM_GROUPS, {"title": "分析", "params": ["solve.n_modes"]}],
                "results": ["solve"],
            },
        },
        {
            "id": "structural.cantilever_harmonic",
            "name": "悬臂梁谐响应分析",
            "discipline": "structural",
            "description": "端部简谐激励下的频率扫描，输出末端观察点频响曲线（对数幅值轴）与共振峰。",
            "tags": ["振动"],
            "nodes": [
                {"id": "model", "type": "example.cantilever_q4"},
                {"id": "solve", "type": "analysis.harmonic", "inputs": {"model": "model.model"}},
            ],
            "ui": {
                "param_groups": [
                    *_BEAM_GROUPS,
                    {
                        "title": "扫频与阻尼",
                        "params": ["solve.f_max", "solve.n_freq", "solve.alpha", "solve.beta"],
                    },
                ],
                "results": ["solve"],
            },
        },
        {
            "id": "structural.cantilever_transient",
            "name": "悬臂梁瞬态动力分析",
            "discipline": "structural",
            "description": "端部阶跃载荷的 Newmark 时间积分，输出末帧变形云图与末端观察点位移时程曲线。",
            "tags": ["振动"],
            "nodes": [
                {"id": "model", "type": "example.cantilever_q4"},
                {"id": "solve", "type": "analysis.transient", "inputs": {"model": "model.model"}},
            ],
            "ui": {
                "param_groups": [
                    *_BEAM_GROUPS,
                    {"title": "时程与阻尼", "params": ["solve.duration", "solve.n_steps", "solve.alpha", "solve.beta"]},
                ],
                "results": ["solve"],
            },
        },
        {
            "id": "structural.column_buckling",
            "name": "压杆稳定（屈曲）分析",
            "discipline": "structural",
            "description": "BEAM2 悬臂柱顶部轴压，输出各阶屈曲载荷因子与失稳模态（一阶 ≈ π²EI/4L²）。",
            "tags": ["稳定性", "基准"],
            "nodes": [
                {"id": "model", "type": "example.column_beam2"},
                {"id": "solve", "type": "analysis.buckling", "inputs": {"model": "model.model"}},
            ],
            "ui": {
                "param_groups": [
                    {"title": "几何与网格", "params": ["model.height", "model.n_elem"]},
                    {"title": "截面", "params": ["model.area", "model.inertia"]},
                    {"title": "载荷", "params": ["model.tip_load"]},
                    {"title": "分析", "params": ["solve.n_modes"]},
                ],
                "results": ["solve"],
            },
        },
        {
            "id": "structural.truss_nonlinear",
            "name": "两杆桁架几何非线性分析",
            "discipline": "structural",
            "description": "TRUSS2 两杆浅桁架顶点受载（接近极限点），输出变形云图与载荷-位移曲线。",
            "tags": ["非线性"],
            "nodes": [
                {"id": "model", "type": "example.truss2_two_bar"},
                {"id": "solve", "type": "analysis.nonlinear", "inputs": {"model": "model.model"}},
            ],
            "ui": {
                "param_groups": [
                    {"title": "几何", "params": ["model.half_span", "model.rise"]},
                    {"title": "材料与截面", "params": ["model.e_modulus", "model.area"]},
                    {"title": "载荷", "params": ["model.apex_load"]},
                    {"title": "分析", "params": ["solve.n_increments"]},
                ],
                "results": ["solve"],
            },
        },
        {
            "id": "structural.column_buckling_linked",
            "name": "压杆屈曲分析（静力参考态链接）",
            "discipline": "structural",
            "description": "静力节点产出参考态，屈曲节点经 reference 端口复用其轴力，演示模块 Link 连接。",
            "tags": ["稳定性", "链接"],
            "nodes": [
                {"id": "model", "type": "example.column_beam2"},
                {"id": "static", "type": "analysis.static", "inputs": {"model": "model.model"}},
                {
                    "id": "buckling",
                    "type": "analysis.buckling",
                    "inputs": {"model": "model.model", "reference": "static.solution"},
                },
            ],
            "ui": {
                "param_groups": [
                    {"title": "几何与网格", "params": ["model.height", "model.n_elem"]},
                    {"title": "截面", "params": ["model.area", "model.inertia"]},
                    {"title": "载荷", "params": ["model.tip_load"]},
                    {"title": "分析", "params": ["buckling.n_modes"]},
                ],
                "results": ["static", "buckling"],
            },
        },
        {
            "id": "structural.truss_nonlinear_linked",
            "name": "两杆桁架非线性（静力初态链接）",
            "discipline": "structural",
            "description": "静力节点产出初态位移，非线性节点经 initial 端口从初态起算，演示模块 Link 连接。",
            "tags": ["非线性", "链接"],
            "nodes": [
                {"id": "model", "type": "example.truss2_two_bar"},
                {"id": "static", "type": "analysis.static", "inputs": {"model": "model.model"}},
                {
                    "id": "nonlinear",
                    "type": "analysis.nonlinear",
                    "inputs": {"model": "model.model", "initial": "static.solution"},
                },
            ],
            "ui": {
                "param_groups": [
                    {"title": "几何", "params": ["model.half_span", "model.rise"]},
                    {"title": "材料与截面", "params": ["model.e_modulus", "model.area"]},
                    {"title": "载荷", "params": ["model.apex_load"]},
                    {"title": "分析", "params": ["nonlinear.n_increments"]},
                ],
                "results": ["static", "nonlinear"],
            },
        },
        {
            "id": "structural.cantilever_combo",
            "name": "悬臂梁静力 + 模态联合分析",
            "discipline": "structural",
            "description": "同一模型共享驱动静力与模态两路分析，演示多学科模块的 Share 连接模式。",
            "tags": ["入门", "组合"],
            "nodes": [
                {"id": "model", "type": "example.cantilever_q4"},
                {"id": "static", "type": "analysis.static", "inputs": {"model": "model.model"}},
                {"id": "modal", "type": "analysis.modal", "inputs": {"model": "model.model"}},
            ],
            "ui": {
                "param_groups": [*_BEAM_GROUPS, {"title": "分析", "params": ["modal.n_modes"]}],
                "results": ["static", "modal"],
            },
        },
        {
            "id": "thermal.joule_plate_2d",
            "name": "通电加热板电-热耦合分析",
            "discipline": "thermal",
            "description": "Q4 平面板左右电极通电，Joule 热顺序耦合稳态温度场；"
            "底边恒温、其余三边对流散热，输出温度/电压云图。",
            "tags": ["电-热", "稳态"],
            "nodes": [
                {"id": "model", "type": "example.joule_plate_2d"},
                {"id": "solve", "type": "analysis.electrothermal", "inputs": {"model": "model.model"}},
            ],
            "ui": {
                "param_groups": [
                    {"title": "几何与网格", "params": ["model.length", "model.height", "model.nx", "model.ny"]},
                    {
                        "title": "材料与厚度",
                        "params": ["model.electric_sigma", "model.thermal_k", "model.thickness"],
                    },
                    {"title": "电学边界", "params": ["model.voltage"]},
                    {"title": "热边界", "params": ["model.t_base", "model.h_conv", "model.t_ambient"]},
                ],
                "results": ["solve"],
            },
        },
        {
            "id": "thermal.joule_series_2d",
            "name": "多材料串联电加热板（热点分析）",
            "discipline": "thermal",
            "description": "电极/电阻区/电极三区多块网格：电流经高阻抗电阻区集浓产热，"
            "稳态热点出现在电阻区；演示多材料 ElementBlock 分区建模与电-热耦合。",
            "tags": ["电-热", "多材料", "热点"],
            "nodes": [
                {"id": "model", "type": "example.joule_series_2d"},
                {"id": "solve", "type": "analysis.electrothermal", "inputs": {"model": "model.model"}},
            ],
            "ui": {
                "param_groups": [
                    {"title": "几何与网格", "params": ["model.length", "model.height", "model.nx", "model.ny"]},
                    {"title": "分区", "params": ["model.electrode_len"]},
                    {
                        "title": "电极材料",
                        "params": ["model.sigma_conductor", "model.k_conductor"],
                    },
                    {
                        "title": "电阻区材料",
                        "params": ["model.sigma_resistor", "model.k_resistor"],
                    },
                    {"title": "厚度与电学边界", "params": ["model.thickness", "model.voltage"]},
                    {"title": "热边界", "params": ["model.t_base", "model.h_conv", "model.t_ambient"]},
                ],
                "results": ["solve"],
            },
        },
    )
)

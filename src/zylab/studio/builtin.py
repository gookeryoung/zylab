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
    )
)

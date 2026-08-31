"""studio.meshing3d 三维网格生成与 3D 电阻建模节点测试：规模/雅可比/解析验证/瞬态节点."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import (
    ElementType,
    MeshError,
    solve_electric,
    solve_electrothermal,
)
from zylab.studio import BUILTIN_TEMPLATES, module_spec, nodes
from zylab.studio.meshing3d import cylinder_resistor_mesh, vfilm_resistor_mesh

__all__ = []


def _jacobian_dets(coords: np.ndarray, conn: np.ndarray) -> np.ndarray:
    """全部单元 8 个高斯点的原始雅可比行列式（符号敏感）."""
    from zylab.fea.conduction import _HEX8_GAUSS, _hex8_shape_derivs

    dets: list[float] = []
    for row in conn:
        for xi, eta, zeta in _HEX8_GAUSS:
            dn = _hex8_shape_derivs(xi, eta, zeta)
            dets.append(float(np.linalg.det(dn @ coords[row])))
    return np.asarray(dets)


class TestCylinderMesh:
    """圆柱电阻网格生成."""

    def test_structure_and_scale(self) -> None:
        """节点/单元数与极坐标范围：z∈[0,L]、半径∈[5%R, R]."""
        geo = cylinder_resistor_mesh(radius=2.0, length=10.0, n_theta=12, n_r=4, n_z=8)
        assert geo.mesh.n_nodes == 5 * 12 * 9
        assert geo.mesh.n_elements == 4 * 12 * 8
        assert geo.mesh.blocks[0].etype is ElementType.HEX8
        coords = geo.mesh.coords
        np.testing.assert_allclose(coords[:, 2].min(), 0.0, atol=1e-12)
        np.testing.assert_allclose(coords[:, 2].max(), 10.0, atol=1e-12)
        radii = np.hypot(coords[:, 0], coords[:, 1])
        assert radii.min() == pytest.approx(2.0 * 0.05, rel=1e-12)
        assert radii.max() == pytest.approx(2.0, rel=1e-12)

    def test_jacobian_positive(self) -> None:
        """全部单元雅可比行列式恒正（节点序右手）."""
        geo = cylinder_resistor_mesh(2.0, 10.0, 12, 4, 8)
        dets = _jacobian_dets(geo.mesh.coords, geo.mesh.blocks[0].conn)
        assert (dets > 0.0).all()
        assert dets.min() > 1.0e-12

    def test_boundary_references(self) -> None:
        """端面电极节点数与对流面片数（外侧面 + 两端面）."""
        geo = cylinder_resistor_mesh(2.0, 10.0, n_theta=8, n_r=3, n_z=5)
        n_ring = 4 * 8
        assert len(geo.end_low_nodes) == n_ring
        assert len(geo.end_high_nodes) == n_ring
        assert set(geo.end_low_nodes).isdisjoint(geo.end_high_nodes)
        assert len(geo.conv_faces) == 8 * 5 + 2 * 3 * 8

    def test_invalid_params_rejected(self) -> None:
        """非法几何/分段参数拒绝."""
        with pytest.raises(MeshError):
            cylinder_resistor_mesh(-1.0, 10.0, 8, 3, 5)
        with pytest.raises(MeshError):
            cylinder_resistor_mesh(2.0, 10.0, 2, 3, 5)
        with pytest.raises(MeshError):
            cylinder_resistor_mesh(2.0, 10.0, 8, 0, 5)


class TestVfilmMesh:
    """V 形薄膜电阻网格生成."""

    def test_structure_and_scale(self) -> None:
        """三块规模与总节点数（薄膜 2 层 + 基底 n_sub+1 层共享 z=0 层）."""
        geo = vfilm_resistor_mesh(10.0, 2.0, 1.0, 0.005, 1.0, 1.5, 4, 8, 2, 4)
        n_st = 2 * 4 + 2 * 8 + 1
        block_sizes = tuple(block.conn.shape[0] for block in geo.mesh.blocks)
        assert block_sizes == (2 * 8 * 2, 2 * 4 * 2, (n_st - 1) * 2 * 4)
        assert geo.mesh.n_nodes == n_st * 3 * (4 + 2)
        for block in geo.mesh.blocks:
            assert block.etype is ElementType.HEX8

    def test_shared_substrate_interface(self) -> None:
        """薄膜与基底共享 z=0 底面节点（节点总数恰为层数乘积，无重复登记）."""
        geo = vfilm_resistor_mesh(10.0, 2.0, 1.0, 0.005, 1.0, 1.5, 4, 8, 2, 4)
        n_st = 2 * 4 + 2 * 8 + 1
        # 总层数 = 薄膜 2 层 + 基底 5 层 - 共享 1 层 = 6 → 25*3*6 = 450
        assert geo.mesh.n_nodes == n_st * 3 * 6
        z_vals = np.unique(np.round(geo.mesh.coords[:, 2], 12))
        np.testing.assert_allclose(z_vals, np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.005]), rtol=1e-12)

    def test_v_apex_and_jacobian(self) -> None:
        """V 带状区域俯视范围（顶点站截面下探 -d-w/2）且全部单元雅可比恒正."""
        geo = vfilm_resistor_mesh(10.0, 2.0, 1.0, 0.005, 1.0, 1.5, 4, 8, 2, 4)
        assert geo.mesh.coords[:, 1].min() == pytest.approx(-2.0 - 0.5, abs=1e-9)  # 顶点站含宽度扩展
        assert geo.mesh.coords[:, 1].max() == pytest.approx(0.5, abs=1e-9)  # 水平段宽度扩展
        for block in geo.mesh.blocks:
            dets = _jacobian_dets(geo.mesh.coords, block.conn)
            assert (dets > 0.0).all()

    def test_invalid_params_rejected(self) -> None:
        """非法几何（电极段过长/负参数）与非法分段数拒绝."""
        with pytest.raises(MeshError):
            vfilm_resistor_mesh(1.0, 2.0, 1.0, 0.005, 1.0, 0.6, 4, 8, 2, 4)  # 2a ≥ L
        with pytest.raises(MeshError):
            vfilm_resistor_mesh(10.0, 2.0, 1.0, -0.005, 1.0, 1.5, 4, 8, 2, 4)
        with pytest.raises(MeshError):
            vfilm_resistor_mesh(10.0, 2.0, 1.0, 0.005, 1.0, 1.5, 0, 8, 2, 4)


class TestBuildCylinderResistor:
    """圆柱电阻建模节点."""

    def test_structure_and_validation(self) -> None:
        """HEX8 单块、材料含热容、双工况通过网格校验."""
        bundle = nodes.build_cylinder_resistor({}, {"n_theta": 8, "n_r": 3, "n_z": 5})
        assert bundle.mesh.blocks[0].etype is ElementType.HEX8
        assert bundle.materials[0].volumetric_heat_capacity > 0.0
        bundle.electric_case.validate(bundle.mesh)
        bundle.thermal_case.validate(bundle.mesh)

    def test_power_matches_polygon_area(self) -> None:
        """总功率 = σ·A_poly·V0²/L（内接多边形截面积，电场线性 HEX8 精确）."""
        bundle = nodes.build_cylinder_resistor({}, {"n_theta": 24, "n_r": 3, "n_z": 10})
        solution = solve_electrothermal(
            bundle.mesh, bundle.materials, bundle.sections, bundle.electric_case, bundle.thermal_case
        )
        r_in = 2.0 * 0.05
        # 内接 24 边形面积 = n/2 * R² * sin(2π/n) - 中心孔同比例
        area = (24 / 2.0) * (2.0**2 - r_in**2) * np.sin(2.0 * np.pi / 24)
        expected = 1.0 * area * 1.0**2 / 20.0
        assert solution.total_power == pytest.approx(expected, rel=5.0e-3)
        assert solution.thermal.t_max > solution.thermal.t_min  # Joule 升温

    def test_report_progress(self) -> None:
        """进度回调按阶段上报并以 1.0 收尾."""
        events: list[tuple[float, str]] = []
        nodes.build_cylinder_resistor({}, {"n_theta": 8, "n_r": 2, "n_z": 4}, report=lambda p, m: events.append((p, m)))
        assert events[0][0] < 1.0
        assert events[-1] == (1.0, "模型就绪")


class TestBuildVfilmResistor:
    """V 形薄膜电阻建模节点."""

    def test_structure_and_validation(self) -> None:
        """三块三材料、陶瓷绝缘、双工况通过网格校验."""
        bundle = nodes.build_vfilm_resistor({}, {"n_lead": 3, "n_diag": 6, "n_width": 2, "n_sub": 3})
        assert len(bundle.mesh.blocks) == 3
        assert bundle.materials[2].electric_sigma < 1.0e-9  # 陶瓷绝缘（装配过滤）
        bundle.electric_case.validate(bundle.mesh)
        bundle.thermal_case.validate(bundle.mesh)

    def test_steady_hotspot_in_v_region(self) -> None:
        """稳态热点位于 V 形阻性区（电极段之间）且温度高于基底恒温."""
        bundle = nodes.build_vfilm_resistor({}, {"n_lead": 3, "n_diag": 6, "n_width": 2, "n_sub": 3})
        solution = solve_electrothermal(
            bundle.mesh, bundle.materials, bundle.sections, bundle.electric_case, bundle.thermal_case
        )
        assert solution.total_power > 0.0
        hotspot = int(np.argmax(solution.thermal.temperatures))
        x_hot = bundle.mesh.coords[hotspot, 0]
        assert 1.5 < x_hot < 10.0 - 1.5  # 热点在电极段之间
        assert solution.thermal.t_max > 20.0  # 高于基底恒温与环境

    def test_voltage_boundary(self) -> None:
        """引入端 0V、引出端 V0（电极给定电压生效）."""
        bundle = nodes.build_vfilm_resistor({}, {"n_lead": 3, "n_diag": 6, "n_width": 2, "n_sub": 3, "voltage": 2.0})
        solution = solve_electric(bundle.mesh, bundle.materials, bundle.sections, bundle.electric_case)
        by_node = {item.node: item.value for item in bundle.electric_case.voltages}
        low = [n for n, v in by_node.items() if v == 0.0]
        high = [n for n, v in by_node.items() if v == 2.0]
        assert low and high
        np.testing.assert_allclose(solution.voltages[low], 0.0, atol=1e-12)
        np.testing.assert_allclose(solution.voltages[high], 2.0, atol=1e-12)


class TestRunElectrothermalTransient:
    """瞬态电-热耦合分析节点."""

    def test_cold_start_monotonic_heatup(self) -> None:
        """冷启动（初温 = 环境温度）：峰值温度逐帧单调升且未达稳态."""
        bundle = nodes.build_vfilm_resistor({}, {"n_lead": 3, "n_diag": 6, "n_width": 2, "n_sub": 3})
        solution = nodes.run_electrothermal_transient(
            {"model": bundle}, {"t_init": 20.0, "duration": 0.5, "n_steps": 20}
        )
        frame_max = solution.thermal.temperatures.max(axis=1)
        assert frame_max.shape == (21,)
        assert (np.diff(frame_max) > 0.0).all()
        steady = solve_electrothermal(
            bundle.mesh, bundle.materials, bundle.sections, bundle.electric_case, bundle.thermal_case
        )
        assert frame_max[-1] < steady.thermal.t_max
        assert solution.total_power == pytest.approx(steady.total_power, rel=1e-12)  # 常物性电场与时间无关

    def test_input_type_checked(self) -> None:
        """输入端口类型校验：非 ConductionBundle 拒绝."""
        with pytest.raises(TypeError):
            nodes.run_electrothermal_transient({"model": object()}, {})


class TestBuiltinTemplates3d:
    """三维电热预制模板加载."""

    _NEW_IDS = (
        "thermal.cylinder_resistor_steady",
        "thermal.cylinder_resistor_transient",
        "thermal.vfilm_resistor_steady",
        "thermal.vfilm_resistor_transient",
    )

    def test_new_templates_registered(self) -> None:
        """4 个 3D 电热模板随包加载."""
        known = {template.id for template in BUILTIN_TEMPLATES}
        for template_id in self._NEW_IDS:
            assert template_id in known

    def test_template_nodes_resolvable(self) -> None:
        """模板节点类型全部可在模块表解析."""
        for template in BUILTIN_TEMPLATES:
            if template.id in self._NEW_IDS:
                for node in template.nodes:
                    module_spec(node.type_id)

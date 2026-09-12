"""拆分式建模节点 split_nodes.py 测试."""

from __future__ import annotations

import pytest

from zylab.fea import ElementType, LinearElastic
from zylab.flowchart.bundle import GeometryBundle, MaterialPayload, ModelBundle
from zylab.flowchart.split_nodes import geom_cantilever_2d, material_linear_elastic, mesh_generate_structural


class TestMaterialLinearElastic:
    """material_linear_elastic 线弹性材料节点."""

    def test_default_params(self) -> None:
        """默认参数：标准钢材 E=2.1e5, ν=0.3, ρ=7.85."""
        payload = material_linear_elastic({}, {})
        assert isinstance(payload, MaterialPayload)
        assert isinstance(payload.material, LinearElastic)
        assert payload.material.e_modulus == pytest.approx(2.1e5)
        assert payload.material.poisson == pytest.approx(0.3)
        assert payload.material.density == pytest.approx(7.85)

    def test_custom_params(self) -> None:
        """自定义参数：铝 E=7e4, ν=0.33, ρ=2.7."""
        params = {"e_modulus": 7e4, "poisson": 0.33, "density": 2.7}
        payload = material_linear_elastic({}, params)
        assert payload.material.e_modulus == pytest.approx(7e4)
        assert payload.material.poisson == pytest.approx(0.33)
        assert payload.material.density == pytest.approx(2.7)


class TestGeomCantilever2d:
    """geom_cantilever_2d 悬臂梁平面应力几何节点."""

    def test_geometry_output(self) -> None:
        """输出 GeometryBundle：Mesh + Section + StaticCase."""
        params = {"length": 100.0, "height": 10.0, "nx": 4, "ny": 2, "tip_load": -1000.0, "thickness": 5.0}
        geo = geom_cantilever_2d({}, params)
        assert isinstance(geo, GeometryBundle)
        assert geo.mesh.n_nodes == (4 + 1) * (2 + 1)  # 15 节点
        assert len(geo.mesh.blocks) == 1
        assert geo.mesh.blocks[0].etype is ElementType.QUAD4
        assert geo.mesh.blocks[0].name == "梁"

    def test_constraints_and_loads(self) -> None:
        """约束固定左端 ny+1=3 个节点的 (0,1) 自由度，载荷施加到右端 3 节点."""
        params = {"length": 100.0, "height": 10.0, "nx": 4, "ny": 2, "tip_load": -1000.0, "thickness": 5.0}
        geo = geom_cantilever_2d({}, params)
        # 左端 ny+1=3 个节点，每个 Constraint 包含两个 DOF
        assert len(geo.case.constraints) == 3
        assert all(c.dofs == (0, 1) for c in geo.case.constraints)
        # tip_load 施加到右端 ny+1=3 个节点
        assert len(geo.case.loads) == 3
        assert all(load.forces == (0.0, -1000.0) for load in geo.case.loads)

    def test_section_thickness(self) -> None:
        """截面厚度参数正确传递."""
        params = {"length": 50.0, "height": 5.0, "nx": 2, "ny": 1, "tip_load": -500.0, "thickness": 3.5}
        geo = geom_cantilever_2d({}, params)
        assert len(geo.sections) == 1
        assert geo.sections[0].thickness == pytest.approx(3.5)


class TestMeshGenerateStructural:
    """mesh_generate_structural 结构网格装配节点."""

    @pytest.fixture()
    def geo(self) -> GeometryBundle:
        return geom_cantilever_2d(
            {},
            {"length": 100.0, "height": 10.0, "nx": 4, "ny": 2, "tip_load": -1000.0, "thickness": 5.0},
        )

    @pytest.fixture()
    def mat(self) -> MaterialPayload:
        return material_linear_elastic({}, {})

    def test_assembly(self, geo: GeometryBundle, mat: MaterialPayload) -> None:
        """GeometryBundle + MaterialPayload → ModelBundle."""
        model = mesh_generate_structural({"geometry": geo, "material": mat}, {})
        assert isinstance(model, ModelBundle)
        assert model.mesh is geo.mesh
        assert model.sections is geo.sections
        assert model.case is geo.case
        assert len(model.materials) == 1
        assert isinstance(model.materials[0], LinearElastic)
        assert model.materials[0].e_modulus == pytest.approx(2.1e5)

    def test_geometry_type_error(self, mat: MaterialPayload) -> None:
        """geometry 端口类型错误抛 TypeError."""
        with pytest.raises(TypeError, match="GeometryBundle"):
            mesh_generate_structural({"geometry": "not_a_bundle", "material": mat}, {})

    def test_material_type_error(self, geo: GeometryBundle) -> None:
        """material 端口类型错误抛 TypeError."""
        with pytest.raises(TypeError, match="MaterialPayload"):
            mesh_generate_structural({"geometry": geo, "material": "not_payload"}, {})

    def test_elastic_type_error(self, geo: GeometryBundle) -> None:
        """material.payload.material 不是 LinearElastic 时抛 TypeError."""
        fake = MaterialPayload(material="not_elastic")
        with pytest.raises(TypeError, match="LinearElastic"):
            mesh_generate_structural({"geometry": geo, "material": fake}, {})

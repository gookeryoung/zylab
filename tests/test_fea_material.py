"""fea.material 单元测试：D 矩阵与参数校验."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import ElementError, LinearElastic, Section, StressState


class TestLinearElastic:
    def test_plane_stress_d_matrix(self) -> None:
        mat = LinearElastic(e_modulus=210.0, poisson=0.3)
        d = mat.d_matrix()
        assert d.shape == (3, 3)
        expected = 210.0 / (1.0 - 0.3**2)
        np.testing.assert_allclose(
            d,
            np.array(
                [
                    [expected, 0.3 * expected, 0.0],
                    [0.3 * expected, expected, 0.0],
                    [0.0, 0.0, 210.0 / 2.6],
                ]
            ),
        )

    def test_plane_strain_d_matrix(self) -> None:
        mat = LinearElastic(e_modulus=210.0, poisson=0.3, state=StressState.PLANE_STRAIN)
        d = mat.d_matrix()
        assert d.shape == (3, 3)
        # 平面应变 D11 = E(1-nu)/((1+nu)(1-2nu))
        expected = 210.0 * (1 - 0.3) / (1.3 * 0.4)
        np.testing.assert_allclose(d[0, 0], expected)

    def test_solid_d_matrix(self) -> None:
        mat = LinearElastic(e_modulus=100.0, poisson=0.25, state=StressState.SOLID)
        d = mat.d_matrix()
        assert d.shape == (6, 6)
        # 对称性
        np.testing.assert_allclose(d, d.T)
        # D11 = E(1-nu)/((1+nu)(1-2nu)) = 100*0.75/(1.25*0.5) = 120
        np.testing.assert_allclose(d[0, 0], 120.0)
        # 剪切项 G = E/(2(1+nu)) = 40
        np.testing.assert_allclose(d[3, 3], 40.0)

    def test_shear_modulus(self) -> None:
        mat = LinearElastic(e_modulus=100.0, poisson=0.25)
        np.testing.assert_allclose(mat.shear_modulus, 40.0)

    def test_rejects_nonpositive_modulus(self) -> None:
        with pytest.raises(ElementError, match="弹性模量"):
            LinearElastic(e_modulus=0.0)

    def test_rejects_poisson_out_of_range(self) -> None:
        with pytest.raises(ElementError, match="泊松比"):
            LinearElastic(e_modulus=1.0, poisson=0.5)

    def test_d_matrix_symmetric(self) -> None:
        for state in StressState:
            d = LinearElastic(210.0, 0.3, state).d_matrix()
            np.testing.assert_allclose(d, d.T)


class TestSection:
    def test_defaults(self) -> None:
        section = Section()
        assert section.area == 1.0
        assert section.thickness == 1.0

    def test_rejects_nonpositive(self) -> None:
        with pytest.raises(ElementError, match="面积"):
            Section(area=0.0)
        with pytest.raises(ElementError, match="厚度"):
            Section(thickness=-1.0)

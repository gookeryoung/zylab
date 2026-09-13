"""3D 视口：pyvistaqt 封装，提供网格加载/渲染/截图等基础能力.

QtInteractor 直接嵌入 QWidget 布局，与 meshio / VTK 数据结构对接；
延迟初始化 VTK 渲染器以避免导入即启动的开销。

典型用法::

    view = VtkView()
    view.load_mesh_file("input.vtu")
    view.fit_all()
    view.save_screenshot("output.png")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..qt_compat import QImage, QWidget

if TYPE_CHECKING:
    import numpy as np
    import pyvista
    import pyvistaqt

logger = logging.getLogger(__name__)

__all__ = ["VtkView"]


class VtkView(QWidget):
    """pyvistaqt 3D 视口：网格渲染 + 交互 + 截图."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化视口（延迟创建 VTK 渲染器，按需触发）."""
        super().__init__(parent)
        self._interactor: pyvistaqt.QtInteractor | None = None
        self._current_mesh: pyvista.DataSet | None = None

    # ---------------------------------------------------------------- 生命周期

    def _ensure_interactor(self) -> pyvistaqt.QtInteractor:
        """延迟创建 QtInteractor（VTK 渲染器首次使用时才初始化）."""
        if self._interactor is not None:
            return self._interactor
        from pyvistaqt import QtInteractor

        self._interactor = QtInteractor(self)
        self._attach_to_layout(self._interactor)
        return self._interactor

    def _attach_to_layout(self, child: QWidget) -> None:
        """把延迟创建的 QtInteractor 挂到父 QWidget 布局上."""
        from ..qt_compat import QVBoxLayout

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(child)
        self.setLayout(layout)

    # -------------------------------------------------------------- 数据加载

    def load_mesh_file(self, path: str | Path) -> None:
        """从磁盘加载网格文件（支持 meshio 可读的全部格式）.

        Args:
            path: 网格文件路径（.vtu / .vtk / .msh / .bdf / .med …）。
        """
        import pyvista as pv

        pv_mesh = pv.read(str(path))
        self._show_mesh(pv_mesh)

    def load_unstructured_grid(self, points: np.ndarray, cells: dict) -> None:
        """从 numpy 数组构造 vtkUnstructuredGrid 并显示.

        Args:
            points: (N, 3) 顶点坐标数组。
            cells: 单元字典，key 为 VTK cell type（int），value 为单元连接数组。
        """
        import pyvista as pv

        pv_mesh = pv.UnstructuredGrid(cells, points)
        self._show_mesh(pv_mesh)

    def load_polydata(self, pv_mesh: pyvista.PolyData) -> None:
        """直接显示 pyvista.PolyData."""
        self._show_mesh(pv_mesh)

    def _show_mesh(self, pv_mesh: pyvista.DataSet) -> None:
        """内部：添加 mesh 到 renderer 并刷新视口."""
        self._current_mesh = pv_mesh
        inter = self._ensure_interactor()
        inter.renderer.clear()
        inter.renderer.add_mesh(pv_mesh)
        inter.renderer.reset_camera()
        inter.render()

    # ---------------------------------------------------------------- 渲染控制

    def fit_all(self) -> None:
        """重置相机以完整显示当前 mesh."""
        if self._interactor is None or self._current_mesh is None:
            return
        self._interactor.renderer.reset_camera()
        self._interactor.render()

    def clear(self) -> None:
        """清除当前 mesh."""
        if self._interactor is None:
            return
        self._interactor.renderer.clear()
        self._current_mesh = None
        self._interactor.render()

    # ---------------------------------------------------------------- 截图

    def save_screenshot(self, path: str | Path) -> None:
        """把当前渲染帧保存为 PNG."""
        if self._interactor is None:
            logger.warning("VtkView.save_screenshot: 尚未渲染，跳过")
            return
        self._interactor.screenshot(str(path))

    def take_qimage(self) -> QImage | None:
        """把当前渲染帧拷贝为 QImage（供报告嵌入 / clipboard）."""
        if self._interactor is None:
            return None
        import numpy as np
        from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter

        renderer = self._interactor.renderer
        win = renderer.GetRenderWindow()
        win.Render()
        v2i = vtkWindowToImageFilter()
        v2i.SetInput(win)
        v2i.ReadFrontBufferOff()
        v2i.Update()
        vtk_img = v2i.GetOutput()

        dims = vtk_img.GetDimensions()
        arr = vtk_img.GetPointData().GetScalars()
        np_img = np.frombuffer(arr, dtype=np.uint8).reshape(dims[1], dims[0], 3)
        # VTK 默认 BGR → RGB 翻转
        np_img = np_img[:, ::-1, ::-1].copy()
        h, w, ch = np_img.shape
        qimg = QImage(np_img.data, w, h, ch * w, QImage.Format_RGB888)
        return qimg.copy()  # 拷贝避免 numpy buffer 释放后悬空

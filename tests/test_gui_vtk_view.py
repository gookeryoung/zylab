"""VtkView 单元测试（纯 API 层，延迟初始化 VTK 渲染器不启动 QVTK 窗口）."""

from __future__ import annotations

from zylab.gui.widgets.vtk_view import VtkView


class _FakeRenderer:
    """Fake pyvista renderer 占位."""

    def __init__(self) -> None:
        self._meshes: list[object] = []
        self._reset_called = 0
        self._clear_called = 0

    def add_mesh(self, m: object) -> None:
        self._meshes.append(m)

    def clear(self) -> None:
        self._clear_called += 1

    def reset_camera(self) -> None:
        self._reset_called += 1

    def render(self) -> None:
        pass


class _FakeQtInteractor:
    """Fake pyvistaqt.QtInteractor（非 QWidget，配合 monkeypatch _attach_to_layout 使用）."""

    renderer = None

    def __init__(self, parent: object) -> None:
        self.parent = parent
        _FakeQtInteractor.renderer = _FakeRenderer()

    def setSizePolicy(self, sp: object) -> None:
        pass

    def render(self) -> None:
        pass

    def screenshot(self, path: str) -> None:
        pass


class _FakePolyData:
    pass


class TestVtkView:
    """VtkView 基础 API（不触发 VTK 渲染器创建的分支）."""

    def test_init_lazy(self, qtbot) -> None:
        """构造时不立即创建 QtInteractor."""
        view = VtkView()
        assert view._interactor is None
        assert view._current_mesh is None

    def test_clear_noop_when_empty(self, qtbot) -> None:
        """空 view 调 clear() 安全 no-op."""
        view = VtkView()
        view.clear()

    def test_fit_all_noop_when_empty(self, qtbot) -> None:
        """空 view 调 fit_all() 安全 no-op."""
        view = VtkView()
        view.fit_all()

    def test_save_screenshot_noop_when_empty(self, qtbot, caplog) -> None:
        """空 view 调 save_screenshot() 不崩溃并打 warning."""
        view = VtkView()
        view.save_screenshot("/tmp/should_not_create.png")
        assert any("尚未渲染" in r.message for r in caplog.records)

    def test_take_qimage_returns_none_when_empty(self, qtbot) -> None:
        """空 view 调 take_qimage() 返回 None."""
        view = VtkView()
        assert view.take_qimage() is None

    def test_ensure_interactor_and_show_mesh(self, qtbot, monkeypatch) -> None:
        """_ensure_interactor 延迟创建 + _show_mesh 调用 mock renderer."""
        import pyvistaqt as real_pvqt

        monkeypatch.setattr(real_pvqt, "QtInteractor", _FakeQtInteractor)
        monkeypatch.setattr(VtkView, "_attach_to_layout", lambda self, child: None)

        view = VtkView()
        inter = view._ensure_interactor()
        assert inter is view._interactor
        assert view._interactor is not None

        # 再调用应复用已创建的 interactor
        inter2 = view._ensure_interactor()
        assert inter2 is inter

        # _show_mesh 应调用 renderer.clear + add_mesh + reset_camera
        pd = _FakePolyData()
        view._show_mesh(pd)
        assert view._current_mesh is pd
        assert _FakeQtInteractor.renderer._clear_called == 1
        assert _FakeQtInteractor.renderer._reset_called == 1
        assert len(_FakeQtInteractor.renderer._meshes) == 1

    def test_fit_and_clear_after_mesh(self, qtbot, monkeypatch) -> None:
        """有 mesh 时 fit_all / clear 能调用 renderer."""
        import pyvistaqt as real_pvqt

        monkeypatch.setattr(real_pvqt, "QtInteractor", _FakeQtInteractor)
        monkeypatch.setattr(VtkView, "_attach_to_layout", lambda self, child: None)

        view = VtkView()
        pd = _FakePolyData()
        view._show_mesh(pd)

        view.fit_all()
        assert _FakeQtInteractor.renderer._reset_called == 2  # show_mesh + fit_all

        view.clear()
        assert view._current_mesh is None
        assert _FakeQtInteractor.renderer._clear_called == 2  # show_mesh + clear


class _FakeWindow:
    def Render(self):
        pass


class _FakeVtkImg:
    def GetDimensions(self):
        return (10, 20, 1)

    def GetPointData(self):
        class _PD:
            def GetScalars(self):
                import numpy as np

                return np.zeros(20 * 10 * 3, dtype=np.uint8)

        return _PD()


class _FakeV2I:
    def SetInput(self, win):
        pass

    def ReadFrontBufferOff(self):
        pass

    def Update(self):
        pass

    def GetOutput(self):
        return _FakeVtkImg()


def test_vtk_save_screenshot_after_mesh(qtbot, monkeypatch):
    """有 interactor 时 save_screenshot 走非空分支."""
    import pyvistaqt as real_pvqt

    monkeypatch.setattr(real_pvqt, "QtInteractor", _FakeQtInteractor)
    monkeypatch.setattr(VtkView, "_attach_to_layout", lambda self, child: None)

    view = VtkView()
    view._ensure_interactor()  # lazy create
    # screenshot 方法已经在 FakeQtInteractor 里了

    # spy: 确认 interactor 被调 screenshot
    called = []

    def _spy(path):
        called.append(path)

    view._interactor.screenshot = _spy
    view.save_screenshot("/tmp/out.png")
    assert called == ["/tmp/out.png"]


def test_vtk_take_qimage_returns_qimage(qtbot, monkeypatch):
    """有 interactor 时 take_qimage 走 VTK 流程."""
    import pyvistaqt as real_pvqt

    monkeypatch.setattr(real_pvqt, "QtInteractor", _FakeQtInteractor)
    monkeypatch.setattr(VtkView, "_attach_to_layout", lambda self, child: None)

    # monkeypatch vtkWindowToImageFilter
    import vtkmodules.vtkRenderingCore as vrc

    monkeypatch.setattr(vrc, "vtkWindowToImageFilter", _FakeV2I)

    view = VtkView()
    inter = view._ensure_interactor()
    inter.renderer.GetRenderWindow = _FakeWindow

    qimg = view.take_qimage()
    assert qimg is not None

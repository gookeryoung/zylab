"""gui.pages.studio_page 工作台页测试：模板实例化、参数联动、运行编排、节点交互."""

from __future__ import annotations

import pytest

from zylab.core.executor import EventKind
from zylab.gui.pages.studio_page import StudioPage, _StudioBridge
from zylab.gui.qt_compat import Qt
from zylab.studio import NodeRunEvent, NodeState


def _select_template(page: StudioPage, template_id: str) -> None:
    """按模板 id 切换模板库选择."""
    for row in range(page._template_list.count()):
        if page._template_list.item(row).data(Qt.UserRole) == template_id:
            page._template_list.setCurrentRow(row)
            return
    raise AssertionError(f"模板不存在: {template_id}")


@pytest.mark.gui
def test_page_builds_with_first_template(qtbot) -> None:
    """页面装配：模板库 + 画布卡片 + 参数表单 + 源节点模型预览."""
    page = StudioPage()
    qtbot.addWidget(page)
    assert page._template_list.count() >= 6
    assert page._graph is not None
    assert page._canvas.card_rect("model") is not None
    assert page._param_form._fields
    # 源节点进程内预览：已完成 + 模型线框
    assert page._graph.node("model").state is NodeState.UP_TO_DATE
    assert "模型预览" in page._result_view._summary.text()
    page.shutdown()


@pytest.mark.gui
def test_template_switch_rebuilds_graph(qtbot) -> None:
    """切换模板重建图与画布."""
    page = StudioPage()
    qtbot.addWidget(page)
    _select_template(page, "structural.truss_nonlinear")
    assert page._graph.template.id == "structural.truss_nonlinear"
    assert page._canvas.card_rect("solve") is not None
    assert ("model", "rise") in page._param_form._fields
    page.shutdown()


@pytest.mark.gui
def test_param_edit_marks_dirty(qtbot) -> None:
    """参数编辑级联失效（模型与下游回到待运行）."""
    page = StudioPage()
    qtbot.addWidget(page)
    _select_template(page, "structural.cantilever_static")
    page._on_param_edited("model", "nx", 20)
    assert page._graph.node("model").state is NodeState.READY
    assert "需重新运行" in page._status_label.text()
    page.shutdown()


@pytest.mark.gui
def test_run_all_static_template(qtbot) -> None:
    """运行全部：子进程求解完成，结果视图呈现位移摘要（真实进程端到端）."""
    page = StudioPage()
    qtbot.addWidget(page)
    _select_template(page, "structural.cantilever_static")
    page._on_run_all()
    assert not page._run_all_button.isEnabled()
    qtbot.waitUntil(page._run_all_button.isEnabled, timeout=60000)
    assert page._graph.node("solve").state is NodeState.UP_TO_DATE
    assert "最大位移" in page._result_view._summary.text()
    page.shutdown()


@pytest.mark.gui
def test_node_click_shows_cached_result(qtbot) -> None:
    """单击已完成节点呈现其结果."""
    page = StudioPage()
    qtbot.addWidget(page)
    _select_template(page, "structural.cantilever_static")
    page._on_run_all()
    qtbot.waitUntil(lambda: page._graph.node("solve").state is NodeState.UP_TO_DATE, timeout=60000)
    qtbot.waitUntil(page._run_all_button.isEnabled, timeout=10000)
    page._result_view.clear()
    page._on_node_clicked("solve")
    assert "最大位移" in page._result_view._summary.text()
    page._on_node_clicked("model")
    assert "模型预览" in page._result_view._summary.text()
    page.shutdown()


@pytest.mark.gui
def test_double_click_runs_to_node(qtbot) -> None:
    """双击待运行节点触发级联运行."""
    page = StudioPage()
    qtbot.addWidget(page)
    _select_template(page, "structural.truss_nonlinear")
    page._on_node_double_clicked("solve")
    qtbot.waitUntil(lambda: page._graph.node("solve").state is NodeState.UP_TO_DATE, timeout=60000)
    qtbot.waitUntil(page._run_all_button.isEnabled, timeout=10000)
    assert "Newton" in page._result_view._summary.text()
    page.shutdown()


@pytest.mark.gui
def test_cancel_running(qtbot) -> None:
    """取消运行：状态复位且按钮恢复."""
    page = StudioPage()
    qtbot.addWidget(page)
    _select_template(page, "structural.cantilever_harmonic")
    page._param_form._fields[("solve", "n_freq")].setValue(200)  # 拉长运行时间确保可取消
    page._on_run_all()
    qtbot.waitUntil(page._cancel_button.isEnabled, timeout=10000)
    page._on_cancel()
    assert page._run_all_button.isEnabled()
    assert "已取消" in page._status_label.text()
    page.shutdown()


@pytest.mark.gui
def test_node_menu_build(qtbot) -> None:
    """节点上下文菜单：查看结果仅已完成时可用."""
    page = StudioPage()
    qtbot.addWidget(page)
    menu = page._build_node_menu("solve")
    actions = [a.text() for a in menu.actions()]
    assert actions == ["运行到此节点", "强制重新运行", "查看结果"]
    assert not menu.actions()[2].isEnabled()  # 未运行不可查看
    page.shutdown()


@pytest.mark.gui
def test_double_click_up_to_date_shows_result(qtbot) -> None:
    """双击已完成节点直接查看结果（不触发运行）."""
    page = StudioPage()
    qtbot.addWidget(page)
    page._result_view.clear()  # model 预览已就绪；先清空再双击查看
    page._on_node_double_clicked("model")
    assert "模型预览" in page._result_view._summary.text()
    page.shutdown()


@pytest.mark.gui
def test_reference_load_from_bundle(qtbot) -> None:
    """屈曲参考载荷取上游模型工况载荷合计."""
    page = StudioPage()
    qtbot.addWidget(page)
    _select_template(page, "structural.column_buckling")
    assert page._reference_load("solve") == pytest.approx(1.0)  # 顶部单位轴力
    page.shutdown()


@pytest.mark.gui
def test_bridge_dispatch(qtbot) -> None:
    """事件桥：四类节点事件转译为 Qt 信号."""
    bridge = _StudioBridge()
    seen: dict[str, list] = {"s": [], "p": [], "r": [], "f": []}
    bridge.node_started.connect(seen["s"].append)
    bridge.node_progress.connect(lambda nid, p, m: seen["p"].append((nid, p, m)))
    bridge.node_result.connect(lambda nid, _result: seen["r"].append(nid))
    bridge.node_failed.connect(lambda _nid, msg: seen["f"].append(msg))
    bridge.dispatch(NodeRunEvent("a", EventKind.STARTED))
    bridge.dispatch(NodeRunEvent("a", EventKind.PROGRESS, (0.5, "装配")))
    bridge.dispatch(NodeRunEvent("a", EventKind.RESULT, object()))
    bridge.dispatch(NodeRunEvent("a", EventKind.ERROR, "错误: x"))
    assert seen["s"] == ["a"]
    assert seen["p"] == [("a", 0.5, "装配")]
    assert len(seen["r"]) == 1
    assert seen["f"] == ["错误: x"]


@pytest.mark.gui
def test_guards_and_misc(qtbot) -> None:
    """守卫路径：空图/运行中切换/预览失败/取消空跑."""
    page = StudioPage()
    qtbot.addWidget(page)

    # 空图守卫
    page._graph = None
    page._on_run_all()
    page._on_cancel()
    page._on_node_clicked("model")
    page._on_node_double_clicked("model")
    page._on_node_context_menu("model", None)
    page._on_node_progress("model", 0.5, "消息")
    page._on_param_edited("model", "nx", 20)
    page._show_node_result("model")
    assert page._reference_load("solve") == 1.0

    # 负行号选择为空操作
    page._on_template_selected(-1)
    assert page._graph is None
    page.shutdown()


@pytest.mark.gui
def test_template_switch_reverts_while_running(qtbot) -> None:
    """运行中切换模板回退选择（图保持不变）."""
    page = StudioPage()
    qtbot.addWidget(page)
    active = page._active_row
    template_id = page._graph.template.id

    class _RunningStub:
        """运行中 runner 替身."""

        running = True

    page._runner = _RunningStub()
    target_row = (active + 1) % page._template_list.count()
    page._template_list.setCurrentRow(target_row)  # 触发选择 -> 应被回退
    assert page._template_list.currentRow() == active
    assert page._graph.template.id == template_id
    page._runner = None
    page.shutdown()


@pytest.mark.gui
def test_preview_failure_marks_failed(qtbot, monkeypatch) -> None:
    """源节点预览失败标记 FAILED 且不阻断页面."""
    page = StudioPage()
    qtbot.addWidget(page)

    def _boom(target: str):
        raise RuntimeError("预览故障")

    monkeypatch.setattr("zylab.gui.pages.studio_page._resolve_target", _boom)
    page._on_template_selected(page._active_row)  # 重新实例化当前模板
    assert page._graph.node("model").state is NodeState.FAILED
    page.shutdown()


@pytest.mark.gui
def test_node_failed_shows_error(qtbot) -> None:
    """节点失败事件：结果视图错误着色 + 状态标签."""
    page = StudioPage()
    qtbot.addWidget(page)
    page._on_node_failed("solve", "SolverError: 矩阵奇异")
    assert page._result_view._summary.objectName() == "errorText"
    assert "失败" in page._status_label.text()
    page.shutdown()


@pytest.mark.gui
def test_refresh_theme(qtbot) -> None:
    """主题刷新分发到画布与结果视图."""
    page = StudioPage()
    qtbot.addWidget(page)
    page.refresh_theme()
    page.shutdown()


@pytest.mark.gui
def test_save_template_as(qtbot, tmp_path) -> None:
    """另存为模板：注册 + 写文件 + 列表新增并选中."""
    page = StudioPage(data_dir=tmp_path)
    qtbot.addWidget(page)
    before = page._template_list.count()
    _select_template(page, "structural.cantilever_static")
    page._graph.set_param("model", "nx", 12)
    template = page._save_template_as("我的悬臂梁")
    assert template is not None
    assert template.id == "user.我的悬臂梁"
    assert template.node("model").params["nx"] == 12  # 当前参数已内嵌
    assert (tmp_path / "templates" / "user.我的悬臂梁.json").exists()
    assert page._template_list.count() == before + 1
    # 重名自动加后缀
    again = page._save_template_as("我的悬臂梁")
    assert again is not None and again.id == "user.我的悬臂梁_2"
    # 空名拒绝
    assert page._save_template_as("  ") is None
    page.shutdown()


@pytest.mark.gui
def test_project_save_and_load(qtbot, tmp_path) -> None:
    """工程保存/打开：模板与参数自包含回读."""
    page = StudioPage(data_dir=tmp_path)
    qtbot.addWidget(page)
    _select_template(page, "structural.truss_nonlinear")
    page._graph.set_param("model", "rise", 0.8)
    path = tmp_path / "case.zprj"
    page._save_project(path)
    assert path.exists()

    # 新页面打开工程：内嵌模板注册并实例化，参数还原
    page2 = StudioPage(data_dir=tmp_path / "other")
    qtbot.addWidget(page2)
    page2._load_project(path)
    assert page2._graph.template.id == "structural.truss_nonlinear"
    assert page2._graph.node("model").params["rise"] == 0.8
    assert "已打开" in page2._status_label.text()
    page.shutdown()
    page2.shutdown()


@pytest.mark.gui
def test_load_project_bad_file(qtbot, tmp_path) -> None:
    """打开非法工程文件：状态标签报错不崩溃."""
    page = StudioPage(data_dir=tmp_path)
    qtbot.addWidget(page)
    bad = tmp_path / "bad.zprj"
    bad.write_text("not a hdf5", encoding="utf-8")
    page._load_project(bad)
    assert "打开失败" in page._status_label.text()
    page.shutdown()

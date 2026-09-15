"""core.runtime_config 运行时配置测试."""

from __future__ import annotations

from zylab.core import (
    AUTOSAVE_MAX,
    AUTOSAVE_MIN,
    DEFAULT_MAX_WORKERS,
    SOLVER_TIMEOUT_MAX,
    WORKERS_MAX,
    WORKERS_MIN,
    get_autosave_interval,
    get_max_workers,
    get_solver_timeout,
    get_workspace_history_limit,
    update_runtime_config,
)


class TestRuntimeConfig:
    """运行时配置模块级状态与 getter/setter 行为测试."""

    def test_default_values(self) -> None:
        """模块初始值符合预期（默认 workers ≈ CPU 核数 / 2）."""
        assert get_max_workers() == DEFAULT_MAX_WORKERS
        assert get_solver_timeout() == 0  # 0 = 不限制
        assert get_autosave_interval() == 60
        assert get_workspace_history_limit() == 10

    def test_update_single_field(self) -> None:
        """单独更新 max_workers 应立即生效，其他字段不受影响."""
        before_workers = get_max_workers()
        before_timeout = get_solver_timeout()
        before_autosave = get_autosave_interval()
        before_limit = get_workspace_history_limit()

        updated = update_runtime_config(max_workers=4)

        assert get_max_workers() == 4
        assert get_solver_timeout() == before_timeout
        assert get_autosave_interval() == before_autosave
        assert get_workspace_history_limit() == before_limit
        assert updated == {"max_workers": 4}

        # 还原
        update_runtime_config(max_workers=before_workers)

    def test_update_multiple_fields(self) -> None:
        """一次调用批量更新多个字段."""
        update_runtime_config(
            max_workers=8,
            solver_timeout_s=300,
            autosave_interval_sec=120,
            workspace_history_limit=20,
        )
        assert get_max_workers() == 8
        assert get_solver_timeout() == 300
        assert get_autosave_interval() == 120
        assert get_workspace_history_limit() == 20

        # 全还原
        update_runtime_config(
            max_workers=DEFAULT_MAX_WORKERS,
            solver_timeout_s=0,
            autosave_interval_sec=60,
            workspace_history_limit=10,
        )

    def test_noop_when_same_value(self) -> None:
        """传入值与当前值相同，不应产生变更条目."""
        current = get_max_workers()
        updated = update_runtime_config(max_workers=current)
        assert updated == {}

    def test_update_unknown_key_ignored(self) -> None:
        """未知字段静默忽略，不报错."""
        before = get_max_workers()
        updated = update_runtime_config(nonexistent_field=42)  # type: ignore[arg-type]
        assert updated == {}
        assert get_max_workers() == before

    def test_update_out_of_range_gets_clamped(self) -> None:
        """越界值应自动夹紧，不抛异常."""
        # 先把 max_workers 调到中间值，避免 DEFAULT_MAX_WORKERS == WORKERS_MIN/MAX
        update_runtime_config(max_workers=min(WORKERS_MAX - 1, max(WORKERS_MIN + 1, 4)))

        # max_workers 小于 WORKERS_MIN → 夹紧到 WORKERS_MIN
        updated = update_runtime_config(max_workers=-5)
        assert get_max_workers() == WORKERS_MIN
        assert updated == {"max_workers": WORKERS_MIN}

        # max_workers 大于 WORKERS_MAX → 夹紧到 WORKERS_MAX
        updated = update_runtime_config(max_workers=9999)
        assert get_max_workers() == WORKERS_MAX
        assert updated == {"max_workers": WORKERS_MAX}

        # 先把 autosave_interval_sec 调到中间值
        update_runtime_config(autosave_interval_sec=120)
        # autosave_interval_sec 小于 0 → 夹紧到 AUTOSAVE_MIN
        updated = update_runtime_config(autosave_interval_sec=-10)
        assert get_autosave_interval() == AUTOSAVE_MIN

        # autosave_interval_sec 大于 AUTOSAVE_MAX → 夹紧到 AUTOSAVE_MAX
        updated = update_runtime_config(autosave_interval_sec=99999)
        assert get_autosave_interval() == AUTOSAVE_MAX

        # solver_timeout_s 超范围夹紧
        updated = update_runtime_config(solver_timeout_s=999999)
        assert get_solver_timeout() == SOLVER_TIMEOUT_MAX

        # 先把 workspace_history_limit 调到中间值
        update_runtime_config(workspace_history_limit=50)
        # workspace_history_limit 超范围夹紧（负数）
        updated = update_runtime_config(workspace_history_limit=-1)
        assert get_workspace_history_limit() == 1

        # 还原
        update_runtime_config(
            max_workers=DEFAULT_MAX_WORKERS,
            autosave_interval_sec=60,
            solver_timeout_s=0,
            workspace_history_limit=10,
        )

    def test_range_constants_visible(self) -> None:
        """对外导出的范围常量应符合预期."""
        assert WORKERS_MIN == 1
        assert WORKERS_MAX == 64
        assert SOLVER_TIMEOUT_MAX == 86400
        assert AUTOSAVE_MIN == 0
        assert AUTOSAVE_MAX == 3600

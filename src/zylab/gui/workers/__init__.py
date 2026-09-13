"""zylab.gui.workers —— 后台任务调度（QThread + QRunnable）.

与 :class:`zylab.core.executor.ProcessExecutor` 的分工：
- **ProcessExecutor**：重任务（FEA 求解、DOE 采样、优化迭代），独立子进程崩溃隔离；
- **QThread / QRunnable**（本模块）：轻量后台任务（短耗时、低 CPU、可随时取消），
  同进程零拷贝，进度轮询 / 日志收集 / UI 数据预加载等场景优先走此通道。
"""

from __future__ import annotations

from .task_worker import TaskWorker, WorkerController
from .thread_pool import WorkerTask, WorkerThreadPool, worker_pool

__all__ = [
    "TaskWorker",
    "WorkerController",
    "WorkerTask",
    "WorkerThreadPool",
    "worker_pool",
]

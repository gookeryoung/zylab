"""zylab GUI 启动入口：供 ``zylab`` 命令与 fspack 打包运行.

fspack 用 runpy 执行整个入口模块（``[project.scripts]`` 的 ``:function``
部分被忽略），故本模块不得被 ``zylab.gui`` 门面 re-export，否则 runpy
执行时会因模块已在 sys.modules 而产生 RuntimeWarning。模块级 ``main``
仅做转发，实际实现位于 ``zylab.gui.app``；``main()`` 调用置于
``__main__`` 保护块内，保证 exe 导入 ``:main`` 与 runpy 整模块执行两种
方式均可正常工作.
"""

from __future__ import annotations


def main() -> int:
    """转发到 GUI 应用入口，供 ``[project.scripts]`` 引用."""
    from .app import main as _app_main

    return _app_main()


if __name__ == "__main__":
    raise SystemExit(main())

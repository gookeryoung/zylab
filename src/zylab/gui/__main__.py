"""zylab GUI 启动入口：供 ``python -m zylab.gui`` 与 fspack 打包运行.

fspack 用 runpy 执行整个入口模块（``[project.scripts]`` 的 ``:function``
部分被忽略），故 ``main()`` 调用须置于 ``__main__`` 保护块内；入口模块
不被 ``zylab.gui`` 门面提前导入，避免 runpy 执行时的 sys.modules 冲突警告
（RuntimeWarning: found in sys.modules after import of package）.
"""

from __future__ import annotations

if __name__ == "__main__":
    from .app import main

    raise SystemExit(main())

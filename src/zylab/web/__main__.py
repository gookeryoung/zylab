"""zylab Web 启动入口：供 ``zylab-web`` 命令运行（等价 ``manage.py``）.

用法示例::

    zylab-web runserver          # 开发服务器（dev 设置）
    zylab-web check              # Django 系统检查

生产部署经 gunicorn 直接指向 ``zylab.web.wsgi:application``（prod 设置）。
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """执行 Django 管理命令（runserver/check/migrate 等）.

    :param argv: 命令行参数（默认 ``sys.argv``，argv[0] 为程序名）。
    :return: 进程退出码；管理命令自身的 ``SystemExit``（如参数错误）向上传播。
    """
    if sys.version_info < (3, 10):
        print("zylab Web 需要 Python 3.10+（Django 5.2 LTS）运行时，桌面线请使用 zylab 命令")
        return 1
    import os

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "zylab.web.settings.dev")
    from django.core.management import execute_from_command_line

    execute_from_command_line(argv if argv is not None else list(sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

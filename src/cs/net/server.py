"""静态文件服务：把目录以 HTTP 方式临时共享出去（浏览器预览、局域网传文件）。

替代旧的 ``create_img_http_server``——那个实现 ``os.chdir`` 到目标目录再用
``os.system("python -m http.server")`` 阻塞，既改全局工作目录、又拼 shell 字符串，
Ctrl-C 之后目录还没切回来。这里用 ``ThreadingHTTPServer`` + 后台线程，
``stop()`` 可确定性地释放端口。

>>> with StaticServer("output") as server:     # doctest: +SKIP
...     print(server.url)
http://127.0.0.1:53124/
"""

from __future__ import annotations

import functools
import logging
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self

from .errors import PortInUseError
from .models import Address

__all__ = ["StaticServer", "serve_static"]

log = logging.getLogger(__name__)


class _Handler(SimpleHTTPRequestHandler):
    """访问日志走 logging，不往 stderr 刷屏。"""

    server_version = "cs-net"

    def log_message(self, format: str, *args: Any) -> None:
        log.debug("%s - %s", self.address_string(), format % args)


class StaticServer:
    """在后台线程提供目录浏览与文件下载。

    ``port=0`` 时由内核分配，实际端口见 :attr:`port`（``start()`` 之后才有意义）。
    """

    __slots__ = ("_httpd", "_thread", "directory", "host", "port")

    def __init__(
        self,
        directory: str | Path,
        *,
        port: int = 0,
        host: str = "127.0.0.1",
    ) -> None:
        path = Path(directory).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"目录不存在：{path}")
        if not path.is_dir():
            raise NotADirectoryError(f"不是目录：{path}")
        self.directory = path.resolve()
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------- 生命周期
    def start(self) -> StaticServer:
        """启动监听（幂等），返回自身以便链式调用。"""
        if self._httpd is not None:
            return self

        handler = functools.partial(_Handler, directory=str(self.directory))
        try:
            httpd = ThreadingHTTPServer((self.host, self.port), handler)
        except OSError as exc:
            raise PortInUseError(
                f"无法监听 {self.host or '0.0.0.0'}:{self.port}：{exc}"
            ) from exc

        httpd.daemon_threads = True
        self._httpd = httpd
        self.port = int(httpd.server_address[1])
        self._thread = threading.Thread(
            target=httpd.serve_forever,
            name=f"cs-net-http-{self.port}",
            daemon=True,
        )
        self._thread.start()
        log.debug("静态服务已启动：%s -> %s", self.url, self.directory)
        return self

    def stop(self) -> None:
        """停止监听并释放端口（幂等）。"""
        httpd, thread = self._httpd, self._thread
        self._httpd = self._thread = None
        if httpd is None:
            return
        httpd.shutdown()
        httpd.server_close()
        if thread is not None:
            thread.join(timeout=5.0)
        log.debug("静态服务已停止：%s", self.url)

    def wait(self) -> None:
        """阻塞到服务被 ``stop()`` 或 Ctrl-C 中断。"""
        thread = self._thread
        if thread is None:
            return
        try:
            while thread.is_alive():
                thread.join(timeout=0.5)
        except KeyboardInterrupt:
            pass

    # ---------------------------------------------------------- 属性
    @property
    def running(self) -> bool:
        return self._httpd is not None

    @property
    def address(self) -> Address:
        """监听地址；``host`` 为空（所有接口）时对外展示 ``127.0.0.1``。"""
        return Address(self.host or "127.0.0.1", self.port)

    @property
    def url(self) -> str:
        return f"http://{self.address}/"

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "host": self.address.host,
            "port": self.port,
            "directory": str(self.directory),
            "running": self.running,
        }

    # ---------------------------------------------------------- 上下文
    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def __repr__(self) -> str:
        state = "running" if self.running else "stopped"
        return f"StaticServer({self.url!r}, {state}, directory={str(self.directory)!r})"


def serve_static(
    directory: str | Path,
    port: int = 0,
    *,
    host: str = "127.0.0.1",
    block: bool = False,
) -> StaticServer:
    """一行启动静态服务。

    ``block=True`` 时行为与旧的 ``python -m http.server`` 一致（Ctrl-C 退出），
    否则立刻返回服务对象，由调用方决定何时 ``stop()``。
    """
    server = StaticServer(directory, port=port, host=host).start()
    if block:
        try:
            server.wait()
        finally:
            server.stop()
    return server

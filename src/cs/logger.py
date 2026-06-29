"""
日志器组件
"""

import logging
import logging.handlers
import time
from enum import Enum
from functools import wraps
from pathlib import Path
from collections.abc import Callable


class LogType(Enum):
    FILE_SIZE = "filesize"
    TIMED = "timed"


class RunMode(Enum):
    DEV = "dev"
    TEST = "test"
    RELEASE = "release"


_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class Logger:
    """统一的日志器封装，支持按大小/按时间轮转，以及开发/测试/发布三种运行模式。"""

    def __init__(
        self,
        file: str | Path,
        name: str = "logger",
        log_type: LogType = LogType.FILE_SIZE,
        run_mode: RunMode = RunMode.TEST,
    ) -> None:
        self.file = Path(file)
        self.name = name
        self.log_type = log_type
        self.run_mode = run_mode

        self._logger = self._build_logger()

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        # 防止重复添加 handler
        if logger.handlers:
            return logger

        logger.setLevel(self._get_log_level())
        formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

        # 控制台输出（仅在 DEV / TEST 模式）
        if self.run_mode in (RunMode.DEV, RunMode.TEST):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        # 文件 handler
        if self.log_type == LogType.FILE_SIZE:
            file_handler = self._build_filesize_handler(formatter)
        else:
            file_handler = self._build_timed_handler(formatter)
        logger.addHandler(file_handler)

        return logger

    def _get_log_level(self) -> int:
        if self.run_mode == RunMode.DEV:
            return logging.DEBUG
        return logging.INFO

    def _build_filesize_handler(
        self,
        formatter: logging.Formatter,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 10,
    ) -> logging.Handler:
        handler = logging.handlers.RotatingFileHandler(
            filename=self.file,
            mode="a",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        return handler

    def _build_timed_handler(
        self,
        formatter: logging.Formatter,
        when: str = "midnight",
        interval: int = 4,
        backup_count: int = 10,
    ) -> logging.Handler:
        # 微秒级时间戳后缀，避免多进程写同一文件冲突，年份取后两位
        timestamp = time.strftime("%y%m%d%H%M%S", time.localtime())
        microsecond = f"{time.time() % 1:.6f}"[2:]
        log_file = f"{self.file}_{timestamp}{microsecond}"

        handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_file,
            when=when,
            interval=interval,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.suffix = "%Y-%m-%d.log"
        handler.setFormatter(formatter)
        return handler

    # ---- 代理方法 ----
    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    def critical(self, msg: str) -> None:
        self._logger.critical(msg)


def timed_log(show_time: bool = False) -> Callable:
    """装饰器：记录被调用方法的类名.方法名，可选展示耗时。"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            cls_name = args[0].__class__.__name__ if args else ""
            log_msg = f"{cls_name}.{func.__name__}" if cls_name else func.__name__

            if show_time:
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info("%s (%.4fs)", log_msg, elapsed)
            else:
                result = func(*args, **kwargs)
                logger.info(log_msg)
            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    logger = Logger(file="test.log", run_mode=RunMode.DEV)
    logger.debug("这是一条 debug 日志")
    logger.info("这是一条 info 日志")
    logger.warning("这是一条 warning 日志")
    logger.error("这是一条 error 日志")

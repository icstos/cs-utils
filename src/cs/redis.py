"""
Redis - redis-py 连接池封装 (Python 3.12+)

特性:
- 连接池复用 + 健康检查 + TCP keepalive
- 常用 KV / Hash / List 操作，带默认值与批量接口
- 支持上下文管理器、from_url / from_config 工厂方法
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, override

import redis

if TYPE_CHECKING:
    from cs.config_manager import RedisConfig

# ============ 类型别名 (PEP 695) ============
type RedisValue = str | bytes
type ScanResult = tuple[int, list[str]]


@dataclass(slots=True, frozen=True)
class RedisOptions:
    """Redis 连接参数 (与 config.toml [redis] 字段对齐)"""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    decode_responses: bool = True
    max_connections: int = 50
    socket_connect_timeout: float = 2.0
    socket_timeout: float = 5.0
    health_check_interval: int = 30


def _build_pool(opts: RedisOptions) -> redis.ConnectionPool:
    kwargs: dict[str, Any] = {
        "host": opts.host,
        "port": opts.port,
        "db": opts.db,
        "decode_responses": opts.decode_responses,
        "max_connections": opts.max_connections,
        "socket_connect_timeout": opts.socket_connect_timeout,
        "socket_timeout": opts.socket_timeout,
        "socket_keepalive": True,
        "health_check_interval": opts.health_check_interval,
    }
    if opts.password:
        kwargs["password"] = opts.password
    return redis.ConnectionPool(**kwargs)


class Redis:
    """Redis 客户端封装：共享连接池，线程安全，可直接调用常用命令。"""

    __slots__ = ("_pool", "_client")

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        *,
        ip: str | None = None,
        password: str | None = None,
        decode_responses: bool = True,
        max_connections: int = 50,
        socket_connect_timeout: float = 2.0,
        socket_timeout: float = 5.0,
        health_check_interval: int = 30,
    ) -> None:
        opts = RedisOptions(
            host=ip or host,
            port=port,
            db=db,
            password=password or None,
            decode_responses=decode_responses,
            max_connections=max_connections,
            socket_connect_timeout=socket_connect_timeout,
            socket_timeout=socket_timeout,
            health_check_interval=health_check_interval,
        )
        self._pool = _build_pool(opts)
        self._client = redis.Redis(connection_pool=self._pool)

    @classmethod
    def from_options(cls, opts: RedisOptions) -> Self:
        obj = cls.__new__(cls)
        obj._pool = _build_pool(opts)
        obj._client = redis.Redis(connection_pool=obj._pool)
        return obj

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> Self:
        """从 redis:// URL 创建客户端，如 redis://:pass@localhost:6379/0"""
        obj = cls.__new__(cls)
        obj._pool = redis.ConnectionPool.from_url(
            url,
            decode_responses=kwargs.pop("decode_responses", True),
            max_connections=kwargs.pop("max_connections", 50),
            socket_keepalive=kwargs.pop("socket_keepalive", True),
            health_check_interval=kwargs.pop("health_check_interval", 30),
            **kwargs,
        )
        obj._client = redis.Redis(connection_pool=obj._pool)
        return obj

    @classmethod
    def from_config(cls, cfg: RedisConfig | RedisOptions) -> Self:
        """从 RedisConfig / RedisOptions 创建客户端"""
        if isinstance(cfg, RedisOptions):
            return cls.from_options(cfg)
        return cls(
            host=cfg.host,
            port=cfg.port,
            db=cfg.db,
            password=cfg.password or None,
            decode_responses=cfg.decode_responses,
        )

    # ---- 生命周期 ----

    @property
    def client(self) -> redis.Redis:
        """底层 redis.Redis 实例，用于高级命令"""
        return self._client

    @property
    def conn(self) -> redis.Redis:
        """兼容旧代码：conn 与 client 等价"""
        return self._client

    def ping(self) -> bool:
        return bool(self._client.ping())

    def close(self) -> None:
        self._client.close()
        self._pool.disconnect()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @override
    def __repr__(self) -> str:
        kw = self._pool.connection_kwargs
        return (
            f"Redis(host={kw.get('host')!r}, port={kw.get('port')}, db={kw.get('db')})"
        )

    # ---- String ----

    def set(
        self,
        key: str,
        value: RedisValue,
        ex: int | None = None,
        *,
        px: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """设置键值；ex 为过期秒数 (兼容旧版 set(key, value, sec) 三参数写法)"""
        return bool(self._client.set(key, value, ex=ex, px=px, nx=nx, xx=xx))

    def setex(self, key: str, seconds: int, value: RedisValue) -> bool:
        return bool(self._client.setex(key, seconds, value))

    def get(self, key: str, default: RedisValue | None = None) -> RedisValue | None:
        val = self._client.get(key)
        return default if val is None else val

    def mget(self, *keys: str) -> list[RedisValue | None]:
        return list(self._client.mget(keys))

    def mset(self, mapping: dict[str, RedisValue]) -> bool:
        return bool(self._client.mset(mapping))

    # ---- Key ----

    def exists(self, *keys: str) -> int:
        return int(self._client.exists(*keys))

    def delete(self, *keys: str) -> int:
        return int(self._client.delete(*keys))

    def expire(self, key: str, seconds: int) -> bool:
        return bool(self._client.expire(key, seconds))

    def ttl(self, key: str) -> int:
        return int(self._client.ttl(key))

    def scan(self, **kwargs: Any) -> ScanResult:
        """模糊匹配扫描键，返回 (cursor, keys)"""
        cursor, keys = self._client.scan(**kwargs)
        return int(cursor), list(keys)

    # ---- Hash ----

    def hset(
        self,
        name: str,
        key: str | None = None,
        value: RedisValue | None = None,
        *,
        mapping: dict[str, RedisValue] | None = None,
    ) -> int:
        if mapping is not None:
            return int(self._client.hset(name, mapping=mapping))
        if key is None or value is None:
            raise ValueError("hset 需要 (name, key, value) 或 mapping=...")
        return int(self._client.hset(name, key, value))

    def hget(self, name: str, key: str) -> RedisValue | None:
        return self._client.hget(name, key)

    def hgetall(self, name: str) -> dict[str, RedisValue]:
        return dict(self._client.hgetall(name))

    def hdel(self, name: str, *keys: str) -> int:
        return int(self._client.hdel(name, *keys))

    # ---- Counter ----

    def incr(self, key: str, amount: int = 1) -> int:
        return int(self._client.incrby(key, amount))

    def decr(self, key: str, amount: int = 1) -> int:
        return int(self._client.decrby(key, amount))

    # ---- List ----

    def brpop(
        self,
        key: str,
        timeout: float = 0,
    ) -> tuple[str, RedisValue] | None:
        """阻塞弹出列表尾部元素；timeout=0 表示无限等待"""
        result = self._client.brpop(key, timeout=timeout)
        if result is None:
            return None
        return result[0], result[1]

    def brpop_many(
        self,
        key: str,
        count: int,
        *,
        timeout: float = 0,
    ) -> list[RedisValue]:
        """连续 brpop count 次；队列空且超时则提前结束"""
        items: list[RedisValue] = []
        for _ in range(max(0, count)):
            result = self._client.brpop(key, timeout=timeout)
            if result is None:
                break
            items.append(result[1])
        return items

    get_multi_data = brpop_many  # 兼容旧方法名

    # ---- Pipeline ----

    @contextmanager
    def pipeline(self, transaction: bool = True):
        """
        批量命令管道 (退出时自动 execute)。

        用法:
            with r.pipeline() as pipe:
                pipe.set("a", 1)
                pipe.set("b", 2)
        """
        with self._client.pipeline(transaction=transaction) as pipe:
            yield pipe


__all__ = ["Redis", "RedisOptions"]

"""``cs.net`` 的测试。

只用回环地址与本地临时目录，不依赖外网：需要联网的用例（公网 IP）用假响应替身，
留一个由 ``CS_NET_LIVE=1`` 打开的真实网络测试。
"""

import asyncio
import json
import os
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs.net import (
    Address,
    AddressError,
    HttpError,
    HttpResponse,
    Net,
    PortError,
    PortInUseError,
    PortRange,
    PortScan,
    PortState,
    StaticServer,
    get_free_port,
    get_host_ip,
    get_wifi_name,
    is_open_port,
    resolve,
)
from cs.net import http as net_http
from cs.net import ports as net_ports
from cs.net.__main__ import main
from cs.net.models import DnsRecord, HostInfo, Interface, PortCheck, PublicIP

# ============================================================
# 夹具
# ============================================================


@pytest.fixture
def listener():
    """在回环地址上起一个真实监听端口，yield ``(host, port)``。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 不设 SO_REUSEADDR：Windows 上它允许两个进程绑同一端口，会让「占用」用例失效
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    try:
        yield "127.0.0.1", sock.getsockname()[1]
    finally:
        sock.close()


@pytest.fixture
def site(tmp_path: Path):
    """一个含 index.html 与纯文本文件的目录。"""
    (tmp_path / "index.html").write_text("<h1>hi</h1>", encoding="utf-8")
    (tmp_path / "hello.txt").write_text("hello cs.net", encoding="utf-8")
    return tmp_path


# ============================================================
# Address
# ============================================================


def test_address_parse_variants():
    assert Address.parse("127.0.0.1:8000").port == 8000
    assert Address.parse("127.0.0.1:8000").host == "127.0.0.1"
    assert Address.parse("example.com").port == 0
    assert Address.parse("example.com", default_port=443).port == 443
    assert Address.parse("[::1]:8080") == Address("::1", 8080)
    assert Address.parse("[::1]").port == 0
    assert Address.parse("::1").host == "::1"


def test_address_validation_and_str():
    assert str(Address("10.0.0.1", 80)) == "10.0.0.1:80"
    assert str(Address("::1", 80)) == "[::1]:80"
    assert Address("10.0.0.1", "80").port == 80  # 字符串端口会被转成 int
    assert Address("[::1]", 1) == Address("::1", 1)  # 方括号被剥掉

    with pytest.raises(AddressError):
        Address("", 80)
    with pytest.raises(AddressError):
        Address("10.0.0.1", 70000)
    with pytest.raises(AddressError):
        Address("10.0.0.1", "http")
    with pytest.raises(AddressError):
        Address.parse("")
    with pytest.raises(AddressError):
        Address.parse("[::1:80")


def test_address_properties():
    loop = Address("localhost", 80)
    assert loop.is_loopback
    assert Address("127.0.0.1", 80).is_loopback
    assert Address("::1", 80).is_loopback
    assert not Address("10.0.0.1", 80).is_loopback
    assert not Address("example.com", 80).is_loopback

    assert Address("10.0.0.1", 80).family == socket.AF_INET
    assert Address("::1", 80).family == socket.AF_INET6
    assert Address("10.0.0.1", 80).sockaddr == ("10.0.0.1", 80)
    assert Address("10.0.0.1", 80).with_port(81) == Address("10.0.0.1", 81)


def test_address_to_dict():
    payload = Address("10.0.0.1", 80).to_dict()
    assert payload == {"host": "10.0.0.1", "port": 80, "address": "10.0.0.1:80"}


# ============================================================
# PortRange / 结果模型
# ============================================================


def test_port_range_parse():
    assert PortRange.parse("8000").ports() == (8000,)
    assert PortRange.parse("8000-8002").ports() == (8000, 8001, 8002)
    assert PortRange.parse("8000:8002").ports() == (8000, 8001, 8002)
    with pytest.raises(AddressError):
        PortRange.parse("8002,8000")  # 闭区间表示不了离散端口
    assert len(PortRange(1, 10)) == 10
    assert 5 in PortRange(1, 10)
    assert 11 not in PortRange(1, 10)
    assert list(PortRange(1, 3)) == [1, 2, 3]
    assert str(PortRange(8000, 8000)) == "8000"
    assert PortRange(1, 3).to_dict() == {"start": 1, "end": 3, "count": 3}


def test_port_range_rejects_bad_values():
    with pytest.raises(AddressError):
        PortRange(0, 10)
    with pytest.raises(AddressError):
        PortRange(10, 1)
    with pytest.raises(AddressError):
        PortRange(1, 65536)
    with pytest.raises(AddressError):
        PortRange(1, 65535).ports(limit=10)
    with pytest.raises(AddressError):
        PortRange.parse("")


def test_port_check_and_scan_dict():
    check = PortCheck(Address("10.0.0.1", 80), PortState.OPEN, 1.234, "")
    assert check.is_open
    assert check.host == "10.0.0.1"
    assert check.port == 80
    assert check.to_dict()["state"] == "open"
    assert check.to_dict()["latency_ms"] == 1.23

    scan = PortScan(
        host="10.0.0.1",
        checks=(
            check,
            PortCheck(Address("10.0.0.1", 81), PortState.CLOSED, 0.5, "连接被拒绝"),
        ),
        elapsed_ms=3.0,
    )
    assert scan.open_ports == (80,)
    assert scan.scanned == 2
    assert scan.to_dict(only_open=True)["ports"] == [check.to_dict()]
    assert scan.to_dict()["scanned"] == 2


def test_data_models_to_dict():
    assert Interface(
        "eth0", ("10.0.0.1",), ("fe80::1",), "aa:bb", True, 1000, 1500
    ).to_dict()["ipv4"] == ["10.0.0.1"]
    assert Interface("eth0", ("10.0.0.1",)).addresses == ("10.0.0.1",)

    info = HostInfo("pc", "pc.local", "10.0.0.1", ("10.0.0.1",), ("::1",))
    assert info.ips == ("10.0.0.1", "::1")

    record = DnsRecord("example.com", ("1.1.1.1", "::1"))
    assert record.ipv4 == ("1.1.1.1",)
    assert record.ipv6 == ("::1",)
    assert record.to_dict()["addresses"] == ["1.1.1.1", "::1"]

    assert PublicIP("1.1.1.1", "test", 1.0).to_dict()["ip"] == "1.1.1.1"


# ============================================================
# HTTP 响应对象
# ============================================================


def test_http_response_helpers():
    body = json.dumps({"a": 1}).encode()
    response = HttpResponse(
        "https://x/", 200, {"Content-Type": "application/json"}, body, 12.0
    )
    assert response.ok
    assert response.size == len(body)
    assert response.json() == {"a": 1}
    assert response.content_type == "application/json"
    payload = response.to_dict()
    assert payload["status"] == 200 and payload["ok"] is True

    bad = HttpResponse("https://x/", 500, {}, b"<html>boom</html>")
    assert not bad.ok
    with pytest.raises(HttpError):
        bad.json()


def test_http_response_truncates_text():
    response = HttpResponse("https://x/", 200, {}, b"x" * 50)
    assert response.to_dict(limit=10)["truncated"] is True
    assert len(response.to_dict(limit=10)["text"]) == 10
    assert response.to_dict(full=True)["truncated"] is False


def test_extract_ip_handles_text_and_json():
    assert net_http._extract_ip("1.2.3.4\n") == "1.2.3.4"
    assert net_http._extract_ip('{"origin": "5.6.7.8, 9.9.9.9"}') == "5.6.7.8"
    assert net_http._extract_ip('{"ip": "::1"}') == "::1"
    assert net_http._extract_ip("not an ip") == ""


def test_public_ip_uses_first_working_provider(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "bad" in url:
            raise HttpError("boom", url=url)
        return HttpResponse(url, 200, {}, b'{"origin": "8.8.8.8"}')

    monkeypatch.setattr(net_http, "get", fake_get)
    result = net_http.public_ip(providers=("https://bad", "https://good"))
    assert result.ip == "8.8.8.8"
    assert result.provider == "https://good"
    assert calls == ["https://bad", "https://good"]


def test_public_ip_all_fail(monkeypatch):
    def fake_get(url, **kwargs):
        raise HttpError("boom", url=url)

    monkeypatch.setattr(net_http, "get", fake_get)
    with pytest.raises(HttpError):
        net_http.public_ip(providers=("https://a", "https://b"))


# ============================================================
# 本机信息
# ============================================================


def test_local_host_info():
    info = Net.host_info()
    assert info.hostname
    assert info.primary_ip
    for ip in info.ips:
        assert ip


def test_local_interfaces_sorted_up_first():
    items = Net.interfaces()
    assert items
    assert [item.is_up for item in items] == sorted(
        (item.is_up for item in items), reverse=True
    )
    assert any(item.is_up for item in items)


def test_local_addresses_excludes_loopback_by_default():
    ipv4, ipv6 = Net.local_addresses()
    assert all(not ip.startswith("127.") for ip in ipv4)
    assert all(ip != "::1" for ip in ipv6)
    with_loop = Net.local_addresses(include_loopback=True)[0]
    assert len(with_loop) >= len(ipv4)


def test_wifi_name_is_optional_string():
    name = Net.wifi()
    assert name is None or isinstance(name, str)


def test_dns_resolve_localhost():
    record = resolve("localhost")
    assert record.addresses
    assert "127.0.0.1" in record.addresses or "::1" in record.addresses
    with pytest.raises(AddressError):
        resolve("")


# ============================================================
# 端口
# ============================================================


def test_check_open_port(listener):
    host, port = listener
    result = Net.check(host, port, timeout=1.0)
    assert result.state is PortState.OPEN
    assert result.is_open
    assert result.latency_ms >= 0
    assert Net.is_open(host, port, timeout=1.0)


def test_check_accepts_host_port_string(listener):
    host, port = listener
    assert Net.check(f"{host}:{port}", timeout=1.0).is_open
    # 显式 port 优先于字符串里的端口
    assert not Net.check(f"{host}:{port}", 1, timeout=0.2).is_open


def test_check_closed_port_is_not_open():
    port = Net.free_port()
    result = Net.check("127.0.0.1", port, timeout=0.2)
    assert not result.is_open
    assert result.state in {PortState.CLOSED, PortState.FILTERED, PortState.ERROR}


def test_check_unresolvable_host_raises():
    with pytest.raises(AddressError):
        Net.check("no-such-host.invalid", 80, timeout=0.5)


def test_is_free(listener):
    host, port = listener
    assert not Net.is_free(port, host=host)
    free = Net.free_port(host=host)
    assert Net.is_free(free, host=host)


def test_free_port_bindable():
    port = Net.free_port()
    assert 0 < port <= 65535
    with socket.socket() as sock:
        sock.bind(("", port))


def test_free_port_with_candidates(monkeypatch):
    taken = Net.free_port()
    candidate = PortRange(taken, taken + 3)
    monkeypatch.setattr(
        net_ports, "is_free", lambda port, *, host="": port == taken + 2
    )
    assert Net.free_port(candidates=candidate) == taken + 2
    monkeypatch.setattr(net_ports, "is_free", lambda port, *, host="": False)
    with pytest.raises(PortError):
        Net.free_port(candidates=candidate)


def test_wait_open_success(listener):
    host, port = listener
    assert Net.wait_open(host, port, timeout=2.0).is_open


def test_wait_open_timeout_returns_last_check():
    port = Net.free_port()
    result = Net.wait_open("127.0.0.1", port, timeout=0.3, interval=0.1)
    assert not result.is_open


def test_wait_closed(listener):
    host, port = listener
    result = Net.wait_closed(host, port, timeout=0.5)  # 一直开着 → 超时返回
    assert result.is_open


def test_scan_finds_listener(listener):
    host, port = listener
    result = Net.scan(host, PortRange(port, port), timeout=1.0, workers=1)
    assert result.open_ports == (port,)
    assert result.scanned == 1


def test_scan_accepts_mixed_port_string(listener):
    host, port = listener
    result = Net.scan(host, f"{port},{port + 1}-{port + 2}", timeout=0.3, workers=3)
    assert result.scanned == 3
    assert port in result.open_ports


def test_scan_rejects_huge_range():
    with pytest.raises(AddressError):
        Net.scan("127.0.0.1", PortRange(1, 65535))


# ============================================================
# 静态服务 + HTTP
# ============================================================


def test_static_server_serves_files(site):
    with StaticServer(site) as server:
        assert server.running
        assert server.port > 0
        response = Net.get(f"{server.url}hello.txt")
        assert response.status == 200
        assert response.text() == "hello cs.net"
        assert Net.get(f"{server.url}nope.txt").status == 404
    assert not server.running
    assert Net.is_free(server.port, host="127.0.0.1")


def test_static_server_download(site):
    with StaticServer(site) as server:
        target = site / "nested" / "copy.txt"
        saved = Net.download(f"{server.url}hello.txt", target)
        assert saved == target
        assert saved.read_text(encoding="utf-8") == "hello cs.net"
        with pytest.raises(FileExistsError):
            Net.download(f"{server.url}hello.txt", target)


def test_static_server_rejects_bad_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        StaticServer(tmp_path / "nope")

    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        StaticServer(file_path)


def test_static_server_port_in_use(site, listener):
    _, port = listener
    with pytest.raises(PortInUseError):
        StaticServer(site, port=port).start()


def test_http_post_returns_5xx_without_raising(site):
    with StaticServer(site) as server:
        response = Net.post_json(f"{server.url}hello.txt", {"a": 1})
        assert response.status == 501
        assert not response.ok


def test_http_request_raises_on_connection_error():
    port = Net.free_port()
    with pytest.raises(HttpError):
        Net.get(f"http://127.0.0.1:{port}/x", timeout=0.3)


def test_http_request_raises_on_bad_url():
    with pytest.raises(HttpError):
        Net.get("nope")


# ============================================================
# 旧接口兼容
# ============================================================


def test_legacy_wrappers(listener):
    host, port = listener
    assert get_free_port() > 0
    assert get_host_ip()
    assert is_open_port(port, host)
    assert get_wifi_name() == Net.wifi()


# ============================================================
# MCP
# ============================================================


def test_mcp_tool_names():
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    from cs.net.mcp_server import create_server

    async def names() -> list[str]:
        async with Client(create_server()) as client:
            return [tool.name for tool in await client.list_tools()]

    assert asyncio.run(names()) == [
        "net_info",
        "net_interfaces",
        "net_wifi",
        "net_ip",
        "net_public_ip",
        "net_resolve",
        "net_reverse",
        "net_port_check",
        "net_port_free",
        "net_free_port",
        "net_scan",
        "net_wait_port",
        "net_http_get",
        "net_http_post",
        "net_static_start",
        "net_static_stop",
    ]


def test_mcp_port_and_static_tools(site):
    pytest.importorskip("fastmcp")
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    from cs.net.mcp_server import create_server

    async def flow() -> dict:
        async with Client(create_server()) as client:
            info = (await client.call_tool("net_info", {})).data
            assert info["ok"] and info["interfaces"]

            free = (
                await client.call_tool("net_free_port", {"port_range": "8000-8100"})
            ).data
            assert 8000 <= free["port"] <= 8100

            started = (
                await client.call_tool("net_static_start", {"directory": str(site)})
            ).data
            url = started["url"]
            fetched = (
                await client.call_tool("net_http_get", {"url": f"{url}hello.txt"})
            ).data
            assert fetched["status"] == 200
            assert fetched["text"] == "hello cs.net"

            checked = (
                await client.call_tool(
                    "net_port_check", {"target": f"127.0.0.1:{started['port']}"}
                )
            ).data
            assert checked["open"] is True

            stopped = (
                await client.call_tool("net_static_stop", {"port": started["port"]})
            ).data
            assert stopped["stopped"] is True

            with pytest.raises(ToolError):
                await client.call_tool("net_resolve", {"host": ""})
            return {"fetched": fetched}

    assert asyncio.run(flow())["fetched"]["ok"] is True


# ============================================================
# CLI
# ============================================================


def test_cli_info_and_ip(capsys):
    assert main(["info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hostname"]
    assert main(["ip"]) == 0
    assert capsys.readouterr().out.strip()


def test_cli_check_and_free(listener, capsys):
    host, port = listener
    assert main(["check", host, str(port)]) == 0
    assert json.loads(capsys.readouterr().out)["open"] is True
    assert main(["check", host, str(Net.free_port())]) == 1
    capsys.readouterr()

    assert main(["free", str(port), "--host", host]) == 1
    capsys.readouterr()
    assert main(["free-port"]) == 0
    assert 0 < int(capsys.readouterr().out.strip()) <= 65535


def test_cli_scan_and_resolve(listener, capsys):
    host, port = listener
    assert main(["scan", host, str(port), "--timeout", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["open"] == [port]
    assert main(["resolve", "localhost"]) == 0
    assert json.loads(capsys.readouterr().out)["addresses"]


def test_cli_http_get(site, capsys):
    with StaticServer(site) as server:
        assert main(["http", "get", f"{server.url}hello.txt"]) == 0
        assert json.loads(capsys.readouterr().out)["text"] == "hello cs.net"


def test_cli_error_exit_code(capsys):
    assert main(["free", "70000"]) == 2
    assert "AddressError" in capsys.readouterr().err


# ============================================================
# 真实网络（默认跳过）
# ============================================================


@pytest.mark.skipif(
    not os.environ.get("CS_NET_LIVE"), reason="需联网：设 CS_NET_LIVE=1 打开"
)
def test_public_ip_live():
    assert Net.public_ip(timeout=8).ip

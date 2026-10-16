"""End-to-end test for the Server's CONNECT (HTTPS) tunnelling path.

The rest of the suite never drives a real CONNECT request through the
server, which is how the original "HTTPS doesn't work" class of bugs
slipped through. This test wires up a tiny upstream proxy plus a tiny
TCP target, both on loopback, and asserts that a client can open a
CONNECT tunnel through :class:`proxybroker.server.Server` and exchange
bytes over it.
"""

import asyncio
import contextlib

import pytest

from proxybroker import Proxy
from proxybroker.server import Server


async def _pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()


class LoopbackServers:
    """Upstream CONNECT proxy + plain-TCP target, both on loopback."""

    def __init__(self):
        self.proxy_port = None
        self.target_port = None
        self._servers = []

    async def start(self):
        target = await asyncio.start_server(self._target, "127.0.0.1", 0)
        proxy = await asyncio.start_server(self._proxy, "127.0.0.1", 0)
        self._servers = [target, proxy]
        self.target_port = target.sockets[0].getsockname()[1]
        self.proxy_port = proxy.sockets[0].getsockname()[1]

    async def stop(self):
        for srv in self._servers:
            srv.close()
            await srv.wait_closed()

    @staticmethod
    async def _target(reader, writer):
        data = await reader.readline()
        writer.write(b"PONG:" + data)
        await writer.drain()
        writer.close()

    @staticmethod
    async def _proxy(reader, writer):
        request = await reader.readuntil(b"\r\n\r\n")
        line = request.split(b"\r\n", 1)[0].decode()
        method, authority = line.split()[:2]
        assert method == "CONNECT"
        host, _, port = authority.rpartition(":")
        up_reader, up_writer = await asyncio.open_connection(host, int(port))
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        await asyncio.gather(
            _pipe(reader, up_writer),
            _pipe(up_reader, writer),
        )


class TestConnectTunnel:
    @staticmethod
    async def _tunnel_once(server_port, target_port):
        """Open one CONNECT tunnel and exchange a line over it."""
        reader, writer = await asyncio.open_connection("127.0.0.1", server_port)
        try:
            writer.write(
                f"CONNECT 127.0.0.1:{target_port} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{target_port}\r\n\r\n".encode()
            )
            await writer.drain()

            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
            assert b"200" in head.split(b"\r\n", 1)[0]

            writer.write(b"ping\n")
            await writer.drain()
            echoed = await asyncio.wait_for(reader.readline(), timeout=5)
            assert echoed == b"PONG:ping\n"
        finally:
            writer.close()

    @pytest.mark.asyncio
    async def test_https_connect_is_tunnelled_through_pool(self, monkeypatch):
        """Repeated CONNECT tunnels work and never stop the server.

        Regression: only the first request succeeded. Afterwards the pool
        held a usable proxy but ``get()`` kept forcing imports from the
        drained source queue, raised NoProxyError and the server called
        ``self.stop()`` on itself.
        """
        upstream = LoopbackServers()
        await upstream.start()

        queue = asyncio.Queue()
        proxy = Proxy("127.0.0.1", upstream.proxy_port, types=("HTTP", "HTTPS"))
        proxy.types = {"HTTP": "High", "HTTPS": None}
        proxy.stat["requests"] = 10  # proven -> heap pool
        await queue.put(proxy)

        server = Server("127.0.0.1", 0, queue, timeout=5, import_timeout=0.3)
        stopped = []
        monkeypatch.setattr(server, "stop", lambda: stopped.append(True))

        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        try:
            # First tunnel drains the source queue into the pool; the
            # second must be served from the pool with the queue empty.
            for _ in range(2):
                await self._tunnel_once(port, upstream.target_port)

            assert stopped == []
            assert server._server is not None
        finally:
            await server.aclose()
            await upstream.stop()

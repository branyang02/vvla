import time

import websockets.sync.client
from loguru import logger

from vlla.serving import msgpack_numpy


class WebsocketPolicyClient:
    """Policy that proxies inference calls to a WebsocketPolicyServer."""

    def __init__(self, host: str = "localhost", port: int = 8765) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    @property
    def server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self):
        logger.info(f"Connecting to server at {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                logger.info("Connected.")
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Server not ready, retrying...")
                time.sleep(1)

    def infer(self, obs: dict) -> dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Server error:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        pass

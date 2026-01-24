"""
clients/libero_client/websocket_policy_client.py

Copy of src/vlla/serving modules as a standalone client file.
"""

import functools
import time

import msgpack
import numpy as np
import websockets.sync.client
from loguru import logger


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in (
        "V",
        "O",
        "c",
    ):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


class WebsocketPolicyClient:
    """Policy that proxies inference calls to a WebsocketPolicyServer."""

    def __init__(self, host: str = "localhost", port: int = 8765) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
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
                metadata = unpackb(conn.recv())
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
        return unpackb(response)

    def reset(self) -> None:
        pass

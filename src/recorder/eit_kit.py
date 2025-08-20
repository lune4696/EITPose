from typing import Optional, NamedTuple
import queue
import threading

import serial
import serial.tools.list_ports


class ThreadPair(NamedTuple):
    send: Optional[threading.Thread] = None
    receive: Optional[threading.Thread] = None


class SerialOrchestrator:
    def __init__(self):
        self.stop_events: dict[str, threading.Event] = {}
        self.handlers: dict[str, SerialHandler] = {}
        self.threads: dict[str, ThreadPair] = {}

    @staticmethod
    def print_port_list():
        for port in serial.tools.list_ports.comports():
            print(f"# {port=} / {port.device=}")

    def set_handler(
        self,
        handler_name: str,
        port: str,
        baudrate: int = 9600,
        parity: str = "N",
    ) -> None:
        if handler_name in self.handlers.keys():
            raise ValueError(f"SerialHandler '{handler_name}' is already set")

        self.stop_events[handler_name] = threading.Event()
        self.handlers[handler_name] = SerialHandler(
            port=port,
            baudrate=baudrate,
            parity=parity,
            stop_event=self.stop_events[handler_name],
        )
        self.threads[handler_name] = ThreadPair()

    def remove_handler(self, handler_name: str) -> None:
        self.handlers[handler_name].close()
        _ = self.stop_events.pop(handler_name)
        _ = self.handlers.pop(handler_name)

    def start(self, handler_name: str) -> None:
        self.threads[handler_name].send = threading.Thread(
            target=self.handlers[handler_name].send_loop,
        )
        self.threads[handler_name].receive = threading.Thread(
            target=self.handlers[handler_name].receive_loop,
        )

        self.threads[handler_name].send.start()
        self.threads[handler_name].receive.start()

    def stop(self, handler_name: str) -> None:
        self.stop_events[handler_name].set()

    def put_send_queue(self, handler_name: str, data: str) -> None:
        self.handlers[handler_name].send_queue.put(data)

    def get_receive_queue(self, handler_name: str) -> Optional[str]:
        if self.handlers[handler_name].receive_queue.empty():
            return None
        return self.handlers[handler_name].receive_queue.get_nowait()


class SerialHandler:
    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        parity: str = "N",
        stop_event: Optional[threading.Event] = None,
    ):
        # 'COM2' 9600bps Parityなしの場合
        self.timeout = 0.001
        self.serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            parity=parity,
            timeout=self.timeout,
        )
        self.send_queue = queue.Queue()
        self.receive_queue = queue.Queue()
        self.stop_event = stop_event

    def close(self):
        self.serial.close()

    def send_loop(self) -> None:
        """
        概要
            シリアル通信によるデータ送信(並列モード)を行う関数
            バイナリデータに変換するために encode() を用いる
        出力
            データ長: int
        例外
            serial.SerialException
        """
        if self.stop_event is None:
            raise ValueError("self.stop_event is not set!")

        while not self.stop_event.is_set():
            try:
                data = self.send_queue.get(timeout=self.timeout)
            except queue.Empty:
                continue
            self.serial.write(data.encode("utf-8"))

    def receive_loop(self, bytes: Optional[int] = None) -> None:
        """
        概要
            シリアル通信によるデータ受信(並列モード)を行う関数
            バイナリデータに変換するために encode() を用いる
        出力
            データ: str
        例外
            serial.SerialException
        """

        if self.stop_event is None:
            raise ValueError("self.stop_event is not set!")

        while not self.stop_event.is_set():
            data = self.serial.read(bytes) if bytes else self.serial.readline()
            if not data:  # タイムアウトで空ならスキップ
                continue
            data = data.strip()
            data = data.decode("utf-8")
            self.receive_queue.put(data, block=True)


def test_pyserial():
    pass


if __name__ == "__main__":
    test_pyserial()

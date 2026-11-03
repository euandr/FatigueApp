"""Detecção de fadiga isolada por conexão WebSocket.

Cada conexão tem seu próprio `Monitor`, com um detector MediaPipe e um
executor de uma única thread (o MediaPipe exige criação e uso na mesma
thread). Não há autenticação, sessão nem persistência: o backend apenas
recebe frames e devolve o resultado da análise.
"""
import asyncio
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor


class Monitor:
    def __init__(self):
        self.detector = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detector")
        self.closed = False

    def process(self, encoded):
        """Executa no executor da conexão (nunca no event loop)."""
        import cv2
        import numpy as np

        from .detector import Detector

        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError("Frame base64 inválido")
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Imagem inválida")
        if self.detector is None:
            self.detector = Detector()
        return self.detector.process_frame(frame)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.detector is not None:
                await asyncio.get_running_loop().run_in_executor(
                    self.executor, self.detector.close
                )
        finally:
            self.executor.shutdown(wait=False)

# -*- coding: utf-8 -*-
"""손그림 화면을 띄우는 loopback 전용 추론 서버."""

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading

from PIL import Image

from checkpoint import load_model
from images import recognize
from paths import WEB_PAGE


def build_server(model, port):
    """추론 HTTP 서버를 만든다. port=0 이면 빈 포트를 OS 가 고른다(테스트용).

    층이 forward 중간 상태를 self 에 저장하므로 모델은 **재진입 불가**다.
    ThreadingHTTPServer 는 요청마다 스레드를 만들기 때문에 추론 구간 전체를
    하나의 lock 으로 직렬화한다.
    """
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            data = WEB_PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid input size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("image"), str
                ):
                    raise ValueError("image must be a base64 string")
                raw = base64.b64decode(payload["image"].split(",")[-1], validate=True)
                with Image.open(io.BytesIO(raw)) as im:
                    with lock:
                        result = recognize(model, im)
                status = 200
            except (ValueError, KeyError, OSError) as error:
                result = {"error": str(error)}
                status = 400
            data = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(args):
    model, _ = load_model(args.model)
    server = build_server(model, args.port)
    print(f"http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

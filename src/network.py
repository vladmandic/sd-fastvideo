import time
import json
import cv2
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import HTTPException # pylint: disable=unused-import # noqa: F401
from fastapi import FastAPI # pylint: disable=unused-import # noqa: F401
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse # pylint: disable=unused-import # noqa: F401
import starlette.status as status
from starlette.websockets import WebSocket, WebSocketState, WebSocketClose, WebSocketDisconnect # pylint: disable=unused-import # noqa: F401
from logger import log
from options import options, stats


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []
        self.upload_ws = None
        # loop = asyncio.get_event_loop()
        # loop.run_in_executor(None, self.notify())

    async def connect(self, ws: WebSocket):
        await ws.accept()
        agent = ws._headers.get("user-agent", "") # pylint: disable=protected-access
        log.debug(f'ws connect: client={ws.client.host} agent="{agent}"')
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        log.debug(f'ws disconnect: {ws.client.host}')
        self.active.remove(ws)

    async def send(self, ws: WebSocket, data: str|dict|bytes):
        if ws.client_state != WebSocketState.CONNECTED:
            return
        if isinstance(data, np.ndarray):
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            _retval, buffer = cv2.imencode('.jpg', data)
            data = buffer.tobytes()
            await ws.send_bytes(data)
        elif isinstance(data, bytes):
            await ws.send_bytes(data)
        elif isinstance(data, dict):
            await ws.send_json(data)
        else:
            await ws.send_text(data)

    async def broadcast(self, data: str|dict|bytes):
        for ws in self.active:
            await self.send(ws, data)

    async def upload(self):
        import engine
        # if engine.upload_queue.qsize() > 0:
        while engine.upload_queue.qsize() > 0:
            image = engine.upload_queue.get()
            if self.upload_ws is not None:
                await self.send(self.upload_ws, image)
                del image
        # self.queue.task_done()


def mount(app: FastAPI):
    import engine
    app.mount("/page", StaticFiles(directory="page", html = True), name="page")
    app.mount("/assets", StaticFiles(directory="assets", html = True), name="assets")

    @app.get("/")
    async def main():
        return RedirectResponse(url="/page/index.html", status_code=status.HTTP_302_FOUND)

    @app.websocket("/ws/{clientid}")
    async def ws_controls(ws: WebSocket, clientid: int):
        await manager.connect(ws)
        await manager.send(ws, options.get())
        try:
            while True:
                data = await ws.receive_text()
                obj = json.loads(data)
                if 'ready' in obj:
                    ready = any(b.qsize() < options.batch for b in engine.buffers)
                    await manager.send(ws, { 'ready': ready })
                    continue
                if hasattr(options, obj['id']):
                    cast = type(getattr(options, obj['id']))
                    log.debug(f'ws: id={clientid} {obj} type={cast}')
                    try:
                        setattr(options, obj['id'], cast(obj['data']))
                        log.info(f'set: {obj["id"]}={getattr(options, obj["id"])}')
                        await manager.send(ws, { 'ok': True })
                    except Exception as e:
                        log.error(f'set: {obj["id"]}={obj["data"]} {e}')
                        await manager.send(ws, { 'ok': False, 'message': str(e) })
                else:
                    log.error(f'ws: id={clientid} {obj} unknown')
        except WebSocketDisconnect:
            manager.disconnect(ws)

    @app.websocket("/data/{clientid}")
    async def ws_data(ws: WebSocket, clientid: int): # pylint: disable=unused-argument
        await manager.connect(ws)
        manager.upload_ws = ws
        try:
            while True:
                data = await ws.receive_bytes()
                t0 = time.time()
                queue_free = engine.queue(data)
                await manager.upload()
                t1 = time.time()
                stats.network += t1 - t0
                if not queue_free:
                    time.sleep(0.05)
        except WebSocketDisconnect:
            manager.disconnect(ws)


manager = ConnectionManager()

import io
import json
import base64
from queue import Queue
import cv2
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import HTTPException # pylint: disable=unused-import # noqa: F401
from fastapi import FastAPI # pylint: disable=unused-import # noqa: F401
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse # pylint: disable=unused-import # noqa: F401
import starlette.status as status
from starlette.websockets import WebSocket, WebSocketState, WebSocketClose, WebSocketDisconnect # pylint: disable=unused-import # noqa: F401
from PIL import Image
import engine
from server import app
from logger import log
from options import options


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []
        self.queue = Queue()
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

    async def send(self, ws: WebSocket, data: str|dict|bytes|Image.Image):
        if ws.client_state != WebSocketState.CONNECTED:
            return
        if isinstance(data, np.ndarray):
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            _retval, buffer = cv2.imencode('.jpg', data)
            data = buffer.tobytes()
            await ws.send_bytes(data)
        if isinstance(data, Image.Image):
            img = io.BytesIO()
            data.save(img, format='JPEG')
            await ws.send_bytes(img.getvalue())
        elif isinstance(data, bytes):
            await ws.send_bytes(data)
        elif isinstance(data, dict):
            await ws.send_json(data)
        else:
            await ws.send_text(data)

    async def broadcast(self, data: str|dict|bytes|Image.Image):
        for ws in self.active:
            await self.send(ws, data)

    async def upload(self):
        if self.queue.qsize() == 0:
            return
        image = self.queue.get()
        if self.upload_ws is not None:
            await self.send(self.upload_ws, image)
        self.queue.task_done()


manager = ConnectionManager()
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
                await manager.send(ws, { 'ready': len(engine.b1) < options.batch or len(engine.b2) < options.batch })
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
                # await manager.broadcast({ id: clientid, data: data })
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
            data = data.decode('utf-8')
            data = base64.b64decode(data.split(",")[1])  # Remove the "data:image/png;base64," prefix
            try:
                image = Image.open(io.BytesIO(data), formats=["JPEG"])
            except Exception:
                image = Image.new('RGB', (options.width, options.height), (255, 255, 255))
            engine.queue(image)
            await manager.upload()
            # log.debug(f'ws: id={clientid} bytes={len(data)} image={image} queue={len(engine.b1)}/{len(engine.b2)}')
            # if not queued:
            #    time.sleep(0.5)
    except WebSocketDisconnect:
        manager.disconnect(ws)

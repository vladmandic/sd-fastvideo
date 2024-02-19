import ssl
import time
import logging
import threading
from asyncio.exceptions import CancelledError
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse
import uvicorn
from logger import log


fastapi_args = {
    "version": '0.0.0',
    "title": "SD.FastVideo",
    "description": "SD.FastVideo",
    "docs_url": None, # "/docs"
    "redoc_url": None, # "/redocs"
    "swagger_ui_parameters": {
        "displayOperationId": True,
        "showCommonExtensions": True,
        "deepLinking": False,
    }
}
httpd_args = {
    "listen": False,
    "port": 8000,
    "keyfile": None,
    "certfile": None,
    "loop": "uvloop", # auto, asyncio, uvloop
    "http": "auto", # auto, h11, httptools
}


def setup_middleware():
    log.debug('init middleware')
    ssl._create_default_https_context = ssl._create_unverified_context # pylint: disable=protected-access
    uvicorn_logger=logging.getLogger("uvicorn.error")
    uvicorn_logger.disabled = True
    app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']
    app.middleware_stack = None # reset current middleware to allow modifying user provided list
    app.add_middleware(GZipMiddleware, minimum_size=2048)

    @app.middleware("http")
    async def log_and_time(req: Request, call_next):
        try:
            ts = time.time()
            res: Response = await call_next(req)
            duration = str(round(time.time() - ts, 4))
            res.headers["X-Process-Time"] = duration
            endpoint = req.scope.get('path', 'err')
            if endpoint == '/nvml':
                return await call_next(req)
            token = req.cookies.get("access-token") or req.cookies.get("access-token-unsecure")
            log.info('http user={user} code={code} {prot}/{ver} {method} {endpoint} {cli} {duration}'.format( # pylint: disable=consider-using-f-string
                user = app.tokens.get(token) if hasattr(app, 'tokens') else None,
                code = res.status_code,
                ver = req.scope.get('http_version', '0.0'),
                cli = req.scope.get('client', ('0:0.0.0', 0))[0],
                prot = req.scope.get('scheme', 'err'),
                method = req.scope.get('method', 'err'),
                endpoint = endpoint,
                duration = duration,
            ))
            return res
        except CancelledError:
            log.warning('WebSocket closed (ignore asyncio.exceptions.CancelledError)')
        except BaseException as e:
            return handle_exception(req, e)

    def handle_exception(req: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "code": vars(e).get('status_code', 500),
            "detail": vars(e).get('detail', ''),
            "body": vars(e).get('body', ''),
            "errors": str(e),
        }
        log.error(f"http: {req.method}: {req.url} {err}")
        if err['code'] == 404 or err['code'] == 401:
            pass
        else:
            log.debug(e, exc_info=True) # print stack trace
        return JSONResponse(status_code=err['code'], content=jsonable_encoder(err))

    @app.exception_handler(HTTPException)
    async def http_exception_handler(req: Request, e: HTTPException):
        return handle_exception(req, e)

    @app.exception_handler(Exception)
    async def general_exception_handler(req: Request, e: Exception):
        if isinstance(e, TypeError):
            return JSONResponse(status_code=500, content=jsonable_encoder(str(e)))
        else:
            return handle_exception(req, e)

    @app.get('/nvml')
    async def nvml():
        import nvml
        return nvml.get()

    app.build_middleware_stack() # rebuild middleware stack on-the-fly


class UvicornServer(uvicorn.Server):
    def __init__(self, listen = None, port = None, keyfile = None, certfile = None, loop = "auto", http = "auto"):
        self.thread: threading.Thread = None
        self.wants_restart = False
        self.config = uvicorn.Config(
            app = app,
            host = "0.0.0.0" if listen else "127.0.0.1",
            port = port or 7861,
            loop = loop, # auto, asyncio, uvloop
            http = http, # auto, h11, httptools
            interface = "auto", # auto, asgi3, asgi2, wsgi
            ws = "auto", # auto, websockets, wsproto
            log_level = logging.WARNING,
            backlog = 4096, # default=2048
            timeout_keep_alive = 60, # default=5
            ssl_keyfile = keyfile,
            ssl_certfile = certfile,
            ws_max_size = 1024 * 1024 * 1024,  # default 16MB
        )
        super().__init__(config=self.config)
        log.debug(f'uvicorn args: {vars(self.config)}')

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.wants_restart = False
        log.info(f'uvicorn start: http://{self.config.host}:{self.config.port}')
        self.thread.start()

    def stop(self):
        self.should_exit = True
        log.info('uvicorn stop')
        self.thread.join()

    def restart(self):
        self.wants_restart = True
        self.stop()
        self.start()


server: UvicornServer = None
app: FastAPI = None


def httpd():
    global app, server # pylint: disable=global-statement
    app = FastAPI(**fastapi_args)
    log.debug(f'fastapi args: {vars(app)}')
    setup_middleware()
    server = UvicornServer(**httpd_args)
    server.start()
    from network import manager # pylint: disable=unused-import # noqa: F401
    return server, app, manager

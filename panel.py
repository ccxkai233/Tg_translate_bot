"""
双向私聊转发的 Web 面板 (aiohttp, 跑在机器人同一个事件循环里)

这个文件只管 HTTP: 登录、路由、安全头。数据和发消息都通过 bot.py 传进来的 api 对象完成,
所以这里不 import bot (bot.py 是以 __main__ 运行的, 反过来 import 会得到第二份状态)。

登录: 机器人在 Telegram 里发给主人的链接带一个 24 小时有效的签名 (?k=...),
点开后换成 30 天的 HttpOnly Cookie 并跳转到干净的地址。没有密码, 没有服务端会话表;
换掉 PANEL_SECRET 就能让所有链接和 Cookie 立刻失效。
"""

import hashlib
import hmac
import logging
import re
import time
from pathlib import Path

from aiohttp import web

log = logging.getLogger("tgtranslate.panel")

LINK_TTL = 86400            # Telegram 里的登录链接有效期
SESSION_TTL = 30 * 86400    # 浏览器 Cookie 有效期
COOKIE = "tgp_session"
MAX_UPLOAD = 50 * 1024 * 1024  # Bot API 上传上限

HTML_PATH = Path(__file__).with_name("panel.html")

# 只有这些类型允许在面板的源下直接显示。访客发来的 html/svg 如果内联打开,
# 里面的脚本就跑在面板的源上了, 所以其他一律强制下载
INLINE_MIME = re.compile(r"^(image/(jpeg|png|gif|webp)|audio/[\w.+-]+|video/[\w.+-]+)$")

EXPIRED_PAGE = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>需要登录</title>
<body style="font:16px/1.6 system-ui;display:grid;place-items:center;height:90vh;margin:0;padding:24px;text-align:center">
<div><h2>🔒 链接已过期或无效</h2><p>请在 Telegram 里给机器人发 <b>/panel</b> 获取新链接。</p></div>"""


def _sign(secret: str, kind: str, exp: int) -> str:
    return hmac.new(secret.encode(), f"{kind}.{exp}".encode(), hashlib.sha256).hexdigest()[:40]


def make_code(secret: str, kind: str, ttl: int) -> str:
    exp = int(time.time()) + ttl
    return f"{exp}.{_sign(secret, kind, exp)}"


def check_code(secret: str, kind: str, code: str) -> bool:
    try:
        exp_s, sig = code.split(".", 1)
        exp = int(exp_s)
    except (ValueError, AttributeError):
        return False
    return exp > time.time() and hmac.compare_digest(sig, _sign(secret, kind, exp))


def login_link(base_url: str, secret: str, user_id: int | None = None) -> str:
    url = f"{base_url}/?k={make_code(secret, 'link', LINK_TTL)}"
    return f"{url}#u={user_id}" if user_id else url


def create_app(api, secret: str, secure_cookie: bool) -> web.Application:
    @web.middleware
    async def security(request: web.Request, handler):
        is_api = request.path.startswith("/api/")
        authed = check_code(secret, "session", request.cookies.get(COOKIE, ""))

        if not is_api and "k" in request.query:
            if not check_code(secret, "link", request.query["k"]):
                return web.Response(text=EXPIRED_PAGE, content_type="text/html", status=401)
            # 相对跳转: 浏览器会保留 #u=... , 签名也不会留在地址栏里
            resp = web.HTTPFound("./")
            resp.set_cookie(
                COOKIE, make_code(secret, "session", SESSION_TTL),
                max_age=SESSION_TTL, httponly=True, secure=secure_cookie, samesite="Lax",
            )
            return resp

        if not authed:
            if is_api:
                return web.json_response({"error": "unauthorized"}, status=401)
            return web.Response(text=EXPIRED_PAGE, content_type="text/html", status=401)
        # 写操作必须带自定义头: 跨站表单带不了, 多一层 CSRF 保险 (Cookie 本身已是 SameSite=Lax)
        if request.method != "GET" and request.headers.get("X-Panel") != "1":
            return web.json_response({"error": "bad request"}, status=400)

        resp = await handler(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        if is_api and "Cache-Control" not in resp.headers:
            resp.headers["Cache-Control"] = "no-store"
        return resp

    def uid_of(request: web.Request) -> int:
        try:
            return int(request.match_info["uid"])
        except ValueError:
            raise web.HTTPNotFound()

    def int_q(request: web.Request, name: str, default: int) -> int:
        try:
            return int(request.query.get(name, default))
        except ValueError:
            return default

    async def index(request):
        return web.Response(
            text=HTML_PATH.read_text(encoding="utf-8"),
            content_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
                    "style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
                ),
            },
        )

    async def meta(request):
        return web.json_response(api.meta())

    async def poll(request):
        after = int_q(request, "after", -1)
        return web.json_response({
            "chats": api.chats(),
            "msgs": api.msgs_after(after) if after >= 0 else [],
            "max_id": api.max_id(),
        })

    async def messages(request):
        return web.json_response({
            "msgs": api.msgs(uid_of(request), before=int_q(request, "before", 0), limit=100)
        })

    async def read(request):
        api.mark_read(uid_of(request))
        return web.json_response({"ok": True})

    async def set_lang(request):
        body = await request.json()
        if not api.set_lang(uid_of(request), str(body.get("lang", ""))):
            return web.json_response({"error": "不支持的语言"}, status=400)
        return web.json_response({"ok": True})

    async def set_block(request):
        body = await request.json()
        api.set_block(uid_of(request), bool(body.get("blocked")))
        return web.json_response({"ok": True})

    async def send(request):
        uid, text, file = uid_of(request), "", None
        if request.content_type.startswith("multipart/"):
            async for part in await request.multipart():
                if part.name == "text":
                    text = (await part.text()).strip()
                elif part.name == "file" and part.filename:
                    data = await part.read(decode=False)
                    file = (bytes(data), part.filename, part.headers.get("Content-Type", ""))
        else:
            text = str((await request.json()).get("text", "")).strip()
        if not text and file is None:
            return web.json_response({"error": "空消息"}, status=400)
        result = await api.send(uid, text, file)
        return web.json_response(result, status=400 if "error" in result else 200)

    async def media(request):
        try:
            found = await api.media(int(request.match_info["mid"]))
        except ValueError:
            found = None
        if found is None:
            raise web.HTTPNotFound()
        path, name, mime = found
        inline = bool(INLINE_MIME.match(mime or ""))
        safe = re.sub(r'[^\w.\- ]', "_", name or "file")
        return web.FileResponse(path, headers={
            "Content-Type": mime if inline else "application/octet-stream",
            "Content-Disposition": f'{"inline" if inline else "attachment"}; filename="{safe}"',
            "Content-Security-Policy": "sandbox; default-src 'none'",
            "Cache-Control": "private, max-age=604800",
        })

    app = web.Application(middlewares=[security], client_max_size=MAX_UPLOAD + 1024 * 1024)
    app.add_routes([
        web.get("/", index),
        web.get("/api/meta", meta),
        web.get("/api/poll", poll),
        web.get("/api/chats/{uid}/messages", messages),
        web.post("/api/chats/{uid}/read", read),
        web.post("/api/chats/{uid}/lang", set_lang),
        web.post("/api/chats/{uid}/block", set_block),
        web.post("/api/chats/{uid}/send", send),
        web.get("/api/media/{mid}", media),
    ])
    return app


async def start(api, secret: str, bind: str, port: int, secure_cookie: bool) -> web.AppRunner:
    runner = web.AppRunner(create_app(api, secret, secure_cookie), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, bind, port).start()
    log.info("panel listening on %s:%d", bind, port)
    return runner

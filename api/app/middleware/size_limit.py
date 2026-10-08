"""Hard request-body size limit to reject oversized/chunked payloads before parsing JSON."""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _RequestBodyTooLarge(Exception):
    pass


class RequestSizeLimitMiddleware:
    """Limit HTTP request bodies without buffering them in application memory."""

    def __init__(self, app: ASGIApp, max_body_bytes: int = 1_000_000) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_body_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                # Let the ASGI server/framework report malformed Content-Length.
                pass

        received_bytes = 0
        body_too_large = False
        response_started = False
        rejection_body_sent = False
        rejection = JSONResponse(
            status_code=413,
            content={
                "detail": f"Размер тела запроса не должен превышать {self.max_body_bytes} байт"
            },
        )

        async def receive_limited() -> Message:
            nonlocal received_bytes, body_too_large
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    body_too_large = True
                    raise _RequestBodyTooLarge
            return message

        async def send_limited(message: Message) -> None:
            nonlocal response_started, rejection_body_sent
            if not body_too_large:
                if message["type"] == "http.response.start":
                    response_started = True
                await send(message)
                return

            # Starlette may translate a receive exception to HTTP 400 while parsing
            # JSON. Replace that response with the semantically correct HTTP 413.
            if message["type"] == "http.response.start":
                response_started = True
                await send(
                    {
                        **message,
                        "status": 413,
                        "headers": rejection.raw_headers,
                    }
                )
            elif message["type"] == "http.response.body" and not rejection_body_sent:
                rejection_body_sent = True
                await send(
                    {
                        "type": "http.response.body",
                        "body": rejection.body,
                        "more_body": False,
                    }
                )

        try:
            await self.app(scope, receive_limited, send_limited)
        except _RequestBodyTooLarge:
            if not response_started:
                await self._reject(scope, receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=413,
            content={
                "detail": f"Размер тела запроса не должен превышать {self.max_body_bytes} байт"
            },
        )
        await response(scope, receive, send)

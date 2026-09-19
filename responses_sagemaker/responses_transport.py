"""An httpx transport that lets the stock OpenAI client call a Responses API on SageMaker.

A SageMaker endpoint forwards one route to the model, ``POST /invocations``, and
authenticates with AWS SigV4 rather than a bearer token. For a container that
serves the OpenAI Responses API on that route, the only client-side work
is at the HTTP layer: rewrite ``.../responses`` to ``.../invocations`` and sign
the request. See ../README.md for building the client.

Streaming (``stream=True``) is not supported: SageMaker streams through
``InvokeEndpointWithResponseStream``, which wraps the SSE body in AWS event-stream
framing the OpenAI SDK cannot read.
"""

import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import BotoCoreError

_RESPONSES = "/responses"
_INVOCATIONS = "/invocations"


class SageMakerResponsesTransport(httpx.BaseTransport):
    """Rewrites ``POST .../responses`` to ``.../invocations``; SigV4-signs it when
    given credentials. Any other route would be a 404 at SageMaker's edge, so it
    is answered here with a 404 that says why (the SDK raises ``NotFoundError``;
    an exception from a transport would be retried and reported as a connection
    error)."""

    def __init__(self, *, credentials=None, region: str | None = None, inner: httpx.BaseTransport | None = None):
        self.credentials = credentials
        self.region = region
        self.inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method != "POST" or not path.endswith(_RESPONSES):
            message = f"{request.method} {path}: a SageMaker endpoint only serves responses.create (POST /invocations)"
            return httpx.Response(
                404, json={"error": {"message": message, "type": "invalid_request_error"}}, request=request
            )
        url = request.url.copy_with(path=path[: -len(_RESPONSES)] + _INVOCATIONS)
        # The OpenAI SDK always sends a bearer token; SageMaker strips it anyway,
        # and it must not be part of the signature.
        headers = {k: v for k, v in request.headers.items() if k.lower() != "authorization"}
        content = request.read()
        if self.credentials is not None:
            try:
                headers = self._sign(url, headers, content)
            except BotoCoreError as error:
                # Expired or missing AWS credentials. Raised from here, the SDK
                # would retry it and report a "connection error"; a 401 says what
                # actually happened.
                message = f"Could not sign the request with your AWS credentials: {error}"
                return httpx.Response(
                    401, json={"error": {"message": message, "type": "authentication_error"}}, request=request
                )
        signed = httpx.Request("POST", url, headers=headers, content=content, extensions=request.extensions)
        return _unwrap_model_error(self.inner.handle_request(signed), signed)

    def _sign(self, url: httpx.URL, headers: dict, content: bytes) -> dict:
        aws_request = AWSRequest(method="POST", url=str(url), data=content, headers=headers)
        SigV4Auth(self.credentials.get_frozen_credentials(), "sagemaker", self.region).add_auth(aws_request)
        return dict(aws_request.headers.items())

    def close(self) -> None:
        self.inner.close()


def _unwrap_model_error(response: httpx.Response, request: httpx.Request) -> httpx.Response:
    """SageMaker reports a container's 4xx/5xx as its own 424 ``ModelError``,
    carrying the original status and body. Hand the SDK the original, so a
    rejected request raises ``BadRequestError`` with the server's message rather
    than an opaque 424."""
    if response.status_code != 424:
        return response
    response.read()
    try:
        error = response.json()
        status, body = int(error["OriginalStatusCode"]), error["OriginalMessage"]
    except (ValueError, KeyError, TypeError):
        return response
    return httpx.Response(
        status, content=body.encode(), headers={"content-type": "application/json"}, request=request
    )

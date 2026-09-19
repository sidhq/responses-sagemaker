"""SageMakerResponsesTransport: responses.create reaches /invocations, signed; nothing else is sent."""

import json

import httpx
import openai
import pytest
from botocore.credentials import Credentials
from botocore.exceptions import LoginRefreshRequired
from openai import OpenAI

from responses_sagemaker import SageMakerResponsesTransport

RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "model": "my-model",
    "status": "completed",
    "output": [],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


def client(transport_kwargs=None):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=RESPONSE)

    transport = SageMakerResponsesTransport(inner=httpx.MockTransport(handler), **(transport_kwargs or {}))
    base = "https://runtime.sagemaker.us-east-2.amazonaws.com/endpoints/my-endpoint"
    return OpenAI(base_url=base, api_key="unused", http_client=httpx.Client(transport=transport), max_retries=0), seen


def test_responses_create_is_posted_to_invocations():
    c, seen = client()
    c.responses.create(model="my-model", input="hi", store=False)
    (request,) = seen
    assert request.method == "POST"
    assert request.url.path == "/endpoints/my-endpoint/invocations"
    assert json.loads(request.content)["input"] == "hi"
    assert "authorization" not in request.headers


def test_other_routes_fail_without_being_sent():
    c, seen = client()
    with pytest.raises(openai.NotFoundError, match="only serves responses.create"):
        c.models.list()
    assert seen == []


def test_a_container_error_surfaces_with_its_own_status():
    message = json.dumps({"error": {"message": "needs a task body", "type": "invalid_request_error"}})

    def handler(request):
        return httpx.Response(
            424, json={"ErrorCode": "CLIENT_ERROR_FROM_MODEL", "OriginalStatusCode": 400, "OriginalMessage": message}
        )

    transport = SageMakerResponsesTransport(inner=httpx.MockTransport(handler))
    c = OpenAI(base_url="https://x/endpoints/e", api_key="unused", http_client=httpx.Client(transport=transport))
    with pytest.raises(openai.BadRequestError, match="needs a task body"):
        c.responses.create(model="m", input="hi")


def test_expired_credentials_surface_as_401_not_a_connection_error():
    """botocore raises while signing when an `aws login` session lapses. Raised
    from the transport, the SDK would retry it and call it a connection error."""
    class Expired:
        def get_frozen_credentials(self):
            raise LoginRefreshRequired()

    c, seen = client({"credentials": Expired(), "region": "us-east-2"})
    with pytest.raises(openai.AuthenticationError, match="AWS credentials"):
        c.responses.create(model="my-model", input="hi")
    assert seen == [], "nothing unsigned was sent"


def test_requests_are_sigv4_signed():
    credentials = Credentials("AKIDEXAMPLE", "secret")
    c, seen = client({"credentials": credentials, "region": "us-east-2"})
    c.responses.create(model="my-model", input="hi")
    (request,) = seen
    assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
    assert "/us-east-2/sagemaker/aws4_request" in request.headers["authorization"]
    assert "x-amz-date" in request.headers

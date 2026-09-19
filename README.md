# responses-sagemaker

Call an Amazon SageMaker endpoint that serves the OpenAI Responses API, with the
stock OpenAI Python client. Any container that answers Responses API requests on
`/invocations` works.

A SageMaker endpoint differs from an OpenAI-compatible server in two ways: it
forwards a single route, `POST /invocations`, and it authenticates callers with
AWS SigV4 request signing instead of a bearer token. `SageMakerResponsesTransport`
is an httpx transport that handles both at the HTTP layer, so the rest of your
code — `client.responses.create(...)` — is unchanged.

## Install

```bash
pip install .   # from a clone of this repository; pulls in httpx and boto3[crt]
```

`boto3[crt]` rather than plain `boto3`, because credentials from `aws login`
profiles need the CRT extension.

## Build the client

Use this wherever you construct your OpenAI client; nothing else in your code
changes.

```python
import boto3
import httpx
from openai import OpenAI
from responses_sagemaker import SageMakerResponsesTransport

session = boto3.Session()  # credentials and region, as boto3 resolves them
client = OpenAI(
    base_url=f"https://runtime.sagemaker.{session.region_name}.amazonaws.com/endpoints/<endpoint-name>",
    api_key="unused",  # SageMaker authenticates with SigV4; the SDK insists on a key
    http_client=httpx.Client(
        transport=SageMakerResponsesTransport(credentials=session.get_credentials(), region=session.region_name)
    ),
)
```

Set the region with `AWS_DEFAULT_REGION` or your profile. **boto3 ignores
`AWS_REGION`**, so setting only that signs requests for the wrong region. Your
credentials need `sagemaker:InvokeEndpoint` on the endpoint.

## What the transport does

| | |
|---|---|
| `responses.create` | posted to `…/invocations` and SigV4-signed |
| any other call (`models.list`, `responses.retrieve`, …) | answered locally with a 404 naming the one supported call — SageMaker would 404 it anyway |
| a request the model rejects | SageMaker wraps it as a 424 `ModelError`; the transport restores the original status and body, so you get e.g. `BadRequestError` with the server's message |
| expired or missing AWS credentials | a 401 (`AuthenticationError`) saying so, rather than a retried "connection error" |

Without credentials the transport rewrites paths but does not sign — useful for
a container running locally: `base_url="http://127.0.0.1:8080"`.

## Limits

These come from SageMaker real-time inference, not from the transport:

* **No streaming** through the OpenAI SDK. SageMaker streams via
  `InvokeEndpointWithResponseStream`, which wraps SSE in AWS event-stream framing.
* **6 MB** request and response bodies and **60 s** per invocation. Every turn
  resends the whole conversation, so keep history under the cap.
* `x-session-affinity` has no effect; SageMaker strips custom headers.

## Tests

```bash
uv run --group dev -m pytest tests -q
```

"""Call an Amazon SageMaker endpoint that serves the OpenAI Responses API, with the stock OpenAI client."""

from .responses_transport import SageMakerResponsesTransport

__all__ = ["SageMakerResponsesTransport"]

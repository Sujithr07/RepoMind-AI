import os
from langfuse import Langfuse

_langfuse_client = None

def get_langfuse() -> Langfuse:
    """
    Returns a singleton Langfuse client instance.
    Initializes the client on first call using environment variables.
    """
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
        )
    return _langfuse_client

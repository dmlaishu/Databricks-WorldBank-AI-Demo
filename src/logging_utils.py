import logging
import time
import uuid
from contextlib import contextmanager

logger = logging.getLogger("world_bank_ai")
logging.basicConfig(level=logging.INFO)

@contextmanager
def traced_request(question: str, route: str):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    logger.info("request_started id=%s route=%s question=%r", request_id, route, question)
    try:
        yield request_id
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info("request_completed id=%s route=%s latency_ms=%.2f", request_id, route, latency_ms)
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.exception("request_failed id=%s route=%s latency_ms=%.2f", request_id, route, latency_ms)
        raise

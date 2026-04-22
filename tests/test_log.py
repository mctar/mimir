import pytest


def test_logger_importable():
    from log import logger
    assert logger is not None


def test_logger_emits_debug_message():
    from log import logger
    import io
    buf = io.StringIO()
    handler_id = logger.add(buf, level="DEBUG", format="{message}")
    logger.debug("mimir-test-debug-message")
    logger.remove(handler_id)
    assert "mimir-test-debug-message" in buf.getvalue()


def test_logger_emits_info_message():
    from log import logger
    import io
    buf = io.StringIO()
    handler_id = logger.add(buf, level="DEBUG", format="{message}")
    logger.info("mimir-test-info-message")
    logger.remove(handler_id)
    assert "mimir-test-info-message" in buf.getvalue()


def test_logger_emits_error_message():
    from log import logger
    import io
    buf = io.StringIO()
    handler_id = logger.add(buf, level="DEBUG", format="{message}")
    logger.error("mimir-test-error-message")
    logger.remove(handler_id)
    assert "mimir-test-error-message" in buf.getvalue()

import importlib
import logging
from pathlib import Path


def _reload_logging_setup_module():
    module = importlib.import_module('app.logging_setup')
    return importlib.reload(module)


def test_configure_logging_creates_rotating_log_files(tmp_path, monkeypatch):
    logging_setup = _reload_logging_setup_module()

    monkeypatch.setenv('LOG_DIR', str(tmp_path))
    monkeypatch.setenv('LOG_FORMAT', 'text')
    monkeypatch.setenv('APP_LOG_FILE', 'application.log')
    monkeypatch.setenv('ERROR_LOG_FILE', 'application.error.log')

    root = logging.getLogger()
    original_handlers = list(root.handlers)
    if hasattr(root, '_app_logging_configured'):
        delattr(root, '_app_logging_configured')

    try:
        logging_setup.configure_logging()
        logger = logging.getLogger('test.logging')
        logger.info('hello')
        logger.error('boom')

        for handler in root.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()

        app_log_path = tmp_path / 'application.log'
        error_log_path = tmp_path / 'application.error.log'

        assert app_log_path.exists()
        assert error_log_path.exists()

        app_log_content = app_log_path.read_text(encoding='utf-8')
        assert 'test_logging_setup.py' in app_log_content
        assert 'test_configure_logging_creates_rotating_log_files' in app_log_content
    finally:
        for handler in root.handlers:
            try:
                handler.close()
            except Exception:
                pass
        root.handlers = original_handlers
        if hasattr(root, '_app_logging_configured'):
            delattr(root, '_app_logging_configured')


def test_json_formatter_contains_code_location_fields():
    logging_setup = _reload_logging_setup_module()
    formatter = logging_setup.JsonLogFormatter()

    record = logging.LogRecord(
        name='demo',
        level=logging.INFO,
        pathname='/tmp/demo.py',
        lineno=42,
        msg='ok',
        args=(),
        exc_info=None,
        func='handler',
    )
    output = formatter.format(record)

    assert '"file": "/tmp/demo.py"' in output
    assert '"line": 42' in output
    assert '"function": "handler"' in output

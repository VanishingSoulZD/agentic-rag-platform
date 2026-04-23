import contextvars
import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("-")
        record.service = os.getenv("APP_NAME", "agentic-rag-platform")
        return True


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": getattr(
                record, "service", os.getenv("APP_NAME", "agentic-rag-platform")
            ),
            "request_id": getattr(record, "request_id", "-"),
            "file": record.pathname,
            "filename": record.filename,
            "line": record.lineno,
            "function": record.funcName,
            "process": record.process,
            "thread": record.thread,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class TextLogFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)s | %(service)s | %(request_id)s | "
            "%(name)s | %(filename)s:%(lineno)d %(funcName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def _build_rotating_file_handler(
    file_path: Path, level: int, formatter: logging.Formatter
) -> RotatingFileHandler:
    max_bytes = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(RequestContextFilter())
    return file_handler


def configure_logging() -> None:
    root_logger = logging.getLogger()
    if getattr(root_logger, "_app_logging_configured", False):
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = os.getenv("LOG_FORMAT", "text").lower()
    formatter: logging.Formatter = (
        JsonLogFormatter() if log_format == "json" else TextLogFormatter()
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestContextFilter())

    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    app_log = log_dir / os.getenv("APP_LOG_FILE", "app.log")
    error_log = log_dir / os.getenv("ERROR_LOG_FILE", "error.log")

    app_file_handler = _build_rotating_file_handler(app_log, level, formatter)
    error_file_handler = _build_rotating_file_handler(
        error_log, logging.ERROR, formatter
    )

    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(app_file_handler)
    root_logger.addHandler(error_file_handler)
    root_logger.setLevel(level)

    root_logger._app_logging_configured = True


def set_request_id(request_id: str) -> contextvars.Token:
    return request_id_ctx.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    request_id_ctx.reset(token)

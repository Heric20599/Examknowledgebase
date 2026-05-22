from contextlib import asynccontextmanager
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.deps import build_openai_client, build_pinecone_client, ensure_pinecone_index
from app.errors import AppError
from app.routers.admin import router as admin_router
from app.routers.catalog import router as catalog_router
from app.routers.cron import router as cron_router
from app.routers.exam import router as exam_router
from app.routers.ingest import router as ingest_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_format = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    daily_log_path = logs_dir / "app.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))

    daily_file_handler = TimedRotatingFileHandler(
        filename=daily_log_path,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    daily_file_handler.suffix = "%Y-%m-%d"
    daily_file_handler.setFormatter(logging.Formatter(log_format))

    root_logger.addHandler(console_handler)
    root_logger.addHandler(daily_file_handler)

    settings = get_settings()
    app.state.settings = settings
    app.state.openai = build_openai_client(settings)
    app.state.pinecone = build_pinecone_client(settings)
    ensure_pinecone_index(settings, app.state.pinecone)
    yield


app = FastAPI(title="School Knowledge Base RAG API", lifespan=lifespan)
logger = logging.getLogger(__name__)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": detail,
            "details": {"detail": exc.detail},
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Invalid request payload. Please check field names and values.",
            "details": {"errors": exc.errors()},
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Something went wrong on the server. Please try again.",
            "details": {},
        },
    )


app.include_router(ingest_router)
app.include_router(catalog_router)
app.include_router(admin_router)
app.include_router(exam_router)
app.include_router(cron_router)

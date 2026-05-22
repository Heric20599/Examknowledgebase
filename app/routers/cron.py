import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/cron", tags=["cron"])


def require_cron_secret(
    request: Request,
    x_cron_secret: str | None = Header(default=None, alias="X-Cron-Secret"),
) -> None:
    expected = (request.app.state.settings.cron_secret or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="CRON_SECRET is not configured on the server. Set it in .env to enable /internal/cron routes.",
        )
    if not x_cron_secret or x_cron_secret != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Cron-Secret header.")


@router.get("/ping", summary="Cron keep-alive (requires X-Cron-Secret)")
def cron_ping(_: None = Depends(require_cron_secret)) -> dict:
    """Lightweight endpoint for an external scheduler (every 15 min). Does not run ingest or exam jobs."""
    logger.info("Cron ping OK")
    return {
        "status": "ok",
        "message": "cron ping",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

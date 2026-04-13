"""AEGIS — Answer Engine & Generative Intelligence Suite.

FastAPI application entry point. Mounts the AEO analyzer and
Query Fan-Out routers under /api/aeo and /api/fanout respectively.

Run with:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app.api import aeo, fanout

app = FastAPI(
    title="AEGIS",
    description="AI-powered content scoring and query decomposition for AEO/GEO optimization.",
    version="1.0.0",
)

app.include_router(aeo.router, prefix="/api/aeo", tags=["AEO Content Scorer"])
app.include_router(fanout.router, prefix="/api/fanout", tags=["Query Fan-Out Engine"])


@app.get("/")
async def root():
    """Health check / welcome endpoint."""
    return {
        "service": "AEGIS",
        "version": "1.0.0",
        "endpoints": ["/api/aeo/analyze", "/api/fanout/generate"],
    }

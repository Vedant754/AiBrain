"""
Application entrypoint.

WHY THIS FILE IS SMALL ON PURPOSE:
main.py should only ASSEMBLE the app - create the FastAPI instance,
attach middleware (like CORS), and include routers. It should contain
ZERO business logic. If you ever find yourself writing "if" statements
about chunking or embeddings in this file, that logic belongs in a
service instead.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health
from app.api.routes import documents
from app.core.config import settings
from app.api.routes import search

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(health.router, prefix="/api", tags=["version"])
app.include_router(documents.router, prefix="/api", tags=["documents"])
app.include_router(search.router, prefix="/api", tags=["search"])
from fastapi import FastAPI

from server.routers import router

app = FastAPI(title="CpG Island Analyzer")

app.include_router(router)

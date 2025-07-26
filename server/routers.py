import asyncio
from fastapi import APIRouter
from concurrent.futures import ProcessPoolExecutor

from models.requests import CpGRequest, WindowRequest, IslandsRequest
from services.cpg import compute_global_cpg, compute_windows_cpg
from services.island_detection import detect_islands
from services.encode_api import fetch_and_annotate_islands
from utils import windows_summary, extract_sequence

router = APIRouter()
executor = ProcessPoolExecutor()


@router.post("/cpg/global")
async def cpg_global(request: CpGRequest) -> dict:
    sequence = extract_sequence(request.sequence, request.format)

    loop = asyncio.get_event_loop()
    result: list[dict] = await loop.run_in_executor(
        executor,
        compute_global_cpg,
        sequence
    )

    return {"cpg_percent": round(result, 2)}


@router.post("/cpg/sliding")
async def sliding_cpg(request: WindowRequest) -> dict:
    sequence = extract_sequence(request.sequence, request.format)

    loop = asyncio.get_event_loop()

    windows: list[dict] = await loop.run_in_executor(
        executor,
        compute_windows_cpg,
        sequence, request.window_size, request.step_size
    )

    return {"windows": windows, "summary": windows_summary(windows)}


@router.post("/cpg/islands")
async def cpg_islands(request: IslandsRequest) -> dict:

    loop = asyncio.get_event_loop()
    islands: list[dict] = await loop.run_in_executor(
        executor,
        detect_islands,
        request.sequence, request.format,
        request.window_size, request.step_size
    )

    annotated = await fetch_and_annotate_islands(islands, request)
    return {"islands": annotated}

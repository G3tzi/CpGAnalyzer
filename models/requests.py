from pydantic import BaseModel


class CpGRequest(BaseModel):
    sequence: str
    format: str


class WindowRequest(CpGRequest):
    window_size: int
    step_size: int


class IslandsRequest(WindowRequest):
    genome: str
    chrom: str | None = None
    start: int | None = None

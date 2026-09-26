import time


def start_timer() -> float:
    return time.perf_counter()


def elapsed(start: float) -> float:
    return time.perf_counter() - start


def print_stage_summary(
    stage: str,
    attempted: int,
    success: int,
    skipped: int,
    failed: int,
    seconds: float,
    unit_name: str,
) -> None:
    average = seconds / attempted if attempted else 0.0
    print(
        f"[{stage}] success={success} skipped={skipped} failed={failed} "
        f"duration={seconds:.1f}s avg={average:.1f}s/{unit_name}"
    )


def estimate_cost(seconds: float, hourly_cost: float) -> float:
    return seconds / 3600.0 * hourly_cost

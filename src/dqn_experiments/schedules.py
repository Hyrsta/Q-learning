from __future__ import annotations


def linear_schedule(initial_value: float, final_value: float, duration: int):
    def func(step: int) -> float:
        mix = min(1.0, step / float(duration))
        return initial_value + mix * (final_value - initial_value)

    return func

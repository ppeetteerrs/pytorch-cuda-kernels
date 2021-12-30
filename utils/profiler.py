from typing import Any, Callable, List, Tuple

from torch.profiler import profiler


def profile(fn: Callable[..., Any], kernels: List[str], iters: int = 1000):
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA], record_shapes=False
    ) as prof:
        for i in range(1000):
            fn(i)

    info: List[Tuple[str, float]] = []

    for activity in prof.key_averages():
        for kernel in kernels:
            if kernel in activity.key:
                info.append((kernel, activity.cuda_time_total / activity.count))

    sorted_info = sorted(info, key=lambda x: x[1])

    print(f" Profiling Results ({iters} iterations): ".center(80, "-"))
    for kernel, time in sorted_info:
        print(f"{kernel}: {time : .2f}ns")
    print("".center(80, "-"))

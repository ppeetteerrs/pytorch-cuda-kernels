from typing import Any, Callable, List, Tuple

from torch.profiler import profiler


def profile(fn: Callable[..., Any], kernels: List[str] = [], iters: int = 1000):
    # Wrap function in Pytorch profiler
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA], record_shapes=False
    ) as prof:
        for i in range(1000):
            fn(i)

    # Extract average time in CUDA for each activity and select relevant ones
    info: List[Tuple[str, float]] = []

    for activity in prof.key_averages():
        if len(kernels) > 0:
            for kernel in kernels:
                if kernel in activity.key:
                    info.append(
                        (activity.key, activity.cuda_time_total / activity.count)
                    )
        else:
            info.append((activity.key, activity.cuda_time_total / activity.count))

    # Sort activities based on average CUDA time
    sorted_info = sorted(info, key=lambda x: x[1])

    # Format the output and print to console
    max_name_len = max([len(kernel) for kernel, _ in sorted_info])
    print(f" Profiling Results ({iters} iterations): ".center(max_name_len + 50, "-"))
    for kernel, time in sorted_info:
        print(f"{kernel.rjust(max_name_len)}: {time : .2f}ns")
    print("".center(max_name_len + 50, "-"))

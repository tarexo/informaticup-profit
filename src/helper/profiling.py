import cProfile
import pstats

import os


def profile(func):
    def wrapper(*args, **kwargs):
        with cProfile.Profile() as pr:
            func(*args, **kwargs)

        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats(filename=os.path.join("debug", "profiling.prof"))

    return wrapper

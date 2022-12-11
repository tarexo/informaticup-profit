import cProfile
import pstats


def profile(func, *args, **kwargs):
    with cProfile.Profile() as pr:
        func(*args, **kwargs)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename="profiling.prof")

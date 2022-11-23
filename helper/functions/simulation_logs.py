from helper.constants.settings import PRINT_ROUND_LOG


def log_start_round(building, round, store_indices, cache_indices):
    """Generates and prints the start of round log for a given building if PRINT_ROUND_LOG is set.

    Args:
        building (Building): The building which requests a log print.
        round (int): Current round.
        store_indices (np.array): Array with indices where building.resources is non-zero.
        cache_indices (np.array): Array with indices where building.resource_cache is non-zero.
    """
    if not PRINT_ROUND_LOG:
        return
    str = f"{round} (start): ({building.x},{building.y}) accepts ["
    for i in cache_indices:
        str += f"{building.resource_cache[i]}x{i}, "
    str = str[:-2]
    str += "], holds ["
    for i in store_indices:
        str += f"{building.resources[i]}x{i}, "
    str = str[:-2]
    str += "]"
    print(str)


def log_factory_end_round(building, round, points):
    """Generates and prints the factory end of round log if PRINT_ROUND_LOG is set.

    Args:
        building (Building): The factory instance which requests a log print.
        round (int): Current round.
        points (int): Points of the produced product.
    """
    if not PRINT_ROUND_LOG:
        return
    str = f"{round} (end): ({building.x},{building.y}) produces {building.subtype} ({points} points)"
    print(str)


def log_deposit_end_round(building, round, takes_out):
    """Generates and prints the deposit end of round log if PRINT_ROUND_LOG is set.

    Args:
        building (Building): The deposit instance which requests a log print.
        round (int): Current round.
        takes_out (int): Number of items taken out of the deposit.resources array.
    """
    if not PRINT_ROUND_LOG:
        return
    str = f"{round} (end): ({building.x},{building.y}) takes [{takes_out}x{building.subtype}], [{building.resources[building.subtype] - takes_out}x{building.subtype}] available"
    print(str)

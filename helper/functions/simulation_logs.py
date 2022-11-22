def log_start_round(building, round, store_indices, cache_indices):
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
    str = f"{round} (end): ({building.x},{building.y}) produces {building.subtype} ({points} points)"
    print(str)


def log_deposit_end_round(building, round, takes_out):
    str = f"{round} (end): ({building.x},{building.y}) takes [{takes_out}x{building.subtype}], [{building.resources[building.subtype] - takes_out}x{building.subtype}] available"
    print(str)

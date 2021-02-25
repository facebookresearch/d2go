import os

def _cache_json_file(json_file):
    # TODO: entirely rely on PathManager for caching
    json_file = os.fspath(json_file)
    return json_file

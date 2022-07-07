

def parse_lat(latitude: str) -> tuple:

    assert ("S" in latitude or "N" in latitude), 'Unexpected name format'
    if "S" in latitude:
        max_lat = int(latitude.split("S")[0]) * -1
    else:
        max_lat = int(latitude.split("N")[0])
    min_lat = max_lat - 10

    return min_lat, max_lat

def parse_lon(longitude: str) -> tuple:
    assert ("W" in longitude or "E" in longitude), 'Unexpected name format'
    if "W" in longitude:
        max_lon = int(longitude.split("W")[0]) * -1       
    else:
        max_lon = int(longitude.split("E")[0])    
    min_lon = max_lon + 10

    return min_lon, max_lon

def get_extent(filename):
    _, _, _, lat, lon = filename.split('_')

    min_lat, max_lat = parse_lat(lat)
    min_lon, max_lon = parse_lon(lon)

    return min_lat, max_lat, min_lon, max_lon
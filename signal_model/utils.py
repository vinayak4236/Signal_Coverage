import math
from typing import Tuple, Optional
import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6371000.0  # meters


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in meters.
    Uses haversine formula for accuracy with small/medium distances.
    """
    rlat1, rlon1, rlat2, rlon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(EARTH_RADIUS_M * c)


def meters_to_deg(lat_ref: float, dx_m: float, dy_m: float) -> Tuple[float, float]:
    """Approximate meter-to-degree conversion relative to a reference latitude.
    - dx_m: east-west meters -> delta longitude degrees
    - dy_m: north-south meters -> delta latitude degrees
    Note: accurate for small distances (< ~10km) near lat_ref.
    """
    # 1 deg latitude ~ 111,320 m
    dlat = dy_m / 111320.0
    # 1 deg longitude ~ 111,320 * cos(lat)
    dlon = dx_m / (111320.0 * math.cos(math.radians(lat_ref)) or 1e-9)
    return dlon, dlat


def deg_to_meters(lat_ref: float, dlon: float, dlat: float) -> Tuple[float, float]:
    """Inverse of meters_to_deg for small deltas.
    Returns (dx_m, dy_m).
    """
    dy_m = dlat * 111320.0
    dx_m = dlon * (111320.0 * math.cos(math.radians(lat_ref)) or 1e-9)
    return dx_m, dy_m


def load_csv_dataset(path: str) -> pd.DataFrame:
    """Load dataset with columns [latitude, longitude, signal_strength].
    - Ensures proper dtypes and drops rows with missing values.
    - Clips signal_strength to a reasonable dBm range (-130..-30) if present.
    """
    df = pd.read_csv(path)
    needed = {"latitude", "longitude", "signal_strength"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df = df.dropna(subset=["latitude", "longitude", "signal_strength"])  # basic cleaning
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["signal_strength"] = pd.to_numeric(df["signal_strength"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    # Clip extreme outliers
    df["signal_strength"] = df["signal_strength"].clip(lower=-130, upper=-30)
    return df


def dataset_centroid(df: pd.DataFrame) -> Tuple[float, float]:
    """Return centroid (lat, lon) of the dataset as a simple mean.
    For small areas, mean is acceptable; for large areas, consider geodesic centroid.
    """
    return float(df["latitude"].mean()), float(df["longitude"].mean())


def generate_prediction_grid(
    center_lat: float,
    center_lon: float,
    radius_m: float = 2000.0,
    spacing_m: float = 50.0,
    clip_to_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> pd.DataFrame:
    """Generate a grid of coordinates around a center using meter spacing.
    - radius_m: radial extent from center
    - spacing_m: grid spacing (approx meters)
    - clip_to_bbox: optional (min_lat, min_lon, max_lat, max_lon)
    Returns DataFrame with [latitude, longitude].
    """
    # Determine step in degrees at center latitude
    dlon_step_deg, dlat_step_deg = meters_to_deg(center_lat, spacing_m, spacing_m)

    # Compute extents in degrees
    dlon_radius_deg, dlat_radius_deg = meters_to_deg(center_lat, radius_m, radius_m)

    lats = np.arange(center_lat - dlat_radius_deg, center_lat + dlat_radius_deg + 1e-12, dlat_step_deg)
    lons = np.arange(center_lon - dlon_radius_deg, center_lon + dlon_radius_deg + 1e-12, dlon_step_deg)

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    # Optionally clip to circular radius (more natural) and bbox
    flat_lat = lat_grid.ravel()
    flat_lon = lon_grid.ravel()

    # Keep only points within radius_m (circle)
    within = np.array([
        haversine_distance_m(center_lat, center_lon, la, lo) <= radius_m for la, lo in zip(flat_lat, flat_lon)
    ])

    if clip_to_bbox is not None:
        min_lat, min_lon, max_lat, max_lon = clip_to_bbox
        within_bbox = (flat_lat >= min_lat) & (flat_lat <= max_lat) & (flat_lon >= min_lon) & (flat_lon <= max_lon)
        within = within & within_bbox

    coords = pd.DataFrame({
        "latitude": flat_lat[within],
        "longitude": flat_lon[within],
    }).reset_index(drop=True)

    return coords
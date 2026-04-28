#!/usr/bin/env python3
"""Build the lightweight JSON grid used by yangliu02.html.

The browser intentionally does not read RTOFS/GFS/NetCDF/GRIB directly. This
script follows the earth.nullschool-style preprocessing step: download current
public model/analysis products, resample them to a small regular grid, and write
one JSON file that the single-page classroom app can load quickly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
import xarray as xr
from scipy.spatial import cKDTree


RTOFS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtofs/prod"
GFS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
GFS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

GRID = {
    "latStart": -89,
    "latEnd": 89,
    "lonStart": -180,
    "lonEnd": 178,
    "latStep": 2,
    "lonStep": 2,
    "latCount": 90,
    "lonCount": 180,
}

USER_AGENT = "oceandata-yangliu-live/1.0 (+https://github.com/xfcy111/oceandata)"


def log(message: str) -> None:
    print(f"[yangliu-data] {message}", flush=True)


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def request_get(url: str, *, timeout: int = 60) -> requests.Response:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response


def list_links(url: str, pattern: str) -> list[str]:
    html = request_get(url, timeout=45).text
    return re.findall(pattern, html)


def head_ok(url: str, timeout: int = 25) -> bool:
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True, headers={"User-Agent": USER_AGENT})
        return response.status_code == 200
    except requests.RequestException:
        return False


def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1024 * 1024:
        log(f"Using cached {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
        return path

    log(f"Downloading {url}")
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=180, headers={"User-Agent": USER_AGENT}) as response:
        response.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp.replace(path)
    log(f"Saved {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    return path


def target_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lats = np.arange(GRID["latStart"], GRID["latEnd"] + 0.001, GRID["latStep"], dtype=np.float32)
    lons = np.arange(GRID["lonStart"], GRID["lonEnd"] + 0.001, GRID["lonStep"], dtype=np.float32)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lats, lons, lat2d, lon2d


def lon_to_180(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180) % 360) - 180


def xyz_from_latlon(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    cos_lat = np.cos(lat_rad)
    return np.column_stack((cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)))


def finite_values(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    arr = np.where(np.abs(arr) > 1.0e20, np.nan, arr)
    return arr


def regrid_nearest(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    src_value: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    *,
    max_degrees: float,
    decimals: int,
) -> list[float | None]:
    lat = np.asarray(src_lat, dtype=np.float64)
    lon = lon_to_180(np.asarray(src_lon, dtype=np.float64))
    value = finite_values(src_value)
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(value)
    if valid.sum() < 4:
        raise RuntimeError("not enough valid source points for regridding")

    src_xyz = xyz_from_latlon(lat[valid].ravel(), lon[valid].ravel())
    tree = cKDTree(src_xyz)
    target_xyz = xyz_from_latlon(target_lat.ravel(), lon_to_180(target_lon).ravel())
    max_chord = 2 * math.sin(math.radians(max_degrees) / 2)
    distance, index = tree.query(target_xyz, k=1, distance_upper_bound=max_chord)
    flat_values = value[valid].ravel()

    out: list[float | None] = []
    for dist, idx in zip(distance, index):
        if not np.isfinite(dist) or idx >= flat_values.size:
            out.append(None)
        else:
            val = float(flat_values[idx])
            out.append(round(val, decimals))
    return out


def find_name(names: list[str], candidates: list[str], contains: list[str] | None = None) -> str | None:
    lower = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if contains:
        for name in names:
            low = name.lower()
            if all(token in low for token in contains):
                return name
    return None


def select_2d(da: xr.DataArray) -> xr.DataArray:
    result = da
    for dim in list(result.dims):
        low = dim.lower()
        if low in {"time", "mt", "date", "valid_time", "step", "depth", "depthu", "depthv", "lev", "level", "z"}:
            result = result.isel({dim: 0})
    while result.ndim > 2:
        result = result.isel({result.dims[0]: 0})
    if result.ndim != 2:
        raise RuntimeError(f"{da.name} is not reducible to 2D; dims={da.dims}")
    return result.squeeze(drop=True)


def dataset_time(ds: xr.Dataset, fallback: str) -> str:
    for key in ("time", "MT", "valid_time", "forecast_reference_time"):
        if key in ds:
            value = np.asarray(ds[key]).ravel()
            if value.size:
                try:
                    return str(np.datetime_as_string(value[0], unit="s")).replace("+0000", "Z")
                except Exception:
                    return str(value[0])
    return fallback


def lat_lon_for(ds: xr.Dataset, da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    names = list(ds.variables)
    lat_name = find_name(names, ["lat", "latitude", "Latitude", "Latitude_t", "nav_lat"], contains=["lat"])
    lon_name = find_name(names, ["lon", "longitude", "Longitude", "Longitude_t", "nav_lon"], contains=["lon"])
    if not lat_name or not lon_name:
        raise RuntimeError("dataset has no recognizable latitude/longitude variables")

    lat = np.asarray(ds[lat_name])
    lon = np.asarray(ds[lon_name])
    shape = da.shape

    if lat.shape == shape and lon.shape == shape:
        return lat, lon
    if lat.ndim == 1 and lon.ndim == 1:
        if lat.size == shape[0] and lon.size == shape[1]:
            lon2d, lat2d = np.meshgrid(lon, lat)
            return lat2d, lon2d
        if lon.size == shape[0] and lat.size == shape[1]:
            lat2d, lon2d = np.meshgrid(lat, lon)
            return lat2d, lon2d
    raise RuntimeError(f"lat/lon shape mismatch for {da.name}: data={shape}, lat={lat.shape}, lon={lon.shape}")


def extract_field(ds: xr.Dataset, candidates: list[str], contains: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    name = find_name(list(ds.data_vars), candidates, contains)
    if not name:
        raise RuntimeError(f"missing variable; tried {', '.join(candidates)}; available={', '.join(list(ds.data_vars)[:40])}")
    da = select_2d(ds[name])
    lat, lon = lat_lon_for(ds, da)
    data = finite_values(da.values)
    return lat, lon, data, name


def latest_rtofs_urls(cache: Path) -> tuple[str, str, str]:
    dirs = sorted(set(list_links(f"{RTOFS_BASE}/", r'href="(rtofs\.\d{8}/)"')), reverse=True)
    if not dirs:
        raise RuntimeError("no RTOFS date directories found")
    for dirname in dirs[:5]:
        base = f"{RTOFS_BASE}/{dirname}"
        prog = f"{base}rtofs_glo_2ds_f000_prog.nc"
        diag = f"{base}rtofs_glo_2ds_f000_diag.nc"
        if head_ok(prog):
            return dirname.rstrip("/").split(".")[-1], prog, diag
    raise RuntimeError("no recent RTOFS f000 prog file is reachable")


def build_rtofs(cache: Path, target_lat: np.ndarray, target_lon: np.ndarray) -> tuple[dict, dict, list[dict]]:
    date_code, prog_url, diag_url = latest_rtofs_urls(cache)
    prog_path = download(prog_url, cache / f"rtofs_{date_code}_f000_prog.nc")
    errors: list[dict] = []
    fields: dict[str, list[float | None]] = {}
    times: dict[str, str | None] = {"current": None, "sst": None}
    sources: list[dict] = []

    with xr.open_dataset(prog_path, decode_times=True, mask_and_scale=True) as ds:
        source_time = dataset_time(ds, f"{date_code}T00:00:00Z")
        try:
            lat_u, lon_u, u, u_name = extract_field(
                ds,
                ["u_velocity", "water_u", "uo", "u", "eastward_sea_water_velocity", "sea_water_x_velocity"],
                contains=["u", "vel"],
            )
            lat_v, lon_v, v, v_name = extract_field(
                ds,
                ["v_velocity", "water_v", "vo", "v", "northward_sea_water_velocity", "sea_water_y_velocity"],
                contains=["v", "vel"],
            )
            fields["uCurrent"] = regrid_nearest(lat_u, lon_u, u, target_lat, target_lon, max_degrees=1.8, decimals=4)
            fields["vCurrent"] = regrid_nearest(lat_v, lon_v, v, target_lat, target_lon, max_degrees=1.8, decimals=4)
            times["current"] = source_time
            sources.append({
                "key": "current",
                "label": "NOAA RTOFS Global surface current",
                "variables": f"{u_name}, {v_name}",
                "units": "m/s",
                "time": source_time,
                "url": prog_url,
                "description": "RTOFS f000 NetCDF, resampled to 2-degree global grid."
            })
        except Exception as exc:
            errors.append({"key": "current", "label": "NOAA RTOFS surface current", "message": str(exc)})

        try:
            lat_t, lon_t, temp, t_name = extract_field(
                ds,
                ["temperature", "water_temp", "sst", "sea_surface_temperature", "thetao", "sea_water_temperature", "temp"],
                contains=["temp"],
            )
            if np.nanmedian(temp) > 100:
                temp = temp - 273.15
            fields["sst"] = regrid_nearest(lat_t, lon_t, temp, target_lat, target_lon, max_degrees=1.8, decimals=2)
            times["sst"] = source_time
            sources.append({
                "key": "sst",
                "label": "NOAA RTOFS surface temperature fallback",
                "variables": t_name,
                "units": "degC",
                "time": source_time,
                "url": prog_url,
                "description": "Temporary SST layer from RTOFS surface temperature; OISST can replace this later."
            })
        except Exception as exc:
            errors.append({"key": "sst", "label": "RTOFS surface temperature fallback", "message": str(exc)})

    return {"fields": fields, "times": times, "sources": sources}, {"date": date_code, "progUrl": prog_url, "diagUrl": diag_url}, errors


def latest_gfs_filter_url() -> tuple[str, str]:
    dirs = sorted(set(list_links(f"{GFS_BASE}/", r'href="(gfs\.\d{8}/)"')), reverse=True)
    if not dirs:
        raise RuntimeError("no GFS date directories found")
    for dirname in dirs[:5]:
        date_code = dirname.strip("/").split(".")[-1]
        hours = sorted(set(list_links(f"{GFS_BASE}/{dirname}", r'href="(\d{2}/)"')), reverse=True)
        for hour_dir in hours:
            hour = hour_dir.strip("/")
            file_name = f"gfs.t{hour}z.pgrb2.0p25.f000"
            query = (
                f"dir={quote(f'/gfs.{date_code}/{hour}/atmos')}"
                f"&file={file_name}"
                "&lev_10_m_above_ground=on&var_UGRD=on&var_VGRD=on"
            )
            url = f"{GFS_FILTER}?{query}"
            try:
                response = requests.get(url, stream=True, timeout=45, headers={"User-Agent": USER_AGENT})
                response.raise_for_status()
                first = next(response.iter_content(chunk_size=32), b"")
                response.close()
                if first.startswith(b"GRIB"):
                    return f"{date_code}T{hour}:00:00Z", url
            except Exception:
                continue
    raise RuntimeError("no recent GFS f000 10m wind filter file is reachable")


def build_gfs(cache: Path, target_lat: np.ndarray, target_lon: np.ndarray) -> tuple[dict, list[dict]]:
    source_time, url = latest_gfs_filter_url()
    path = download(url, cache / f"gfs_{source_time.replace(':', '').replace('-', '')}_10m_wind.grib2")
    try:
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    except Exception as exc:
        raise RuntimeError(f"cfgrib could not open filtered GFS file: {exc}") from exc
    with ds:
        lat_u, lon_u, u, u_name = extract_field(ds, ["u10", "10u", "u", "UGRD"], contains=["u"])
        lat_v, lon_v, v, v_name = extract_field(ds, ["v10", "10v", "v", "VGRD"], contains=["v"])
        fields = {
            "windU": regrid_nearest(lat_u, lon_u, u, target_lat, target_lon, max_degrees=2.2, decimals=3),
            "windV": regrid_nearest(lat_v, lon_v, v, target_lat, target_lon, max_degrees=2.2, decimals=3),
        }
    return {
        "fields": fields,
        "time": source_time,
        "source": {
            "key": "wind",
            "label": "NOAA GFS f000 10m wind",
            "variables": f"{u_name}, {v_name}",
            "units": "m/s",
            "time": source_time,
            "url": url,
            "description": "GFS analysis hour f000, filtered to 10m U/V wind and resampled to 2-degree grid."
        }
    }, []


def empty_field(size: int) -> list[None]:
    return [None] * size


def count_valid(values: list[float | None]) -> int:
    return sum(1 for value in values if value is not None)


def build(output: Path, cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    _, _, target_lat, target_lon = target_grid()
    size = GRID["latCount"] * GRID["lonCount"]
    fields: dict[str, list[float | None]] = {
        "uCurrent": empty_field(size),
        "vCurrent": empty_field(size),
        "sst": empty_field(size),
        "windU": empty_field(size),
        "windV": empty_field(size),
    }
    times: dict[str, str | None] = {"current": None, "sst": None, "wind": None}
    sources: list[dict] = []
    errors: list[dict] = []

    try:
        rtofs, _, rtofs_errors = build_rtofs(cache, target_lat, target_lon)
        fields.update(rtofs["fields"])
        times.update({key: value for key, value in rtofs["times"].items() if value})
        sources.extend(rtofs["sources"])
        errors.extend(rtofs_errors)
    except Exception as exc:
        errors.append({"key": "current", "label": "NOAA RTOFS", "message": str(exc)})

    try:
        gfs, gfs_errors = build_gfs(cache, target_lat, target_lon)
        fields.update(gfs["fields"])
        times["wind"] = gfs["time"]
        sources.append(gfs["source"])
        errors.extend(gfs_errors)
    except Exception as exc:
        errors.append({"key": "wind", "label": "NOAA GFS f000 10m wind", "message": str(exc)})

    for source in sources:
        key = source["key"]
        if key == "current":
            source["validCount"] = min(count_valid(fields["uCurrent"]), count_valid(fields["vCurrent"]))
        elif key == "sst":
            source["validCount"] = count_valid(fields["sst"])
        elif key == "wind":
            source["validCount"] = min(count_valid(fields["windU"]), count_valid(fields["windV"]))

    return {
        "version": "yangliu-live-v1",
        "generatedAt": iso_now(),
        "grid": GRID,
        "fields": fields,
        "times": times,
        "sources": sources,
        "errors": errors,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/yangliu-live-v1.json")
    parser.add_argument("--cache", default=".cache/yangliu")
    args = parser.parse_args(argv)

    output = Path(args.output)
    cache = Path(args.cache)
    payload = build(output, cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(output.parent), suffix=".json") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write("\n")
        tmp_name = fh.name
    os.replace(tmp_name, output)
    log(f"Wrote {output} ({output.stat().st_size / 1024:.1f} KiB)")
    for source in payload["sources"]:
        log(f"Loaded {source['label']}: {source.get('validCount', 0)} valid cells")
    for error in payload["errors"]:
        log(f"Layer warning {error['label']}: {error['message']}")
    if not payload["sources"]:
        log("No layers were generated. The JSON still records the failure state.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

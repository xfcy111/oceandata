#!/usr/bin/env python3
"""Build fixed winter/summer JSON snapshots for the scenario mode.

The output is intentionally shaped like the live JSON used by index.html, but
wrapped in two named scenes. It uses public, no-login sources:

- NOAA RTOFS public S3 2D diagnostics for historical ocean current vectors.
- NOAA OISST ERDDAP subset for daily sea surface temperature.
- NOAA GFS public S3 f000 10m wind vectors, with NOAA PSL/NCEP as a fallback.

The browser still only reads small JSON files; it never downloads NetCDF/GRIB.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from scipy.spatial import cKDTree


SCENES = [
    ("winter", "冬季", "2025-01-01"),
    ("summer", "夏季", "2025-07-30"),
]

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

USER_AGENT = "oceandata-yangliu-seasonal/1.0 (+https://github.com/xfcy111/oceandata)"
RTOFS_S3 = "https://noaa-nws-rtofs-pds.s3.amazonaws.com"
GFS_S3 = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
OISST_ERDDAP = "https://www.ncei.noaa.gov/erddap/griddap/ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon.nc"
PSL_NCEP_BASE = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface_gauss"


def log(message: str) -> None:
    print(f"[yangliu-seasonal] {message}", flush=True)


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def request_get(url: str, *, timeout: int = 60) -> requests.Response:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response


def configure_ecmwflibs() -> None:
    """Make pip-installed ecCodes discoverable on Windows when available."""
    try:
        import ecmwflibs  # type: ignore

        lib_path = Path(ecmwflibs.find("eccodes"))
        os.environ.setdefault("ECCODES_PYTHON_USE_FINDLIBS", "1")
        os.environ["PATH"] = f"{lib_path.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    except Exception:
        return


def download(url: str, path: Path, *, timeout: int = 240) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1024:
        log(f"Using cached {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
        return path

    log(f"Downloading {url}")
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT}) as response:
        response.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp.replace(path)
    log(f"Saved {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    return path


def download_gfs_10m_wind(date: str, cache: Path) -> tuple[Path, str, str]:
    date_code = date.replace("-", "")
    hour = "00"
    file_name = f"gfs.t{hour}z.pgrb2.0p25.f000"
    base_url = f"{GFS_S3}/gfs.{date_code}/{hour}/atmos"
    data_url = f"{base_url}/{file_name}"
    idx_url = f"{data_url}.idx"
    out = cache / f"gfs_{date_code}_{hour}_10m_wind.grib2"
    if out.exists() and out.stat().st_size > 1024:
        log(f"Using cached {out.name} ({out.stat().st_size / 1024:.1f} KiB)")
        return out, data_url, f"{date}T{hour}:00:00Z"

    lines = request_get(idx_url, timeout=45).text.splitlines()
    entries: list[tuple[int, str]] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) >= 3:
            entries.append((int(parts[1]), line))
    ranges = []
    for index, (start, line) in enumerate(entries):
        if (":UGRD:" in line or ":VGRD:" in line) and ":10 m above ground:" in line:
            end = entries[index + 1][0] - 1 if index + 1 < len(entries) else None
            ranges.append((start, end, line))
    if len(ranges) < 2:
        raise RuntimeError("GFS index did not contain both 10m UGRD and VGRD records")

    log(f"Downloading GFS 10m wind byte ranges from {idx_url}")
    tmp = out.with_suffix(out.suffix + ".part")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as fh:
        for start, end, line in ranges[:2]:
            header = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
            response = requests.get(data_url, headers={"Range": header, "User-Agent": USER_AGENT}, timeout=120)
            response.raise_for_status()
            fh.write(response.content)
            log(f"  {header} {line}")
    tmp.replace(out)
    return out, data_url, f"{date}T{hour}:00:00Z"


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
    max_chord = 2 * np.sin(np.radians(max_degrees) / 2)
    distance, index = tree.query(target_xyz, k=1, distance_upper_bound=max_chord)
    flat_values = value[valid].ravel()

    out: list[float | None] = []
    for dist, idx in zip(distance, index):
        if not np.isfinite(dist) or idx >= flat_values.size:
            out.append(None)
        else:
            out.append(round(float(flat_values[idx]), decimals))
    return out


def standardize_lon_lat(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in [("longitude", "lon"), ("latitude", "lat")]:
        if old in ds.coords and new not in ds.coords:
            rename[old] = new
    if rename:
        ds = ds.rename(rename)
    if "lon" in ds.coords:
        lon = (((ds["lon"] + 180) % 360) - 180).astype(float)
        ds = ds.assign_coords(lon=lon).sortby("lon")
    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    return ds


def select_time(da: xr.DataArray, date: str) -> xr.DataArray:
    for dim in da.dims:
        if dim.lower() in {"time", "valid_time", "mt"}:
            return da.sel({dim: np.datetime64(date)}, method="nearest")
    return da


def squeeze_to_lat_lon(da: xr.DataArray) -> xr.DataArray:
    result = da
    for dim in list(result.dims):
        if dim.lower() in {"time", "valid_time", "mt", "level", "zlev"}:
            result = result.isel({dim: 0})
    while result.ndim > 2:
        result = result.isel({result.dims[0]: 0})
    return result.squeeze(drop=True)


def regrid_regular(da: xr.DataArray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    ds = standardize_lon_lat(da.to_dataset(name="field"))
    field = ds["field"].interp(lat=lats, lon=lons, method="linear")
    return field.transpose("lat", "lon").values


def dataset_time(ds: xr.Dataset, fallback: str) -> str:
    for key in ("time", "MT", "valid_time"):
        if key in ds:
            value = np.asarray(ds[key]).ravel()
            if value.size:
                try:
                    return str(np.datetime_as_string(value[0], unit="s")).replace("+0000", "Z")
                except Exception:
                    return str(value[0])
    return f"{fallback}T00:00:00Z"


def build_rtofs_current(cache: Path, date: str, target_lat: np.ndarray, target_lon: np.ndarray) -> tuple[dict, dict]:
    date_code = date.replace("-", "")
    url = f"{RTOFS_S3}/rtofs.{date_code}/rtofs_glo_2ds_f000_diag.nc"
    path = download(url, cache / f"rtofs_{date_code}_f000_diag.nc")
    with xr.open_dataset(path, decode_times=True, mask_and_scale=True) as ds:
        time = dataset_time(ds, date)
        u = squeeze_to_lat_lon(select_time(ds["u_barotropic_velocity"], date)).values
        v = squeeze_to_lat_lon(select_time(ds["v_barotropic_velocity"], date)).values
        lat = np.asarray(ds["Latitude"])
        lon = np.asarray(ds["Longitude"])
        fields = {
            "uCurrent": regrid_nearest(lat, lon, u, target_lat, target_lon, max_degrees=1.8, decimals=4),
            "vCurrent": regrid_nearest(lat, lon, v, target_lat, target_lon, max_degrees=1.8, decimals=4),
        }
    return fields, {
        "key": "current",
        "label": "NOAA RTOFS barotropic ocean current",
        "variables": "u_barotropic_velocity, v_barotropic_velocity",
        "units": "m/s",
        "time": time,
        "url": url,
        "dataType": "historical",
        "description": "RTOFS public S3 f000 2D diagnostic current vectors, resampled to a 2-degree global grid.",
    }


def build_oisst(cache: Path, date: str, lats: np.ndarray, lons: np.ndarray) -> tuple[dict, dict]:
    date_code = date.replace("-", "")
    query = (
        f"sst[({date}T12:00:00Z)][(0.0)]"
        "[(-89.875):8:(89.875)][(0.125):8:(359.875)]"
    )
    url = f"{OISST_ERDDAP}?{query}"
    path = download(url, cache / f"oisst_{date_code}.nc", timeout=120)
    with xr.open_dataset(path, decode_times=True, mask_and_scale=True) as ds:
        da = squeeze_to_lat_lon(select_time(ds["sst"], date))
        values = regrid_regular(da, lats, lons)
        time = dataset_time(ds, f"{date}T12:00:00Z")
    return {"sst": round_or_none(values, 2)}, {
        "key": "sst",
        "label": "NOAA OISST daily sea surface temperature",
        "variables": "sst",
        "units": "degC",
        "time": time,
        "url": "https://www.ncei.noaa.gov/products/optimum-interpolation-sst",
        "dataType": "historical",
        "description": "NOAA OISST AVHRR daily sea surface temperature subset via NCEI ERDDAP.",
    }


def build_gfs_wind(cache: Path, date: str, lats: np.ndarray, lons: np.ndarray) -> tuple[dict, dict]:
    configure_ecmwflibs()
    path, url, time = download_gfs_10m_wind(date, cache)
    try:
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    except Exception as exc:
        raise RuntimeError(f"cfgrib could not open GFS 10m wind GRIB2: {exc}") from exc
    with ds:
        u_name = "u10" if "u10" in ds else "u"
        v_name = "v10" if "v10" in ds else "v"
        u = squeeze_to_lat_lon(ds[u_name])
        v = squeeze_to_lat_lon(ds[v_name])
        u_values = regrid_regular(u, lats, lons)
        v_values = regrid_regular(v, lats, lons)
    return {
        "windU": round_or_none(u_values, 3),
        "windV": round_or_none(v_values, 3),
    }, {
        "key": "wind",
        "label": "NOAA GFS f000 10m wind",
        "variables": f"{u_name}, {v_name}",
        "units": "m/s",
        "time": time,
        "url": url,
        "dataType": "historical",
        "description": "GFS analysis hour f000 10m U/V wind from NOAA public S3, resampled to a 2-degree global grid.",
    }


def build_ncep_wind(cache: Path, date: str, lats: np.ndarray, lons: np.ndarray) -> tuple[dict, dict]:
    year = date[:4]
    u_url = f"{PSL_NCEP_BASE}/uwnd.10m.gauss.{year}.nc"
    v_url = f"{PSL_NCEP_BASE}/vwnd.10m.gauss.{year}.nc"
    u_path = download(u_url, cache / f"uwnd_10m_{year}.nc")
    v_path = download(v_url, cache / f"vwnd_10m_{year}.nc")
    with xr.open_dataset(u_path, decode_times=True, mask_and_scale=True) as uds, xr.open_dataset(v_path, decode_times=True, mask_and_scale=True) as vds:
        u = squeeze_to_lat_lon(select_time(uds["uwnd"], date))
        v = squeeze_to_lat_lon(select_time(vds["vwnd"], date))
        u_values = regrid_regular(u, lats, lons)
        v_values = regrid_regular(v, lats, lons)
        time = dataset_time(uds.sel(time=np.datetime64(date), method="nearest"), date)
    return {
        "windU": round_or_none(u_values, 3),
        "windV": round_or_none(v_values, 3),
    }, {
        "key": "wind",
        "label": "NOAA PSL NCEP/NCAR Reanalysis 10m wind",
        "variables": "uwnd, vwnd",
        "units": "m/s",
        "time": time,
        "url": "https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.surfaceflux.html",
        "dataType": "historical",
        "description": "Historical 10m U/V wind from NOAA PSL NCEP/NCAR Reanalysis, resampled to a 2-degree global grid.",
    }


def build_wind(cache: Path, date: str, lats: np.ndarray, lons: np.ndarray) -> tuple[dict, dict]:
    try:
        return build_gfs_wind(cache, date, lats, lons)
    except Exception as exc:
        log(f"GFS wind unavailable for {date}; using NCEP/NCAR fallback: {exc}")
        fields, source = build_ncep_wind(cache, date, lats, lons)
        source["description"] = f"{source['description']} Fallback used because GFS S3 GRIB2 could not be read: {exc}"
        return fields, source


def round_or_none(values: np.ndarray, decimals: int) -> list[float | None]:
    arr = finite_values(values)
    out: list[float | None] = []
    for value in arr.ravel():
        out.append(round(float(value), decimals) if np.isfinite(value) else None)
    return out


def empty_field(size: int) -> list[None]:
    return [None] * size


def count_valid(values: list[float | None]) -> int:
    return sum(1 for value in values if value is not None)


def valid_count_for(key: str, fields: dict[str, list[float | None]]) -> int:
    if key == "current":
        return min(count_valid(fields["uCurrent"]), count_valid(fields["vCurrent"]))
    if key == "wind":
        return min(count_valid(fields["windU"]), count_valid(fields["windV"]))
    if key == "sst":
        return count_valid(fields["sst"])
    return 0


def build_scene(cache: Path, scene_id: str, label: str, date: str) -> dict:
    lats, lons, target_lat, target_lon = target_grid()
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

    for key, builder in [
        ("current", lambda: build_rtofs_current(cache, date, target_lat, target_lon)),
        ("sst", lambda: build_oisst(cache, date, lats, lons)),
        ("wind", lambda: build_wind(cache, date, lats, lons)),
    ]:
        try:
            layer_fields, source = builder()
            fields.update(layer_fields)
            times[key] = source["time"]
            source["validCount"] = valid_count_for(key, fields)
            sources.append(source)
        except Exception as exc:
            errors.append({"key": key, "label": key, "message": str(exc)})

    return {
        "id": scene_id,
        "label": label,
        "date": date,
        "dataType": "historical",
        "timeModeLabel": "历史固定日期",
        "generatedAt": iso_now(),
        "grid": GRID,
        "fields": fields,
        "times": times,
        "sources": sources,
        "errors": errors,
    }


def build(output: Path, cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    scenes = {scene_id: build_scene(cache, scene_id, label, date) for scene_id, label, date in SCENES}
    return {
        "version": "yangliu-seasonal-v1",
        "generatedAt": iso_now(),
        "grid": GRID,
        "scenes": scenes,
        "sources": [
            {
                "name": "NOAA RTOFS public S3",
                "url": RTOFS_S3,
                "note": "Historical f000 2D diagnostic ocean current vectors.",
            },
            {
                "name": "NOAA OISST via NCEI ERDDAP",
                "url": "https://www.ncei.noaa.gov/products/optimum-interpolation-sst",
                "note": "Daily sea surface temperature.",
            },
            {
                "name": "NOAA GFS public S3",
                "url": GFS_S3,
                "note": "Historical f000 10m U/V wind, with NOAA PSL NCEP/NCAR Reanalysis fallback if GRIB2 cannot be read.",
            },
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/yangliu-seasonal-v1.json")
    parser.add_argument("--cache", default=".cache/yangliu-seasonal")
    args = parser.parse_args(argv)

    output = Path(args.output)
    payload = build(output, Path(args.cache))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(output.parent), suffix=".json") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write("\n")
        tmp_name = fh.name
    os.replace(tmp_name, output)
    log(f"Wrote {output} ({output.stat().st_size / 1024:.1f} KiB)")
    for scene in payload["scenes"].values():
        loaded = ", ".join(f"{source['key']}={source.get('validCount', 0)}" for source in scene["sources"])
        log(f"{scene['label']} {scene['date']}: {loaded or 'no layers'}")
        for error in scene["errors"]:
            log(f"Layer warning {scene['id']} {error['label']}: {error['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

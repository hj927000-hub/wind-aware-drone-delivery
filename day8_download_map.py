"""Download/cache OSM building polygons around Times Square (no API key).

Network access is needed only to refresh the bundled snapshot.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import osmnx as ox
import pandas as pd
from pyproj import Transformer
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parent
CENTER_LON, CENTER_LAT = -73.9855, 40.758
CRS_METRIC = 'EPSG:32618'  # UTM zone 18N; units are meters.
SIDE_M = 1000.0


def download(destination, refresh=False):
    """One bounded Overpass request; keep raw XML for reproducibility."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / 'manhattan_buildings.osm'
    fwd = Transformer.from_crs(4326, CRS_METRIC, always_xy=True)
    inv = Transformer.from_crs(CRS_METRIC, 4326, always_xy=True)
    cx, cy = fwd.transform(CENTER_LON, CENTER_LAT)
    # Fetch beyond the study square so boundary-crossing buildings are retained.
    pad = SIDE_M / 2 + 100
    corners = [inv.transform(cx + dx, cy + dy)
               for dx in (-pad, pad) for dy in (-pad, pad)]
    west = min(p[0] for p in corners); east = max(p[0] for p in corners)
    south = min(p[1] for p in corners); north = max(p[1] for p in corners)
    query = (f'[out:xml][timeout:45];nwr["building"]'
             f'({south},{west},{north},{east});(._;>;);out body;')
    if refresh or not raw_path.exists():
        req = urllib.request.Request(
            'https://overpass-api.de/api/interpreter',
            data=urllib.parse.urlencode({'data': query}).encode(),
            headers={'User-Agent': 'HJ-DroneLearning/0.1 (educational map query)',
                     'Accept': 'application/xml'},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read()
        root = ET.fromstring(raw)
        if root.tag != 'osm' or root.find('remark') is not None:
            raise ValueError('Overpass did not return a complete OSM extract.')
        raw_path.write_bytes(raw)
    raw_root = ET.parse(raw_path).getroot()
    meta = raw_root.find('meta')
    timestamp = meta.get('osm_base') if meta is not None else None
    frame = ox.features_from_xml(raw_path, tags={'building': True})
    frame = frame[frame.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
    if frame.empty:
        raise ValueError('No building polygons returned; no synthetic replacement used.')
    tags = ('name', 'building', 'height', 'building:levels', 'min_height', 'roof:height')
    features = []
    for (element, osmid), row in frame.iterrows():
        props = {'osm_id': f'{element}/{osmid}'}
        for tag in tags:
            value = row.get(tag)
            props[tag] = None if value is None or pd.isna(value) else str(value)
        features.append({'type': 'Feature', 'properties': props,
                         'geometry': mapping(row.geometry)})
    collection = {'type': 'FeatureCollection', 'features': features}
    out = destination / 'manhattan_buildings.geojson'
    out.write_text(json.dumps(collection, ensure_ascii=False, allow_nan=False))
    metadata = {
        'source': 'OpenStreetMap via Overpass API',
        'attribution': '© OpenStreetMap contributors',
        'source_url': 'https://www.openstreetmap.org',
        'license': 'ODbL 1.0', 'license_url': 'https://www.openstreetmap.org/copyright',
        'osm_base_timestamp': timestamp,
        'processed_utc': datetime.now(timezone.utc).isoformat(),
        'center_lon_lat': [CENTER_LON, CENTER_LAT], 'side_m': SIDE_M,
        'metric_crs': CRS_METRIC, 'origin_utm_m': [cx - SIDE_M/2, cy - SIDE_M/2],
        'feature_count_in_fetch': len(features),
        'height_policy': 'Preserve raw OSM tags. Missing heights are not inferred.',
        'query_bbox_lon_lat': [west, south, east, north],
    }
    (destination / 'manhattan_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Saved {len(features)} polygons: {out}')
    print(f'OSM snapshot: {timestamp}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    try:
        download(args.data_dir, args.refresh)
    except Exception as exc:
        parser.exit(1, f'Map download failed: {exc}\nUse the bundled snapshot, or retry later.\n')


if __name__ == '__main__':
    main()

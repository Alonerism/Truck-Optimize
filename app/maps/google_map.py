"""Write a tiny HTML file that draws polylines using Google Maps JS.

This template now:
- Loads the Google Maps JS API exactly once using async+defer with a callback.
- Includes the geometry library for polyline decoding.
- Fails gracefully with a visible warning if the API key is missing.
"""

from typing import List
import json
from pathlib import Path


TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Routes Map</title>
    <style>
      html, body, #map { height: 100%; margin: 0; padding: 0; }
      .warning { padding: 12px; margin: 12px; background: #fff4f4; color: #b00020; border: 1px solid #ffcccc; border-radius: 8px; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
    </style>
    <script>
      function init() {
        if (!window.google || !google.maps || !google.maps.geometry || !google.maps.geometry.encoding) {
          const el = document.getElementById('map');
          el.innerHTML = '<div class="warning">Google Maps geometry library not available. Check your API key and network.</div>';
          return;
        }
        const map = new google.maps.Map(document.getElementById('map'), {
          zoom: 11,
          center: {lat: 34.05, lng: -118.25}
        });
        const colors = ['#d32f2f','#1976d2','#388e3c','#fbc02d','#7b1fa2','#0097a7'];
        const polylines = __POLYLINES_JSON__;
        polylines.forEach((p, i) => {
          if (!p) return;
          const path = google.maps.geometry.encoding.decodePath(p);
          new google.maps.Polyline({
            path,
            geodesic: true,
            strokeColor: colors[i % colors.length],
            strokeOpacity: 0.9,
            strokeWeight: 4,
            map,
          });
        });
      }
      window.init = init;
    </script>
    __SCRIPT_TAG__
  </head>
  <body>
    <div id="map"></div>
    __WARNING_BLOCK__
  </body>
</html>
"""


def render_map_html(polylines: List[str], out_path: str, api_key: str) -> str:
  """Render a simple polyline map HTML.

  If api_key is falsy, no Google script is injected and a warning is shown.
  """
  if api_key:
    script_tag = (
      f"<script async defer src=\"https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=geometry,marker&callback=init\"></script>"
    )
    warning_block = ""
  else:
    script_tag = ""  # avoid InvalidKey warnings; show a friendly message instead
    warning_block = (
      "<div class=\"warning\">"
      "Missing Google Maps API key. Set GOOGLE_MAPS_API_KEY in your .env and regenerate this file."
      "</div>"
    )

  out = TEMPLATE
  out = out.replace("__POLYLINES_JSON__", json.dumps(polylines))
  out = out.replace("__SCRIPT_TAG__", script_tag)
  out = out.replace("__WARNING_BLOCK__", warning_block)
  path = Path(out_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(out, encoding='utf-8')
  return str(path)


# --- CSV-based map (example/day1) with AdvancedMarkerElement ---
DAY_CSV_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Truck Jobs Map - day1</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    html, body, #map { height: 100%; width: 100%; margin: 0; padding: 0; }
    .legend { position: absolute; top: 12px; left: 12px; background: white; padding: 8px 10px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); font-family: Arial, sans-serif; font-size: 13px; }
    .legend .row { display: flex; align-items: center; gap: 6px; margin: 2px 0; }
    .swatch { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
    .warning { position: absolute; top: 12px; right: 12px; background: #fff4f4; color: #b00020; border: 1px solid #ffcccc; border-radius: 6px; padding: 8px 10px; font-family: Arial, sans-serif; }
  </style>
  <script>
    const CSV_PATH = '../example_input/day1.csv';

    async function fetchCsvText(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error('Failed to load CSV: ' + res.status);
      return await res.text();
    }

    function parseCsv(text) {
      const lines = text.trim().split(/\\r?\\n/);
      const headers = lines[0].split(',');
      const rows = [];
      for (let i = 1; i < lines.length; i++) {
        const row = splitCsvLine(lines[i]);
        const obj = {};
        headers.forEach((h, idx) => obj[h.trim()] = row[idx] !== undefined ? row[idx].replace(/^"|"$/g, '') : '');
        rows.push(obj);
      }
      return rows;
    }

    function splitCsvLine(line) {
      const out = [];
      let cur = '';
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') {
          if (inQuotes && line[i+1] === '"') { cur += '"'; i++; }
          else inQuotes = !inQuotes;
        } else if (ch === ',' && !inQuotes) {
          out.push(cur);
          cur = '';
        } else {
          cur += ch;
        }
      }
      out.push(cur);
      return out;
    }

    async function geocodeAddress(geocoder, address) {
      return new Promise((resolve) => {
        geocoder.geocode({ address }, (results, status) => {
          if (status === 'OK' && results && results[0]) {
            const loc = results[0].geometry.location;
            resolve({ lat: loc.lat(), lng: loc.lng() });
          } else {
            console.warn('Geocode failed', status, address);
            resolve(null);
          }
        });
      });
    }

    function placeMarker(map, position, title, color) {
      // Use AdvancedMarkerElement exclusively to avoid Marker deprecation warnings
      if (!google.maps.marker || !google.maps.marker.AdvancedMarkerElement) {
        console.error('AdvancedMarkerElement is not available. Ensure the marker library is loaded.');
        return null;
      }
      const el = document.createElement('div');
      el.style.width = '12px';
      el.style.height = '12px';
      el.style.borderRadius = '50%';
      el.style.backgroundColor = color;
      el.style.border = '1px solid #333';
      return new google.maps.marker.AdvancedMarkerElement({
        map,
        position,
        title,
        content: el,
      });
    }

    function firstPickupIndex(rows) {
      return rows.findIndex(r => (r.action || '').toLowerCase() === 'pickup');
    }
    function lastDropIndex(rows) {
      const idxs = rows.map((r, i) => ((r.action || '').toLowerCase() === 'drop') ? i : -1).filter(i => i >= 0);
      return idxs.length ? idxs[idxs.length - 1] : rows.length - 1;
    }

    async function init() {
      if (!window.google || !google.maps) {
        const warn = document.getElementById('warn');
        if (warn) warn.textContent = 'Google Maps failed to load. Check your API key and network.';
        return;
      }

      const map = new google.maps.Map(document.getElementById('map'), {
        center: { lat: 34.2, lng: -118.4 },
        zoom: 11,
        mapTypeId: 'roadmap',
        mapId: undefined,
      });

      const geocoder = new google.maps.Geocoder();
      const csvText = await fetchCsvText(CSV_PATH);
      const rows = parseCsv(csvText);

      // Geocode all rows serially to keep quota safe for demo.
      const geocoded = [];
      for (const r of rows) {
        const coords = await geocodeAddress(geocoder, r.address);
        geocoded.push({ ...r, coords });
      }

      // Place markers and build waypoints
      const bounds = new google.maps.LatLngBounds();
      for (const r of geocoded) {
        if (!r.coords) continue;
        const isPickup = (r.action || '').toLowerCase() === 'pickup';
        const color = isPickup ? '#34a853' : '#ea4335';
        const title = `${r.location_name} (${isPickup ? 'pickup' : 'drop'})`;
        placeMarker(map, r.coords, title, color);
        bounds.extend(r.coords);
      }
      if (!bounds.isEmpty()) map.fitBounds(bounds);

      // Build routing: from first pickup -> all stops -> last drop
      const firstIdx = firstPickupIndex(geocoded);
      const lastIdx = lastDropIndex(geocoded);
      const valid = geocoded.filter(r => !!r.coords);
      if (valid.length < 2) return; // Need at least origin/destination

      const origin = (firstIdx >= 0 && geocoded[firstIdx].coords) ? geocoded[firstIdx].coords : valid[0].coords;
      const destination = (lastIdx >= 0 && geocoded[lastIdx].coords) ? geocoded[lastIdx].coords : valid[valid.length - 1].coords;

      const waypoints = geocoded
        .map((r, i) => ({ r, i }))
        .filter(x => x.r.coords && x.i !== firstIdx && x.i !== lastIdx)
        .map(x => ({ location: x.r.coords, stopover: true }));

      const ds = new google.maps.DirectionsService();
      const dr = new google.maps.DirectionsRenderer({ map, suppressMarkers: true, polylineOptions: { strokeColor: '#1a73e8', strokeWeight: 5 } });

      ds.route({
        origin,
        destination,
        waypoints,
        optimizeWaypoints: false,
        travelMode: google.maps.TravelMode.DRIVING,
      }, (result, status) => {
        if (status === 'OK' && result) {
          dr.setDirections(result);
        } else {
          console.warn('Directions failed', status, result);
        }
      });
    }
    window.init = init;
  </script>
  __SCRIPT_TAG__
</head>
<body>
  <div id="map"></div>
  <div class="legend">
    <div class="row"><span class="swatch" style="background:#34a853"></span> Pickup</div>
    <div class="row"><span class="swatch" style="background:#ea4335"></span> Drop</div>
    <div class="row"><span class="swatch" style="background:#1a73e8"></span> Route</div>
  </div>
  <div id="warn" class="warning" style="display:__WARN_DISPLAY__;">__WARN_TEXT__</div>
</body>
</html>
"""


def render_day_csv_map_html(out_path: str, api_key: str, csv_rel_path: str | None = None) -> str:
  """Render an HTML that fetches the example CSV and displays markers/routes.

  The script tag is injected with async+defer and a callback to init.
  Uses AdvancedMarkerElement for markers (requires marker library).
  """
  if api_key:
    script_tag = (
      f"<script async defer src=\"https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=geometry,marker&callback=init\"></script>"
    )
    warn_display = "none"
    warn_text = ""
  else:
    script_tag = ""
    warn_display = "block"
    warn_text = "Missing GOOGLE_MAPS_API_KEY. Set it in .env and regenerate this file."

  html = DAY_CSV_TEMPLATE.replace("../example_input/day1.csv", csv_rel_path or "../example_input/day1.csv")
  html = html.replace("__SCRIPT_TAG__", script_tag)
  html = html.replace("__WARN_DISPLAY__", warn_display)
  html = html.replace("__WARN_TEXT__", warn_text)
  path = Path(out_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(html, encoding="utf-8")
  return str(path)

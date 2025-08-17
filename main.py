import webbrowser
from pathlib import Path
import pandas as pd
import folium
from folium.plugins import HeatMap
from branca.colormap import LinearColormap

from signal_model import (
    load_csv_dataset,
    generate_prediction_grid,
    SignalStrengthModel,
    build_signal_map,
)

# ------------------------------------------------------------------
# Paths (work both in dev and in the PyInstaller bundle)
# ------------------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent
DATA_PATH     = BASE_DIR / "data" / "sample_measurements.csv"
OUTPUT_MAP    = BASE_DIR / "signal_map.html"

# ------------------------------------------------------------------
# Colour helpers
# ------------------------------------------------------------------
COLOUR_SCALE = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]
SIGNAL_VMIN, SIGNAL_VMAX = -110, -50   # fixed range for consistent colours

def make_colormap():
    return LinearColormap(
        colors=COLOUR_SCALE,
        vmin=SIGNAL_VMIN,
        vmax=SIGNAL_VMAX,
        caption="Signal strength (dBm)"
    )

def quality(signal: float) -> str:
    if signal > -60:      return "Excellent"
    elif signal > -75:    return "Good"
    elif signal > -90:    return "Fair"
    else:                 return "Poor"

# ------------------------------------------------------------------
# Browser GPS helper (unchanged)
# ------------------------------------------------------------------
def get_browser_location() -> tuple[float, float]:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import threading, urllib.parse

    class LocHandler(BaseHTTPRequestHandler):
        lat = lon = None
        def do_GET(self):
            if self.path.startswith("/loc"):
                qs = urllib.parse.parse_qs(self.path.split("?", 1)[1])
                LocHandler.lat = float(qs["lat"][0])
                LocHandler.lon = float(qs["lon"][0])
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Location captured! You can close this tab.")
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                html = """
                <!DOCTYPE html><html><head><meta charset="utf-8"/></head><body>
                <p>Allow location access when prompted…</p>
                <script>
                navigator.geolocation.getCurrentPosition(pos=>{
                    fetch(`/loc?lat=${pos.coords.latitude}&lon=${pos.coords.longitude}`);
                });
                </script></body></html>
                """
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))

    with HTTPServer(("127.0.0.1", 0), LocHandler) as httpd:
        port = httpd.socket.getsockname()[1]
        url = f"http://localhost:{port}/"
        print("Opening browser to fetch accurate location …")
        webbrowser.open(url)
        httpd.serve_forever()
    return LocHandler.lat, LocHandler.lon

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main(model_type: str = "rf") -> None:
    # 1. Load & train
    df = load_csv_dataset(str(DATA_PATH))
    print(f"Loaded {len(df)} samples")
    model = SignalStrengthModel(model_type=model_type)
    result = model.fit(df)
    print(f"Trained {result.model_type}  R²={result.r2:.2f}  MAE={result.mae:.2f}")

    # 2. Get accurate location
    lat, lon = get_browser_location()
    print(f"Browser location: ({lat}, {lon})")

    # 3. Predict grid
    grid = generate_prediction_grid(lat, lon, radius_m=2000, spacing_m=50)
    preds = model.predict(grid)
    grid["signal"] = preds

    lat_col = next(c for c in grid.columns if c.lower() in {"lat", "latitude"})
    lon_col = next(c for c in grid.columns if c.lower() in {"lon", "lng", "longitude"})
    sig_col = "signal"

    # 4. Build map
    cmap = make_colormap()
    fmap = folium.Map(location=[lat, lon], zoom_start=14)

    # 4a. Heat map (toggleable)
    heat_data = [[row[lat_col], row[lon_col], row[sig_col]] for _, row in grid.iterrows()]
    HeatMap(heat_data, min_opacity=0.4, radius=25, blur=15, name="Heatmap").add_to(fmap)

    # 4b. Coloured circles with pop-ups
    for _, r in grid.iterrows():
        sig = r[sig_col]
        folium.Circle(
            location=[r[lat_col], r[lon_col]],
            radius=25,
            color=None,
            fill=True,
            fill_color=cmap(sig),
            fill_opacity=0.7,
            popup=f"Signal: {sig:.1f} dBm<br>Quality: {quality(sig)}"
        ).add_to(fmap)

    # 4c. User marker & 2 km radius
    folium.Marker(
        location=[lat, lon],
        popup="You are here",
        icon=folium.Icon(color="black", icon="user", prefix="fa"),
    ).add_to(fmap)
    folium.Circle(
        location=[lat, lon],
        radius=2000,
        color="#0078ff",
        weight=2,
        fill=False,
        name="2 km radius"
    ).add_to(fmap)

    # 4d. Legend & layer control
    cmap.add_to(fmap)
    folium.LayerControl().add_to(fmap)

    # 5. Save & open
    fmap.save(str(OUTPUT_MAP))
    webbrowser.open(str(OUTPUT_MAP))


if __name__ == "__main__":
    main("rf")

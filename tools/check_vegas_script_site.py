"""Smoke check: the Vegas Script shows up on the live endpoint and in the
static Predict payload, for the latest season."""
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import _latest_season, app  # noqa: E402
from build_static_site import _env, build_predict  # noqa: E402

season = _latest_season()
client = TestClient(app)
for home, away in (("Florida", "Ole Miss"), ("Indiana", "Northwestern"),
                   ("Tennessee", "Texas"), ("Arkansas", "Tulsa")):
    html = client.post("/predict", data={
        "home_team": home, "away_team": away, "season": season}).text
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    alert = re.search(r"(THE SCRIPT IS IN\..{0,70})", text)
    print(f"LIVE {away} @ {home}:",
          re.search(r"([A-Z][\w ]+ by [\d.]+)", text).group(1).split(" AT ")[-1],
          "| script:", re.search(r"The Vegas Script (.+?) What", text).group(1).strip(),
          "|", alert.group(1) + "..." if alert else "no alert")

with tempfile.TemporaryDirectory() as out:
    build_predict(_env(), Path(out), season)
    page = (Path(out) / "predict.html").read_text(encoding="utf-8")
    payload = json.loads(re.search(
        r'id="model-data">(.*?)</script>', page, re.S).group(1))
v = payload["vegas"]
ours = (payload["calibrated"]["Florida"] - payload["calibrated"]["Ole Miss"]
        + payload["home_field"])
mkt = v["calibrated"]["Florida"] - v["calibrated"]["Ole Miss"] + v["home_field"]
print(f"STATIC: ours {ours:+.1f}, script "
      f"{v['ours_weight'] * ours + v['market_weight'] * mkt:+.1f} (Florida view)")

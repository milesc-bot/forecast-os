"""Capture console (TUI) screenshots for the README, headlessly.

Boots the demo terminal, walks each screen, saves an SVG via textual's
screenshot exporter, and rasterizes it to PNG with cairosvg.

Run:  python scripts/capture_console.py   (needs the [terminal] extra + cairosvg)
Writes docs/assets/console-*.png.
"""

from __future__ import annotations

import asyncio
import pathlib
import tempfile

import cairosvg

from forecast_os.terminal.app import ForecastOSApp
from forecast_os.terminal.workspace import Workspace

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "assets"
SIZE = (118, 34)

# textual's SVG names Fira Code (rarely installed for rasterizers); swap to a
# monospace font that ships block/sparkline + plot glyphs so cairosvg renders
# them instead of tofu. Only affects the PNG raster, not the app.
_RASTER_FONT = "Menlo, DejaVu Sans Mono, monospace"


async def capture() -> None:
    tmp = pathlib.Path(tempfile.mkdtemp())
    # a display-friendly home for the status line; demo mode never writes.
    workspace = Workspace(home=pathlib.Path("~/.forecast-os").expanduser())
    app = ForecastOSApp(workspace, demo=True)
    shots: list[tuple[str, str]] = []
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        def shot(name: str) -> None:
            svg = pathlib.Path(app.save_screenshot(f"{name}.svg", path=str(tmp)))
            shots.append((name, str(svg)))

        shot("console-dashboard")

        app.action_drill_down("west/alice")
        await app.workers.wait_for_complete()
        await pilot.pause()
        shot("console-forecast")

        await pilot.press("l")
        await app.workers.wait_for_complete()
        await pilot.pause()
        shot("console-leaderboard")

        await pilot.press("g")
        await app.workers.wait_for_complete()
        await pilot.pause()
        shot("console-governance")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, svg_path in shots:
        svg = pathlib.Path(svg_path).read_text().replace("Fira Code, monospace", _RASTER_FONT)
        png = OUT / f"{name}.png"
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png), scale=2.0)
        print(f"wrote {png.relative_to(OUT.parents[1])}")


if __name__ == "__main__":
    asyncio.run(capture())

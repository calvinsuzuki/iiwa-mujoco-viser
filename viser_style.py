"""Page styling for the Viser UI: colours and the CSS injected into the page.

Viser has no styling API, so the CSS goes in through gui.add_html (a <style> tag
in innerHTML applies to the whole page). To style one folder, add a marker
element to it with `marker(name)` and select the folder with

    div:has(> div > .name)     the folder's element container

Each GUI element sits in its own wrapper div directly inside that container, so
`> div:nth-child(i)` is the i-th element added to the folder. The `> div >` part
matters: a looser `:has(.name)` also matches every ancestor (the whole panel).
Mantine/Viser internals targeted here (class names, the "width: 6.6em" label
column) are from viser 1.1.1.
"""

import base64
import pathlib

GREEN, ORANGE, AMBER, RED = "#3ccf6e", "#ff8a3b", "#f5a524", "#e5484d"
KUKA_ORANGE = (255, 108, 10)
BG_CENTRE = (52, 58, 66)          # backdrop gradient centre (RGB)
BG_EDGE = (20, 22, 26)            # backdrop edge, fog and horizon colour
JOINT_COLORS = ["#e5484d", "#f5a524", "#e8e35a", "#3ccf6e", "#3fb6e8", "#8e7cff", "#e86bd1"]

FORMULA_DIR = pathlib.Path(__file__).resolve().parent / "formulas"
LABEL_COL = 'div[style*="width: 6.6"]'      # Viser's label column of an input row


def marker(name):
    """HTML for a CSS hook: add it (gui.add_html) inside the folder to style."""
    return f'<span class="{name}"></span>'


def folder(name):
    return f"div:has(> div > .{name})"


PANEL_CSS = f"""
<style>
/* Hide Viser's connection icon in the panel header; the pill shows the state. */
[data-dock-header] .tabler-icon-cloud-check,
[data-dock-header] .tabler-icon-player-pause,
[data-dock-header] .mantine-Loader-root,
[data-dock-header] > div:first-child > div[style*="width: 1.25em"],
[data-dock-header] > div:first-child > div[style*="width: 0.4em"] {{ display: none !important; }}

#iiwa-pill {{
  position: fixed; top: 14px; right: 16px; z-index: 1000; pointer-events: none;
  display: flex; align-items: center; gap: 7px; padding: 3px 11px 3px 9px;
  border-radius: 999px; background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.09); backdrop-filter: blur(6px);
  color: rgba(225, 231, 238, 0.7); font-size: 12px; font-weight: 500; letter-spacing: 0.02em;
}}
#iiwa-pill::before {{
  content: ""; width: 7px; height: 7px; border-radius: 50%;
  background: {GREEN}; box-shadow: 0 0 6px rgba(60, 207, 110, 0.7);
}}
#iiwa-pill::after {{ content: "Online"; }}
/* Viser removes its cloud-check icon whenever the websocket is down. */
body:not(:has(.tabler-icon-cloud-check)) #iiwa-pill::before {{
  background: {RED}; box-shadow: 0 0 6px rgba(229, 72, 77, 0.7);
}}
body:not(:has(.tabler-icon-cloud-check)) #iiwa-pill::after {{ content: "Offline"; }}

/* iiwa-pair: the folder's first two checkboxes share a row. */
{folder("iiwa-pair")} > div:nth-child(-n+2) {{ display: inline-block; width: 50%; vertical-align: top; }}
{folder("iiwa-pair")} > div:nth-child(-n+2) {LABEL_COL} {{ width: auto !important; white-space: nowrap; }}
{folder("iiwa-pair")} > div:nth-child(-n+2) .mantine-Flex-root {{ justify-content: flex-start; }}

/* iiwa-btn3: the folder's first three buttons share a row, the rest is full width. */
{folder("iiwa-btn3")} {{
  display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr)); column-gap: 6px;
  padding-inline: var(--mantine-spacing-xs);
}}
{folder("iiwa-btn3")} > div {{ grid-column: 1 / -1; }}
{folder("iiwa-btn3")} > div:nth-child(-n+3) {{ grid-column: auto; margin-inline: 0 !important; }}
{folder("iiwa-btn3")} > div:nth-child(-n+3) button {{ width: 100%; padding-inline: 4px; }}

/* iiwa-motion: elbow slider label on its own line, slider full width. */
{folder("iiwa-motion")} > div:first-child .mantine-Flex-root:has({LABEL_COL}) {{ flex-wrap: wrap; }}
{folder("iiwa-motion")} > div:first-child {LABEL_COL} {{ width: 100% !important; padding-bottom: 2px; }}

/* iiwa-joints: narrow bold label column, the slider takes the rest. */
{folder("iiwa-joints")} {LABEL_COL} {{ width: 2.1em !important; }}
{folder("iiwa-joints")} .mantine-Text-root {{
  font-size: 0.95em !important; font-weight: 600 !important;
  color: var(--mantine-color-text) !important; letter-spacing: 0 !important;
}}

.iiwa-note {{ font-size: 0.75em; color: var(--mantine-color-dimmed); line-height: 1.35; margin-top: 2px; }}
@keyframes iiwa-blink {{ 50% {{ opacity: 0.3; }} }}

/* Sliders: neutral track with a centre tick at 0 (all ranges are symmetric),
   round orange thumb, no marks or min/max labels (they collided with the track;
   the ranges are in the hints), compact value box. */
.mantine-Slider-root {{
  --slider-size: 4px !important; padding-top: 0.55em !important; padding-bottom: 0.55em !important;
}}
.mantine-Slider-track::before {{ background: rgba(255, 255, 255, 0.20) !important; border-radius: 99px; }}
.mantine-Slider-bar {{ background: transparent !important; }}
.mantine-Slider-track::after {{
  content: ""; position: absolute; left: 50%; top: -4px; bottom: -4px; width: 2px;
  transform: translateX(-50%); background: rgba(255, 255, 255, 0.28); border-radius: 1px;
}}
.mantine-Slider-thumb {{
  width: 14px !important; height: 14px !important; border-radius: 50% !important;
  background: rgb(255, 120, 30) !important; border: 2px solid #f4f5f7 !important;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.5); z-index: 2;
}}
.mantine-Slider-thumb:hover, .mantine-Slider-thumb:focus-visible {{
  box-shadow: 0 0 0 5px rgba(255, 120, 30, 0.25), 0 1px 4px rgba(0, 0, 0, 0.5);
}}
.mantine-Slider-mark, .mantine-Slider-markLabel {{ display: none !important; }}
.mantine-Slider-root + .mantine-NumberInput-root {{ width: 3.9rem !important; }}
.mantine-Slider-root + .mantine-NumberInput-root input {{
  text-align: right; font-variant-numeric: tabular-nums; font-size: 0.85rem !important;
  background: rgba(255, 255, 255, 0.04) !important; border-color: rgba(255, 255, 255, 0.10) !important;
}}

/* Plots: titles and legends only in the "Expand plot" modal; the small plots
   have a colour key under them instead. */
.uplot .u-title, .uplot .u-legend {{ display: none; }}
.mantine-Modal-body .uplot .u-title {{ display: block; font-size: 1em; padding-bottom: 6px; }}
.mantine-Modal-body .uplot .u-legend {{ display: block; font-size: 0.85em; margin: 4px 0 2px; }}
.mantine-Modal-body .uplot .u-legend .u-series:first-child {{ display: none; }}
.mantine-Modal-body .uplot .u-axis {{ font-size: 0.8em; }}
/* Status table (the panel holding the plots): fixed label column, so the values
   don't shift sideways as they change. */
.mantine-Paper-root:has(.uplot-container) table td:first-child {{ width: 9.5em; white-space: nowrap; }}
</style>
<div id="iiwa-pill"></div>
"""


def joint_limit_css(levels):
    """Slider tint per joint limit level (see IKController.limit_levels): amber
    thumb when near, red blinking slider at the limit. Goes into the iiwa-joints
    marker element, whose folder's children are: marker, then J1..J7."""
    rules = []
    for i, level in enumerate(levels):
        row = f"{folder('iiwa-joints')} > div:nth-child({i + 2})"
        if level == 1:
            rules.append(f"{row} .mantine-Slider-thumb {{ background: {AMBER} !important; }}")
        elif level == 2:
            rules.append(f"{row} .mantine-Slider-thumb {{ background: {RED} !important; }}"
                         f"{row} .mantine-Slider-track {{ animation: iiwa-blink 0.8s ease-in-out infinite; }}"
                         f"{row} .mantine-Slider-track::before {{ background: rgba(229,72,77,0.75) !important; }}")
    return f"<style>{''.join(rules)}</style>"


def ik_folder_html(methods):
    """Hover cards with each IK method's formulas (formulas/<key>.svg, made by
    formulas/make_formulas.py), plus the layout of the IK method folder.

    Add it to the IK method folder after its elements: the method buttons (1..n),
    then the two flexibility checkboxes (n+1, n+2). Card i shows while button i
    is hovered: `button-wrapper:hover ~ div` reaches this element, a later sibling.
    """
    n = len(methods)
    css, imgs = [], []
    for i, (key, label) in enumerate(methods):
        svg = FORMULA_DIR / f"{key}.svg"
        if not svg.exists():
            continue
        data = base64.b64encode(svg.read_bytes()).decode()
        imgs.append(f'<img class="iiwa-formula iiwa-formula-{i}" alt="{label} formulas" '
                    f'src="data:image/svg+xml;base64,{data}">')
        css.append(f"div:nth-child({i + 1}):has(> button:hover) ~ div .iiwa-formula-{i}"
                   " { opacity: 1; visibility: visible; }")
    flex = [f"{folder('iiwa-formula')} > div:nth-child({n + k})" for k in (1, 2)]
    return f"""<style>
.iiwa-formula {{
  position: fixed; top: 60px; right: 350px; z-index: 1000; width: 540px; height: auto;
  max-height: calc(100vh - 75px); object-fit: contain; box-sizing: border-box;
  padding: 10px 14px; border-radius: 10px; pointer-events: none;
  background: rgba(26, 29, 34, 0.96); border: 1px solid rgba(255, 255, 255, 0.10);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
  opacity: 0; visibility: hidden; transition: opacity 120ms ease, visibility 120ms;
}}
{chr(10).join(css)}
/* Reach flexibility | Restrain only Z-axis side by side, labels sized to their text. */
{flex[0]}, {flex[1]} {{ display: inline-block; vertical-align: top; }}
{flex[0]} {{ width: 44%; }}
{flex[1]} {{ width: 56%; }}
{flex[0]} {LABEL_COL}, {flex[1]} {LABEL_COL} {{ width: auto !important; white-space: nowrap; }}
{flex[0]} .mantine-Flex-root, {flex[1]} .mantine-Flex-root {{ justify-content: flex-start; }}
</style>{"".join(imgs)}"""

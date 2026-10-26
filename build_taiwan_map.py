#!/usr/bin/env python3
"""Convert the taiwan-atlas township TopoJSON into a light SVG-path bundle keyed
by the same township / county keys the vote explorer uses -> data/taiwan_map.json.

County outlines are DERIVED from the township geometry (topology-preserving
dissolve of the shared arcs), not taken from a separate file, so township fills
and county borders share exactly the same edges and line up perfectly.

The input TopoJSON is pulled automatically from the jsDelivr CDN (npm package
`taiwan-atlas`) and cached in ./build_cache/ . Just run:

    python build_taiwan_map.py

Set MAPSRC to a folder holding towns.topo.json to use your own local copy.
"""
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "build_cache")
SRC = os.environ.get("MAPSRC", CACHE)
BUNDLE = os.path.join(ROOT, "data", "president_partisan.json")

# npm taiwan-atlas @ 2021.9.20 -- already mercator-projected, coarse (10t) TopoJSON
CDN = "https://cdn.jsdelivr.net/npm/taiwan-atlas@2021.9.20"
INPUTS = {"towns.topo.json": CDN + "/towns-mercator-10t.json"}

MINDIST = 0.45   # arc point-decimation threshold, in projected units

# -- outlying islands: taiwan-atlas stacks all three in a left-hand gutter, none
#    at its true position. Penghu is moved to roughly its real geographic spot
#    (due west of Yunlin / Chiayi). Kinmen and Matsu can't sit at true scale, so
#    they go into equal-width dashed inset boxes at the TOP-left, US-map style
#    (like Alaska / Hawaii). Two far-flung islets (Wuqiu, Dongyin) are dropped
#    from the map so the insets frame the main clusters cleanly.
PEN_SHIFT = (7.0, -185.0)
OUTLIER_ISLETS = {"KIN-WUQIU", "LIE-DONGYIN"}
INSET_CONTENT_W = 52.0
INSET_PAD = 7.0
INSET_GAP = 18.0
INSET_TL = (22.0, 26.0)
_NUM = re.compile(r"(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)")


def ensure_inputs():
    if os.environ.get("MAPSRC"):
        missing = [f for f in INPUTS if not os.path.exists(os.path.join(SRC, f))]
        if missing:
            sys.exit(f"MAPSRC={SRC} is missing: {', '.join(missing)}")
        return
    os.makedirs(SRC, exist_ok=True)
    for fname, url in INPUTS.items():
        dst = os.path.join(SRC, fname)
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            continue
        print(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "build_taiwan_map"})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = r.read()
        with open(dst, "wb") as fh:
            fh.write(data)
        print(f"  -> {dst}  ({len(data)/1024:.0f} KB)")


def decode_arcs(topo):
    """dequantize every arc, then decimate it -- keeping the two endpoints so that
    arcs that share an endpoint still meet exactly."""
    sc = topo["transform"]["scale"]
    tr = topo["transform"]["translate"]
    md2 = MINDIST * MINDIST
    out = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x += dx
            y += dy
            pts.append((x * sc[0] + tr[0], y * sc[1] + tr[1]))
        if len(pts) > 2:
            keep = [pts[0]]
            for p in pts[1:-1]:
                ex = p[0] - keep[-1][0]
                ey = p[1] - keep[-1][1]
                if ex * ex + ey * ey >= md2:
                    keep.append(p)
            keep.append(pts[-1])
            pts = keep
        out.append(pts)
    return out


def arc_seg(a, darcs):
    return darcs[a] if a >= 0 else darcs[~a][::-1]


def ring_points(arc_idxs, darcs):
    pts = []
    for idx in arc_idxs:
        seg = arc_seg(idx, darcs)
        if pts and seg and abs(pts[-1][0] - seg[0][0]) < 1e-9 and abs(pts[-1][1] - seg[0][1]) < 1e-9:
            pts.extend(seg[1:])
        else:
            pts.extend(seg)
    return pts


def _emit(rings, prec=1):
    d = []
    minx = miny = 1e18
    maxx = maxy = -1e18
    for ring in rings:
        if len(ring) < 4:
            continue
        for i, (x, y) in enumerate(ring):
            minx = min(minx, x); miny = min(miny, y)
            maxx = max(maxx, x); maxy = max(maxy, y)
            d.append(("M" if i == 0 else "L") + f"{x:.{prec}f},{y:.{prec}f}")
        d.append("Z")
    if not d:
        return "", None
    return "".join(d), (round(minx, 1), round(miny, 1),
                        round(maxx - minx, 1), round(maxy - miny, 1))


def geom_to_path(geom, darcs):
    if geom["type"] == "Polygon":
        polys = [geom["arcs"]]
    elif geom["type"] == "MultiPolygon":
        polys = geom["arcs"]
    else:
        return "", None
    return _emit([ring_points(ring, darcs) for poly in polys for ring in poly])


def dissolve(geoms, darcs):
    """merge a set of township polygons into their shared outline by dropping
    every arc that is used by more than one of them (an interior border) and
    stitching what remains."""
    polygons = []
    for g in geoms:
        if g["type"] == "Polygon":
            polygons.extend([g["arcs"]])
        elif g["type"] == "MultiPolygon":
            polygons.extend(g["arcs"])
    usage = {}
    for poly in polygons:
        for ring in poly:
            for a in ring:
                k = a if a >= 0 else ~a
                usage[k] = usage.get(k, 0) + 1
    boundary = [a for poly in polygons for ring in poly for a in ring
                if usage[a if a >= 0 else ~a] == 1]
    if not boundary:
        return "", None

    def kp(p):
        return (round(p[0], 1), round(p[1], 1))

    starts = {}
    for i, a in enumerate(boundary):
        starts.setdefault(kp(arc_seg(a, darcs)[0]), []).append(i)
    pool = set(range(len(boundary)))
    rings = []
    while pool:
        i0 = pool.pop()
        ring = list(arc_seg(boundary[i0], darcs))
        start_k = kp(ring[0])
        guard = 0
        while kp(ring[-1]) != start_k and guard < 100000:
            guard += 1
            nxt = None
            for j in starts.get(kp(ring[-1]), []):
                if j in pool:
                    nxt = j
                    break
            if nxt is None:
                break
            pool.discard(nxt)
            ring.extend(arc_seg(boundary[nxt], darcs)[1:])
        rings.append(ring)
    return _emit(rings)


# ---- island repositioning (operates on the finished {d, bbox} shapes) ----
def _map_path(d, fn):
    return _NUM.sub(lambda m: "%s,%s" % fn(float(m.group(1)), float(m.group(2))), d)


def _group_keys(shapes, prefix):
    return [k for k in shapes if k == prefix or k.startswith(prefix + "-")]


def shift_group(shapes, prefix, dx, dy):
    for k in _group_keys(shapes, prefix):
        s = shapes[k]
        s["d"] = _map_path(s["d"], lambda x, y: (f"{x+dx:.1f}", f"{y+dy:.1f}"))
        b = s["bbox"]
        s["bbox"] = [round(b[0] + dx, 1), round(b[1] + dy, 1), b[2], b[3]]


def scale_group(shapes, prefix, s, tx, ty):
    for k in _group_keys(shapes, prefix):
        sh = shapes[k]
        sh["d"] = _map_path(sh["d"], lambda x, y: (f"{x*s+tx:.1f}", f"{y*s+ty:.1f}"))
        b = sh["bbox"]
        sh["bbox"] = [round(b[0] * s + tx, 1), round(b[1] * s + ty, 1),
                      round(b[2] * s, 1), round(b[3] * s, 1)]


def union_bbox(shapes, prefix):
    xs, ys, xe, ye = [], [], [], []
    for k in _group_keys(shapes, prefix):
        b = shapes[k]["bbox"]
        xs.append(b[0]); ys.append(b[1])
        xe.append(b[0] + b[2]); ye.append(b[1] + b[3])
    return [min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys)]


def main():
    ensure_inputs()
    bundle = json.load(open(BUNDLE, encoding="utf-8"))
    by_zh, by_en = {}, {}
    for t in bundle["townships"]:
        cen = bundle["counties"][t["county"]]["en"]
        by_zh[(cen, (t["zh"] or "").strip())] = t["key"]
        by_en[(cen, (t["en"] or "").strip().lower())] = t["key"]
    key_iso = {t["key"]: t["county"] for t in bundle["townships"]}

    topo = json.load(open(os.path.join(SRC, "towns.topo.json"), encoding="utf-8"))
    darcs = decode_arcs(topo)
    tobj = topo["objects"]["towns"]["geometries"]

    towns_out = {}
    geoms_by_iso = {}
    unmatched, dropped = [], []
    for g in tobj:
        p = g["properties"]
        cen = p.get("COUNTYENG", "").strip()
        key = by_zh.get((cen, p.get("TOWNNAME", "").strip())) \
            or by_en.get((cen, p.get("TOWNENG", "").strip().lower()))
        if not key:
            unmatched.append((cen, p.get("TOWNNAME"), p.get("TOWNENG")))
            continue
        if key in OUTLIER_ISLETS:
            dropped.append(key)
            continue
        d, bbox = geom_to_path(g, darcs)
        if not d:
            continue
        towns_out[key] = {"d": d, "bbox": bbox}
        geoms_by_iso.setdefault(key_iso[key], []).append(g)

    counties_out = {}
    for iso, geoms in geoms_by_iso.items():
        d, bbox = dissolve(geoms, darcs)
        if d:
            counties_out[iso] = {"d": d, "bbox": bbox}

    # Penghu -> true geographic location
    shift_group(towns_out, "PEN", *PEN_SHIFT)
    shift_group(counties_out, "PEN", *PEN_SHIFT)

    # Kinmen + Matsu -> equal-width dashed inset boxes, stacked at the top-left
    insets = []
    cur_y = INSET_TL[1]
    for iso, label in (("LIE", "Lienchiang (Matsu)"), ("KIN", "Kinmen")):
        b = union_bbox(towns_out, iso)
        s = INSET_CONTENT_W / b[2]
        tx = INSET_TL[0] - b[0] * s
        ty = cur_y - b[1] * s
        scale_group(towns_out, iso, s, tx, ty)
        scale_group(counties_out, iso, s, tx, ty)
        ch = b[3] * s
        box = [round(INSET_TL[0] - INSET_PAD, 1), round(cur_y - INSET_PAD, 1),
               round(INSET_CONTENT_W + 2 * INSET_PAD, 1), round(ch + 2 * INSET_PAD, 1)]
        counties_out[iso]["bbox"] = box
        insets.append({"label": label, "iso": iso, "box": box})
        cur_y = box[1] + box[3] + INSET_GAP + INSET_PAD
    pb = union_bbox(towns_out, "PEN")
    insets.append({"label": "Penghu", "box": None,
                   "at": [round(pb[0], 1), round(pb[1] - 7, 1)]})

    boxes = [t["bbox"] for t in towns_out.values()] + \
            [i["box"] for i in insets if i.get("box")]
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    xe = [b[0] + b[2] for b in boxes]
    ye = [b[1] + b[3] for b in boxes]
    pad = 5
    vb = [round(min(xs) - pad, 2), round(min(ys) - pad - 4, 2),
          round(max(xe) - min(xs) + 2 * pad, 2),
          round(max(ye) - min(ys) + 2 * pad + 4, 2)]

    out = {
        "meta": {"source": "taiwan-atlas (npm) township mercator-10t TopoJSON, re-keyed "
                 "to the vote-explorer keys; county outlines dissolved from the township "
                 "geometry; Penghu moved to its true location, Kinmen and Lienchiang "
                 "shown as top-left insets"},
        "viewBox": vb,
        "insets": insets,
        "towns": towns_out,
        "counties": counties_out,
    }
    dst = os.path.join(ROOT, "data", "taiwan_map.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False,
              separators=(",", ":"))
    print(f"wrote {dst}  ({os.path.getsize(dst)/1024:.0f} KB)")
    print(f"  {len(towns_out)} town shapes, {len(counties_out)} county shapes (dissolved)")
    print(f"  viewBox {vb}")
    if dropped:
        print(f"  dropped outlier islets from the map (kept in vote data): {sorted(dropped)}")
    if unmatched:
        print(f"  !! {len(unmatched)} unmatched town geometries:")
        for u in unmatched:
            print("     ", u)
    missing = (set(t["key"] for t in bundle["townships"]) - set(towns_out)) - OUTLIER_ISLETS
    if missing:
        print(f"  !! {len(missing)} bundle townships with no shape: {sorted(missing)}")


if __name__ == "__main__":
    main()

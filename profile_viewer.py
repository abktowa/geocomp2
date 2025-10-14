#!/usr/bin/env python3
"""
profile_viewer.py
Author: ChatGPT 5.0
Modified by: Christian Duncan
Class: CSC615, Fall 2025

NOTE: Written initially using ChatGPT, then edited. See bottom for the original queries used.

Display target query profiles overlaid with suggested route profiles.

Inputs:
  1) graph JSONL file (nodes with x,y,elev; directed edges with id,u,v,length_m,climb_m,slope)
  2) query file:
       first line: integer Q
       next Q lines: Cx Cy D d1 z1 d2 z2 ... (cumulative distances + relative elevations)
  3) suggested route file:
       one line per query: s t edgeId1 edgeId2 ... edgeIdM
       where s,t in [0,1] are fractional start/end positions along first/last edge.

Behavior:
  - Always include the explicit start point (0,0) for both target and suggested profiles.
  - Suggested profile is shifted vertically by --shift-pct * (range of target profile) for visibility.

Usage:
  python profile_viewer.py graph.jsonl sample_input.txt sample_output.txt --shift-pct 0.2 --show
"""
import json, math, argparse, sys
import matplotlib.pyplot as plt
import os

# ---------- Data structures ----------
class Node:
    __slots__ = ("id","x","y","elev")
    def __init__(self, nid, x, y, elev):
        self.id = nid; self.x = x; self.y = y; self.elev = elev

class Edge:
    __slots__ = ("id","u","v","length","climb","slope")
    def __init__(self, eid, u, v, length_m, climb_m, slope):
        self.id = eid; self.u = u; self.v = v
        self.length = float(length_m); self.climb = float(climb_m); self.slope = float(slope)

# ---------- I/O ----------
def load_graph_jsonl(path):
    """ 
    Loads in the graph - storing the nodes and edges in two separate dictionaries
    Although the ids are integers, storing them as strings anyway (since they 
    could be different)
    """
    nodes = {}
    edge_by_id = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            t = rec.get("type")
            if t == "node":
                n = Node(rec["id"], float(rec["x"]), float(rec["y"]), float(rec["elev"]))
                nodes[n.id] = n
            elif t == "edge":
                e = Edge(rec["id"], rec["u"], rec["v"], rec["length_m"], rec["climb_m"], rec["slope"])
                edge_by_id[e.id] = e
    return nodes, edge_by_id

def load_queries(path):
    """ 
    Loads the queries into memory for processing.
    """
    queries = []
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    q = int(lines[0])
    for i in range(1, 1+q):
        parts = list(map(float, lines[i].split()))
        if len(parts) < 5 or ((len(parts) - 3) % 2 != 0):
            raise ValueError(f"Bad query line {i}: expected Cx Cy D followed by d z pairs")
        Cx, Cy, D = parts[0], parts[1], parts[2]
        profile = []
        j = 3
        while j < len(parts):
            d = parts[j]; z = parts[j+1]
            profile.append((d, z))
            j += 2
        
        # Sorting the profiles but they should already be in order
        profile.sort(key=lambda t: t[0])

        # Ensuring an explicit (0,0) start
        if profile[0][0] > 0.0:
            # Add (0,0) to the start
            profile = [(0.0, 0.0)] + profile
        elif profile[0][0] == 0.0 and abs(profile[0][1]) > 1e-9:
            raise ValueError(f"Bad query line {i}: The start of the profile should have elevation 0 at distance 0 (either implicitly or explicitly)")
        elif profile[0][0] < 0.0:
            raise ValueError(f"Bad query line {i}: The profile distances (d) should all be positive.")
        # Store each query in a list as a dictionary of three items C,D, and profile.
        queries.append({"C": (Cx, Cy), "D": D, "profile": profile})
    return queries

def load_suggested_routes(path):
    """ 
    Loads the suggested routes. These would be the output of the main project to find routes.
    """
    routes = []
    with open(path, "r") as f:
        for i, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts: continue
            if len(parts) < 3:
                raise ValueError(f"Bad suggested route on line {i}: need s t and at least one edge id")
            s = float(parts[0]); t = float(parts[1])
            edge_ids = list(map(int, parts[2:]))
            routes.append({"s": s, "t": t, "edges": edge_ids})
    return routes

# ---------- Profile reconstruction ----------
def reconstruct_route_segments(nodes, edge_by_id, s_frac, t_frac, edge_ids):
    """
    Build segments for the route as a list of (len_m, elev_start, elev_end),
    applying fractional start s_frac on first edge and fractional end t_frac on last edge.
    Elevations are derived from node elevations (linear along each edge).
    """
    if not edge_ids:
        return []

    def e_end_elev(e):
        eu = nodes[e.u].elev
        ev = nodes[e.v].elev
        return eu, ev

    segs = []
    # Single-edge path
    if len(edge_ids) == 1:
        e0 = edge_by_id[edge_ids[0]]
        eu, ev = e_end_elev(e0)
        s_frac = max(0.0, min(1.0, s_frac))
        t_frac = max(0.0, min(1.0, t_frac))

        # If the end points are out of order, swap them.
        if t_frac < s_frac:
            s_frac, t_frac = t_frac, s_frac
        L = e0.length * (t_frac - s_frac)
        if L <= 0:
            return []
        zA = eu + s_frac*(ev - eu)
        zB = eu + t_frac*(ev - eu)
        segs.append((L, zA, zB))
        return segs

    # Cumulative distance
    distance = 0

    # First edge (from s_frac to end)
    e0 = edge_by_id[edge_ids[0]]
    eu0, ev0 = e_end_elev(e0)
    s_frac = max(0.0, min(1.0, s_frac))
    L0 = e0.length * (1.0 - s_frac)
    zA0 = eu0 + s_frac*(ev0 - eu0)
    zB0 = ev0
    segs.append((0, zA0))  # Everything starts at distance 0
    if L0 > 0:
        distance = L0
        segs.append((distance, zB0))

    # Middle full edges
    for eid in edge_ids[1:-1]:
        e = edge_by_id[eid]
        eu, ev = e_end_elev(e)
        distance = distance + e.length  # Cumulative distance
        segs.append((distance, ev))

    # Last edge (from start to t_frac)
    elast = edge_by_id[edge_ids[-1]]
    eul, evl = e_end_elev(elast)
    t_frac = max(0.0, min(1.0, t_frac))
    Ll = elast.length * t_frac
    if Ll > 0:
        distance = distance + Ll
        zBl = eul + t_frac*(evl - eul)
        segs.append((distance, zBl))

    # Shift everything so start elevation is 0
    start_elev = segs[0][1]
    segs = [(d, z - start_elev) for (d,z) in segs]

    return segs

# def sample_route_profile(segs, sample_distances):
#     """
#     Given segments [(L_i, z_i0, z_i1)], compute route elevation at each cumulative distance s.
#     Returns relative elevations (subtract start elevation).
#     """
#     # cumulative lengths
#     cum = [0.0]
#     for L, *_ in segs:
#         cum.append(cum[-1] + L)
#     total = cum[-1]
#     if total <= 0:
#         return [0.0 for _ in sample_distances]

#     z0 = segs[0][1]
#     out = []
#     for s in sample_distances:
#         s = max(0.0, min(total, s))
#         # find segment index j with cum[j] <= s <= cum[j+1]
#         j = 0
#         while j+1 < len(cum) and cum[j+1] < s - 1e-9:
#             j += 1
#         L, zA, zB = segs[j]
#         t = 0.0 if L <= 0 else (s - cum[j]) / L
#         z = zA + t*(zB - zA)
#         out.append(z - z0)
#     return out

# ---------- Plotting ----------
def plot_query(i, target, suggested,
               shift_pct, outdir, show=False):
    """
    Plot the given profile
    """

    # Separate the target and suggested profiles into distance and z
    target_distances = [d for d,_ in target]
    target_z  = [z for _,z in target]

    suggested_distances = [d for d,_ in suggested]
    suggested_z = [z for _,z in suggested]

    # Shift the suggested_z by the given amount (to prevent overlap)
    rng = (max(target_z) - min(target_z)) if target_z else 0.0
    offset = shift_pct * rng
    suggested_z = [z + offset for z in suggested_z]  # Shift it!

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(target_distances, target_z, label="Target profile", color="blue", linewidth=2)
    ax.plot(suggested_distances, suggested_z, label=f"Suggested profile (+{int(shift_pct*100)}%)", color="red", linewidth=2)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Relative elevation (m)")
    ax.set_title(f"Query {i+1}")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", linewidth=0.8)
    fig.tight_layout()
    outpath = f"{outdir}/query_{i+1}_overlay.png"
    fig.savefig(outpath, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return outpath

# ---------- Main ----------
def main():
    # Set up the command-line parameters
    ap = argparse.ArgumentParser(description="Overlay target query profiles with suggested route profiles.")
    ap.add_argument("graph_jsonl")
    ap.add_argument("query_file")
    ap.add_argument("suggested_file")
    ap.add_argument("--shift-pct", type=float, default=0.20, help="vertical offset as fraction of target profile range (default 0.20)")
    ap.add_argument("--show", action="store_true", help="display figures interactively")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    nodes, edge_by_id = load_graph_jsonl(args.graph_jsonl)
    queries = load_queries(args.query_file)
    routes  = load_suggested_routes(args.suggested_file)

    if len(routes) < len(queries):
        print(f"Warning: fewer routes ({len(routes)}) than queries ({len(queries)}); trailing queries will be skipped.", file=sys.stderr)

    os.makedirs(args.outdir, exist_ok=True)

    outputs = []
    for i, q in enumerate(queries):
        if i >= len(routes):
            break
        target = q["profile"]
        # if target_distances[0] != 0.0:
        #     target_distances = [0.0] + target_distances
        #     target_z  = [0.0] + target_z
        # elif abs(target_z[0]) > 1e-9:
        #     target_z[0] = 0.0

        r = routes[i]
        suggested = reconstruct_route_segments(nodes, edge_by_id, r["s"], r["t"], r["edges"])
        # suggested_z = sample_route_profile(segs, distances)

        outpng = plot_query(i, target, suggested, args.shift_pct, args.outdir, show=args.show)
        outputs.append(outpng)

    for p in outputs:
        print(p)

if __name__ == "__main__":
    main()


"""
Initial query:
Attached is the current assignment, the sample JSONL file with the map details, a sample query file (sample_input.txt) and a sample output file (sample_output.txt). What I would like to work on now is a program (in Python) that I can provide the students that displays the query profiles overlaid with the profile of the suggested route. The program will need three inputs: the graph (JSONL) file, the query file (for the query routes), and the suggested route file (for each of the suggested routes - the output of the program). I am designing it so the students can view their output and compare it with the profiles in question. This will require a few steps. First, the program has to read in the JSONL file (to get route and elevation data) and store in a graph G. It should then read in the suggested route file, match each route in there with the graph to determine the suggested route profiles. It should then read in the query file to get the query routes. Then it should display one graph per query. That graph should draw the query profile (in blue) and overlay it with the suggested profile (in red). Since both profiles start at (0,0), the suggested profile should be raised about 20% to aid in visualizing the two. The 20% should be controlled by an optional command line parameter. There should be adequate labels on the x-axis (distance) and y-axis (relative elevation) and title (indicating which query number).

Follow-up query:
The profiles look a little off. The route should start at 0,0 (which is implied). Can you adjust so that this starting point is included?
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Optional
import heapq

# DATA STRUCTURES

# Store coordinates/elevation and which edges leave node
@dataclass
class Node:
    id: int # unique node ID
    x: float # meters 
    y: float # meters 
    z: float # elevation
    out_edges: list[int] = field(default_factory=list)

# Geometry + elevation change to navigate search
@dataclass
class Edge:
    id: int
    u: int # from node id
    v: int # to node id
    length: float # meters along road
    climb: float # z_v - z_u (meters)
    slope: float # climb / length

# Entire network in one object, pass "graph" for fast look-up using adj with bound defintion
@dataclass
class Graph:
    nodes: dict[int, Node]
    edges: dict[int, Edge]
    adj: dict[int, list[int]] # u -> [edge ids]
    bounds: tuple[float, float, float, float]  # minx, miny, maxx, maxy

# Uniform grid over map
@dataclass
class SpatialIndex:
    cell: dict[tuple[int, int], list[int]] # (ix,iy) -> [edge ids]
    minx: float
    miny: float
    cs: float # cell size (meters)

# Start of path representing start at a node or start inside an edge
@dataclass
class StartPos:
    at_node: Optional[int]
    edge_id: Optional[int]
    si: float # start fraction along first edge when edge_id is not None

# Step in the route, record that we used this edge and how much of it
@dataclass
class PathStep:
    edge_id: int
    frac: float

# Full route candidate, concrete path to score and output
@dataclass
class CandidatePath:
    steps: list[PathStep] # ordered edges with fractions
    total_length: float
    si: float # fraction of the first edge used (start offset)
    ti: float # fraction of the last edge used (end offset)

# Tracking beam-search state, what beam/A* keeps in queue while searching
@dataclass(order=True)
class BeamState:
    priority: float
    dist: float = field(compare=False)
    node: int = field(compare=False)
    steps: list[PathStep] = field(compare=False, default_factory=list)

# GEOMETRY HELPERS

# Returns x/y/z for edge's start and end nodes
def _edge_endpoints(graph: Graph, e: Edge) -> tuple[tuple[float,float,float], tuple[float,float,float]]:
    nu = graph.nodes[e.u]
    nv = graph.nodes[e.v]
    return (nu.x, nu.y, nu.z), (nv.x, nv.y, nv.z)

# Return point/elevation along edge at fraction t, start inside an edge or sample elevation
def interpolate_edge_point(graph: Graph, e: Edge, t: float) -> tuple[float, float, float]:
    (x0,y0,z0), (x1,y1,z1) = _edge_endpoints(graph, e)
    return x0 + (x1-x0)*t, y0 + (y1-y0)*t, z0 + (z1-z0)*t

def point_segment_projection(px, py, ax, ay, bx, by) -> tuple[float, float, float]:
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx*vx + vy*vy
    if vv <= 0:
        return 0.0, ax, ay
    t = (vx*wx + vy*wy) / vv
    if t < 0:
        t = 0.0
    elif t > 1:
        t = 1.0
    qx, qy = ax + vx*t, ay + vy*t
    return t, qx, qy


def dist2(ax, ay, bx, by) -> float:
    dx, dy = ax - bx, ay - by
    return dx*dx + dy*dy

# LOAD GRAPH & SPATIAL INDEX

# Reads JSONL file, finds edges and node's, builds respective dict's, returns Graph object
def load_graph(jsonl_path: str) -> Graph:
    nodes: dict[int, Node] = {}
    edges: dict[int, Edge] = {}
    minx = miny = float('inf')
    maxx = maxy = float('-inf')

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Edge vs Node detection
            if 'u' in obj and 'v' in obj:
                e = Edge(
                    id=int(obj['id']), u=int(obj['u']), v=int(obj['v']),
                    length=float(obj.get('length_m') or obj.get('length') or 0.0),
                    climb=float(obj.get('climb_m') or obj.get('climb') or 0.0),
                    slope=float(obj.get('slope') or 0.0)
                )
                edges[e.id] = e
            elif 'x' in obj and 'y' in obj and ('elev' in obj or 'z' in obj):
                z = float(obj.get('elev', obj.get('z')))
                n = Node(id=int(obj['id']), x=float(obj['x']), y=float(obj['y']), z=z)
                nodes[n.id] = n
                minx = min(minx, n.x); maxx = max(maxx, n.x)
                miny = min(miny, n.y); maxy = max(maxy, n.y)
            else:
                # meta line; ignore
                continue

    adj: dict[int, list[int]] = {nid: [] for nid in nodes}
    for e in edges.values():
        adj.setdefault(e.u, []).append(e.id)
        nodes[e.u].out_edges.append(e.id)

    return Graph(nodes=nodes, edges=edges, adj=adj, bounds=(minx, miny, maxx, maxy))

# Layers grid over map, for each edge puts ID into all grid cells
def build_spatial_index(graph: Graph, cell_size: float = 200.0) -> SpatialIndex:
    cell: dict[tuple[int,int], list[int]] = {}
    minx, miny, maxx, maxy = graph.bounds
    cs = cell_size

    def cell_range(x0, y0, x1, y1):
        ix0 = int(math.floor((min(x0,x1) - minx) / cs))
        iy0 = int(math.floor((min(y0,y1) - miny) / cs))
        ix1 = int(math.floor((max(x0,x1) - minx) / cs))
        iy1 = int(math.floor((max(y0,y1) - miny) / cs))
        return ix0, iy0, ix1, iy1

    for e in graph.edges.values():
        (x0,y0,_), (x1,y1,_) = _edge_endpoints(graph, e)
        ix0, iy0, ix1, iy1 = cell_range(x0,y0,x1,y1)
        for ix in range(ix0, ix1+1):
            for iy in range(iy0, iy1+1):
                cell.setdefault((ix,iy), []).append(e.id)

    return SpatialIndex(cell=cell, minx=minx, miny=miny, cs=cs)

# Computes which gird cells overlap a search circle which we used to query index
def _cells_overlapping_circle(sidx: SpatialIndex, cx: float, cy: float, r: float):
    minx, miny, cs = sidx.minx, sidx.miny, sidx.cs
    ix0 = int(math.floor((cx - r - minx) / cs))
    iy0 = int(math.floor((cy - r - miny) / cs))
    ix1 = int(math.floor((cx + r - minx) / cs))
    iy1 = int(math.floor((cy + r - miny) / cs))
    for ix in range(ix0, ix1+1):
        for iy in range(iy0, iy1+1):
            yield (ix, iy)

# Looks for closest valid start within radius D, looks at all nodes and all edges in grid cells via SpatialIndex
def find_start_within_radius(graph: Graph, sidx: SpatialIndex, C: tuple[float,float], D: float) -> Optional[StartPos]:
    cx, cy = C
    D2 = D*D
    best: tuple[float, StartPos] | None = None

    # Nodes first (cheap)
    for nid, n in graph.nodes.items():
        d2 = dist2(cx, cy, n.x, n.y)
        if d2 <= D2:
            cand = StartPos(at_node=nid, edge_id=None, si=0.0)
            if best is None or d2 < best[0]:
                best = (d2, cand)

    # Edges via grid
    seen: set[int] = set()
    for cell in _cells_overlapping_circle(sidx, cx, cy, D):
        for eid in sidx.cell.get(cell, []):
            if eid in seen:
                continue
            seen.add(eid)
            e = graph.edges[eid]
            (x0,y0,_), (x1,y1,_) = _edge_endpoints(graph, e)
            t, qx, qy = point_segment_projection(cx, cy, x0, y0, x1, y1)
            d2 = dist2(cx, cy, qx, qy)
            if d2 <= D2:
                cand = StartPos(at_node=None, edge_id=eid, si=float(t))
                if best is None or d2 < best[0]:
                    best = (d2, cand)

    return None if best is None else best[1]

# PATH → ELEVATION PROFILE HELPERS

# Builds piecewise-linear profile for entire path
def path_polyline_sz(graph: Graph, steps: list[PathStep], si: float) -> list[tuple[float, float]]:
    """Return a piecewise-linear elevation profile as (s, z) points.
    s = cumulative distance from path start (meters). z = elevation (m).
    """
    sz: list[tuple[float,float]] = []
    s = 0.0
    first_edge = True

    for st in steps:
        e = graph.edges[st.edge_id]
        (x0,y0,z0), (x1,y1,z1) = _edge_endpoints(graph, e)
        f0 = si if first_edge and si > 0 else 0.0
        f1 = st.frac  # absolute end fraction on this edge

        z_start = z0 + (z1 - z0) * f0
        z_end   = z0 + (z1 - z0) * f1

        if first_edge:
            sz.append((s, z_start))

        seg_len = e.length * (f1 - f0)
        s += seg_len
        sz.append((s, z_end))

        first_edge = False
        si = 0.0

    return sz

# Returns elevation at each distance, easiest way for you to compare against target profile defined at specific distance
def sample_path_elevations(graph: Graph, steps: list[PathStep], si: float, query_ds: list[float]) -> list[Optional[float]]:
    """Sample elevation z at distances in query_ds from the path start.
    Returns None for any d beyond total path length.
    """
    # Build spans of absolute s along the path
    spans: list[tuple[float,float, Edge, float, float]] = []  # (s0, s1, edge, f0, f1)
    s = 0.0
    first_edge = True
    for st in steps:
        e = graph.edges[st.edge_id]
        f0 = si if first_edge and si > 0 else 0.0
        f1 = st.frac
        seg_len = e.length * (f1 - f0)
        spans.append((s, s+seg_len, e, f0, f1))
        s += seg_len
        first_edge = False
        si = 0.0

    total_len = s
    out: list[Optional[float]] = []
    for d in query_ds:
        if d < 0 or d > total_len + 1e-9:
            out.append(None)
            continue
        zval = None
        for s0, s1, e, f0, f1 in spans:
            if d <= s1 + 1e-9:
                w = 0.0 if s1 == s0 else (d - s0) / (s1 - s0)
                t = f0 + (f1 - f0) * w
                _, _, z = interpolate_edge_point(graph, e, t)
                zval = z
                break
        out.append(zval)
    return out

# SEARCH (LENGTH-CONSTRAINED BEAM)

# Returns allowed length window
def default_length_tolerance(L: float) -> float:
    return max(5.0, 0.05 * L)

def _push_beam(heap: list[BeamState], L: float, dist: float, node: int, steps: list[PathStep]):
    heapq.heappush(heap, BeamState(priority=abs(L - dist), dist=dist, node=node, steps=steps))

def make_profile_scorer(
        # INSERT HERE
)

def gather_paths_near_length(
    graph: Graph,
    start: StartPos,
    L: float,
    scorer=None,
    *,
    beam_width: int = 64,
    max_expansions: int = 200_000,
    length_tolerance: Optional[float] = None,
    collect_k: int = 10,
) -> list[CandidatePath]:
    """Search for up to k candidate paths whose length is within [L-ε, L+ε].

    If `scorer` is provided, it will be called as `scorer(steps)` and should
    return a numeric error — lower is better. Use a closure to capture other
    context (graph, si, target profile) inside your scorer.
    """
    eps = default_length_tolerance(L) if length_tolerance is None else length_tolerance

    # Seed beam from start
    beam: list[BeamState] = []
    si = 0.0
    if start.at_node is not None:
        _push_beam(beam, L, 0.0, start.at_node, [])
    else:
        e0 = graph.edges[start.edge_id]
        si = float(start.si)
        rem_len = e0.length * (1.0 - si)
        # Convention: PathStep.frac is absolute end fraction; for first edge, that's 1.0
        _push_beam(beam, L, rem_len, e0.v, [PathStep(edge_id=e0.id, frac=1.0)])

    best: list[tuple[float, float, CandidatePath]] = []  # (score_or_lenDiff, |len-L|, cand)
    expansions = 0

    while beam and expansions < max_expansions:
        # Limit beam width (drop worst while size > width)
        while len(beam) > beam_width:
            heapq.heappop(beam)
        st = heapq.heappop(beam)
        expansions += 1

        dist = st.dist
        if (L - eps) <= dist <= (L + eps):
            steps = st.steps
            s_val = float(scorer(steps)) if scorer else abs(dist - L)
            cand = CandidatePath(steps=list(steps), total_length=dist, si=si, ti=(steps[-1].frac if steps else 0.0))
            heapq.heappush(best, (s_val, abs(dist - L), cand))
            if len(best) > collect_k:
                heapq.heappop(best)
            # Keep exploring; there might be even better matches

        if dist > L + eps:
            continue

        # Expand
        u = st.node
        for eid in graph.adj.get(u, []):
            e = graph.edges[eid]
            ndist = dist + e.length
            nsteps = st.steps + [PathStep(edge_id=eid, frac=1.0)]

            if ndist > L + eps:
                # Try cutting inside this new edge to land inside the window, prefer near L
                need_to_L = max(0.0, L - dist)
                cut_end = min(1.0, need_to_L / e.length)  # absolute end fraction since f0=0 on new edge
                if cut_end > 0.0:
                    steps_cut = st.steps + [PathStep(edge_id=eid, frac=cut_end)]
                    clen = dist + e.length * cut_end
                    if (L - eps) <= clen <= (L + eps):
                        s_val = float(scorer(steps_cut)) if scorer else abs(clen - L)
                        cand = CandidatePath(steps=steps_cut, total_length=clen, si=si, ti=cut_end)
                        heapq.heappush(best, (s_val, abs(clen - L), cand))
                        if len(best) > collect_k:
                            heapq.heappop(best)
                continue

            _push_beam(beam, L, ndist, e.v, nsteps)

    return [t[2] for t in sorted(best, key=lambda r: (r[0], r[1]))]

# OUTPUT

def format_output(candidate: CandidatePath) -> str:
    if not candidate.steps:
        return "0.0 0.0"
    edge_ids = [str(s.edge_id) for s in candidate.steps]
    return f"{candidate.si:.6f} {candidate.ti:.6f} " + " ".join(edge_ids)

# CLI (simple demo)

def _cli():
    ap = argparse.ArgumentParser(description="Initial path gathering for Project Profile Finder")
    ap.add_argument("graph", help="Path to JSONL road graph")
    ap.add_argument("cx", type=float, help="Query center X (meters)")
    ap.add_argument("cy", type=float, help="Query center Y (meters)")
    ap.add_argument("D", type=float, help="Radius around center (meters)")
    ap.add_argument("L", type=float, help="Target path length (meters)")
    ap.add_argument("--beam", type=int, default=64, help="Beam width")
    ap.add_argument("--maxexp", type=int, default=200000, help="Max beam expansions")
    ap.add_argument("--cellsz", type=float, default=200.0, help="Spatial grid cell size (m)")
    ap.add_argument("--topk", type=int, default=3, help="Print top-k candidates")

    args = ap.parse_args()

    graph = load_graph(args.graph)
    sidx = build_spatial_index(graph, cell_size=args.cellsz)
    start = find_start_within_radius(graph, sidx, (args.cx, args.cy), args.D)
    if start is None:
        print("NO_START_IN_RADIUS")
        return

    # No scorer by default → choose by closest length to L
    # PLUG SORTER INTO SCORER
    cands = gather_paths_near_length(graph, start, args.L, scorer=None,
                                     beam_width=args.beam,
                                     max_expansions=args.maxexp,
                                     collect_k=args.topk)
    if not cands:
        print("NO_PATH_IN_WINDOW")
        return

    for cand in cands:
        print(format_output(cand))


if __name__ == "__main__":
    _cli()

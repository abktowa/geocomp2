#!/usr/bin/env python3
"""
Project Profile Finder
Aban Khan + Matthew Glennon
Single query:
  python solution3.py <graph>.jsonl Cx Cy D L --profile "0:0,50:2,100:1,..."

Text file:
  python solution3.py <graph>.jsonl --batch < input.txt
  # input.txt (either):
  #   A) Integer of queries on first line, then followed by lines of: cx cy D d1 z1 ... dn zn
  
  python solution3.py <graph>.jsonl --batch < input.txt > my_output.txt
  # my_output.txt
  # Output for algorithm in text file instead of command line
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Tuple
import heapq
from itertools import count
from bisect import bisect_left

_TIE = count()

# ========================= #
# DATA STRUCTURES           #
# ========================= # 

# Represents a point in the road graph, storing its coordinates and elevation.
@dataclass
class Node:
    id: int
    x: float
    y: float
    z: float
    out_edges: List[int] = field(default_factory=list)

# Represents a directed edge connecting two nodes, including its length and slope data.
@dataclass
class Edge:
    id: int
    u: int
    v: int
    length: float
    climb: float
    slope: float

# Stores the full road graph which includes all nodes, edges, adjacency list, and map boundaries.
@dataclass
class Graph:
    nodes: Dict[int, Node]
    edges: Dict[int, Edge]
    adj: Dict[int, List[int]]  # u -> [edge ids]
    bounds: Tuple[float, float, float, float]

# Provides a spatial grid index to rapidly find edges near a coordinate.
@dataclass
class SpatialIndex:
    cell: Dict[Tuple[int, int], List[int]]
    minx: float
    miny: float
    cs: float

# Defines a possible starting position. This can either be at a node or partway along an edge.
@dataclass
class StartPos:
    at_node: Optional[int]
    edge_id: Optional[int]
    si: float  # start fraction on first edge if edge_id != None

# Represents only one step along a path — an edge and how far along the path it travels.
@dataclass
class PathStep:
    edge_id: int
    frac: float  # end fraction [0..1] on this edge

# Represents an entire candidate path which includes all steps, total length, and edge fractions.
@dataclass
class CandidatePath:
    steps: List[PathStep]
    total_length: float
    si: float
    ti: float

# Holds the target profile distances and elevations, this defines the shape we intend to match.
@dataclass
class TargetProfile:
    ds: List[float]
    zs: List[float]
    @property
    def L(self) -> float:
        return self.ds[-1] if self.ds else 0.0

# ========================= #
# MATH HELPERS              #
# ========================= #

# Grabs the (x, y, z) coordinates of the two endpoint nodes for a given edge.
def _edge_endpoints(graph: Graph, e: Edge):
    nu = graph.nodes[e.u]
    nv = graph.nodes[e.v]
    return (nu.x, nu.y, nu.z), (nv.x, nv.y, nv.z)

# Estimates a point along an edge given a fraction t where 0 = start, 1 = end). This is used to sample locations or elevations partway along an edge.
def estimate_edge_point(graph: Graph, e: Edge, t: float):
    (x0,y0,z0), (x1,y1,z1) = _edge_endpoints(graph, e)
    return x0 + (x1-x0)*t, y0 + (y1-y0)*t, z0 + (z1-z0)*t

# Projects a point (px, py) onto the line segment from (ax, ay) to (bx, by). It returns the fractional distance t along the segment and the projected coordinates.
def point_segment_project(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx*vx + vy*vy
    if vv <= 0:
        return 0.0, ax, ay
    t = (vx*wx + vy*wy) / vv
    if t < 0: t = 0.0
    elif t > 1: t = 1.0
    qx, qy = ax + vx*t, ay + vy*t
    return t, qx, qy

# Calculates the distance squared between two 2D points which saves time by skipping the square root.
def dist2(ax, ay, bx, by):
    dx, dy = ax - bx, ay - by
    return dx*dx + dy*dy

# Loads the graph data from the provided JSONL file and builds Node and Edge objects. Also responsible for adjacency lists and bounding box information for the map.
def load_graph(jsonl_path: str) -> Graph:
    nodes: Dict[int, Node] = {}
    edges: Dict[int, Edge] = {}
    minx = miny = float('inf')
    maxx = maxy = float('-inf')
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if 'u' in obj and 'v' in obj:
                e = Edge(
                    id=int(obj['id']), u=int(obj['u']), v=int(obj['v']),
                    length=float(obj.get('length_m') or obj.get('length m') or obj.get('length') or 0.0),
                    climb=float(obj.get('climb_m') or obj.get('climb m') or obj.get('climb') or 0.0),
                    slope=float(obj.get('slope') or 0.0)
                )
                edges[e.id] = e
            elif 'x' in obj and 'y' in obj and ('elev' in obj or 'z' in obj):
                z = float(obj.get('elev', obj.get('z')))
                n = Node(id=int(obj['id']), x=float(obj['x']), y=float(obj['y']), z=z)
                nodes[n.id] = n
                minx = min(minx, n.x); maxx = max(maxx, n.x)
                miny = min(miny, n.y); maxy = max(maxy, n.y)
    adj: Dict[int, List[int]] = {nid: [] for nid in nodes}
    for e in edges.values():
        adj.setdefault(e.u, []).append(e.id)
        nodes[e.u].out_edges.append(e.id)
    for lst in adj.values():
        lst.sort()
    return Graph(nodes=nodes, edges=edges, adj=adj, bounds=(minx, miny, maxx, maxy))

# Creates a grid-based spatial index so nearby edges can be found efficiently. Each grid cell stores the IDs of edges that pass through it.
def build_spatial_index(graph: Graph, cell_size: float = 200.0) -> SpatialIndex:
    cell: Dict[Tuple[int,int], List[int]] = {}
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
    for k in cell.keys():
        cell[k].sort()
    return SpatialIndex(cell=cell, minx=minx, miny=miny, cs=cs)

# Yields all spatial grid cells that overlap a circular area of radius r around (cx, cy). This is used to find which edges lie within a search radius.
def _cells_overlapping_circle(sidx: SpatialIndex, cx: float, cy: float, r: float):
    minx, miny, cs = sidx.minx, sidx.miny, sidx.cs
    ix0 = int(math.floor((cx - r - minx) / cs))
    iy0 = int(math.floor((cy - r - miny) / cs))
    ix1 = int(math.floor((cx + r - minx) / cs))
    iy1 = int(math.floor((cy + r - miny) / cs))
    for ix in range(ix0, ix1+1):
        for iy in range(iy0, iy1+1):
            yield (ix, iy)

# ========================= #
# PROFILE / SCORER          #
# ========================= #

def sample_path_elevations(graph: Graph, steps: List[PathStep], si: float,
                           query_ds: List[float]) -> List[Optional[float]]:
    spans: List[Tuple[float,float, Edge, float, float]] = []
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
    out: List[Optional[float]] = []
    for d in query_ds:
        if d < 0 or d > total_len + 1e-9:
            out.append(None); continue
        zval = None
        for s0, s1, e, f0, f1 in spans:
            if d <= s1 + 1e-9:
                w = 0.0 if s1 == s0 else (d - s0) / (s1 - s0)
                t = f0 + (f1 - f0) * w
                _, _, z = estimate_edge_point(graph, e, t)
                zval = z
                break
        out.append(zval)
    return out

def make_profile_scorer(graph: Graph, si_first_edge: float, *,
                        target_ds: List[float], target_zs: List[float],
                        allow_vertical_offset: bool = True):
    td = list(target_ds)
    tz = list(target_zs)
    assert len(td) == len(tz) and len(td) >= 2, "profile needs ≥ 2 points"
    for i in range(1, len(td)):
        if not (td[i] > td[i-1]):
            raise ValueError("target distances must be strictly increasing")
    L_target = td[-1]

    def candidate_length(steps: List[PathStep]) -> float:
        s = 0.0
        first = True
        si = si_first_edge
        for st in steps:
            e = graph.edges[st.edge_id]
            f0 = si if first and si > 0 else 0.0
            f1 = st.frac
            s += e.length * max(0.0, f1 - f0)
            first = False
            si = 0.0
        return s

    def score(steps: List[PathStep]) -> float:
        zs = sample_path_elevations(graph, steps, si_first_edge, td)
        if zs[-1] is None:
            return float("inf")

        z0 = zs[0]
        rel = [z - z0 for z in zs]

        if allow_vertical_offset:
            z_off = sum((rel[i] - tz[i]) for i in range(len(tz))) / len(tz)
        else:
            z_off = 0.0

        eps = 1e-6
        total_w = 0.0
        acc_sum = 0.0
        for i in range(1, len(td)):
            w = td[i] - td[i-1]
            tgt = (tz[i] - tz[i-1])
            act = ((rel[i] - z_off) - (rel[i-1] - z_off))
            denom = max(eps, abs(tgt))
            seg_acc = 1.0 - abs(act - tgt) / denom
            seg_acc = min(1.0, max(0.0, seg_acc))
            acc_sum += seg_acc * w
            total_w += w
        climb_acc = acc_sum / max(eps, total_w)

        L_cand = candidate_length(steps)
        len_err = abs(L_cand - L_target) / max(1.0, L_target)
        len_acc = max(0.0, 1.0 - len_err)

        total_acc = climb_acc * len_acc
        loss = 1.0 - total_acc
        return loss

    return score

# ========================= #
# UTILITIES                 #
# ========================= #

# Builds a lookup that maps each edge to its reverse edge but only if it exists. If we have edges A > B and B > A this will link them together.
def build_reverse_edge_map(graph: Graph) -> Dict[int, Optional[int]]:
    by_pair: Dict[Tuple[int,int], int] = {}
    for e in graph.edges.values():
        by_pair[(e.u, e.v)] = e.id
    reverse_of: Dict[int, Optional[int]] = {}
    for e in graph.edges.values():
        reverse_of[e.id] = by_pair.get((e.v, e.u))
    return reverse_of

# Defines how much length variation is acceptable when matching a target path. For now we use 3% of total length (~5m) as the tolerance window.
def default_len_tolerance(L: float) -> float:
    return max(5.0, 0.03 * L)

# Reduces a dense target profile to a smaller number of evenly spaced samples. We keep around 150 points so elevation comparisons stay fast but accurate.
def _reduce_profile_for_guidance(ds: List[float], zs: List[float], max_pts: int = 150):
    n = len(ds)
    if n <= max_pts:
        return ds, zs
    idxs = [int(round(i*(n-1)/(max_pts-1))) for i in range(max_pts)]
    return [ds[i] for i in idxs], [zs[i] for i in idxs]

# Estimates how closely the current partial path matches the shape of the target elevation profile. This is used as a guidance heuristic during search. It will returns a loss between 0 (perfect match) and 1 (poor match).
#   Converts the partial path into cumulative distance spans.
#   Truncates the target profile to the same covered distance.
#   Samples elevations along the path using binary search.
#   Compares elevation changes segment-by-segment to compute an accuracy score.
#   Returns 1 - accuracy as the loss value.
def partial_path_check(graph: Graph, steps: List[PathStep], si_first_edge: float,
                         target_ds: List[float], target_zs: List[float]) -> float:
    if not steps or len(target_ds) < 2:
        return 0.5

    s = 0.0
    first = True
    si = si_first_edge
    spans = []
    span_ends = []
    for st in steps:
        e = graph.edges[st.edge_id]
        f0 = si if first and si > 0 else 0.0
        f1 = st.frac
        seg_len = e.length * max(0.0, f1 - f0)
        if seg_len > 0.0:
            s0 = s
            s1 = s + seg_len
            spans.append((s0, s1, e, f0, f1))
            span_ends.append(s1)
            s = s1
        first = False
        si = 0.0

    if s < 1e-6:
        return 0.5

    last_idx = bisect_left(target_ds, s + 1e-9)
    if last_idx < 1:
        return 0.5
    td = target_ds[:last_idx]
    tz = target_zs[:last_idx]

    zs = []
    for d in td:
        k = bisect_left(span_ends, d)
        if k >= len(spans):
            zs.append(None); continue
        s0, s1, e, f0, f1 = spans[k]
        w = 0.0 if s1 == s0 else (d - s0) / (s1 - s0)
        t = f0 + (f1 - f0) * max(0.0, min(1.0, w))
        _, _, z = estimate_edge_point(graph, e, t)
        zs.append(z)

    if not zs or zs[-1] is None:
        return 1.0

    z0 = zs[0]
    rel = [z - z0 for z in zs]
    z_off = sum((rel[i] - tz[i]) for i in range(len(tz))) / max(1, len(tz))

    eps = 1e-6
    total_w = 0.0
    acc_sum = 0.0
    for i in range(1, len(td)):
        w = td[i] - td[i-1]
        tgt = (tz[i] - tz[i-1])
        act = ((rel[i] - z_off) - (rel[i-1] - z_off))
        denom = max(eps, abs(tgt))
        seg_acc = 1.0 - abs(act - tgt) / denom
        seg_acc = min(1.0, max(0.0, seg_acc))
        acc_sum += seg_acc * w
        total_w += w
    climb_acc = acc_sum / max(eps, total_w)
    return 1.0 - climb_acc

# ========================= #
# SEARCH                    #
# ========================= #

# This is for one item in the priority queue of the beam search. Carries the partial path plus metadata used for pruning and guidance. Ordered by `priority`, then a stable `tie` counter.
@dataclass(order=True)
class OpenNode:
    priority: float
    tie: int = field(compare=True)
    dist: float = field(compare=False)
    node: int = field(compare=False)
    steps: List[PathStep] = field(compare=False, default_factory=list)
    last_eid: Optional[int] = field(compare=False, default=None)
    used_counts: Dict[int, int] = field(compare=False, default_factory=dict)
    scorer: Optional[Callable[[List[PathStep]], float]] = field(compare=False, default=None)
    ploss: float = field(compare=False, default=0.5)
    bin50: int = field(compare=False, default=0)

# Pushes a new OpenNode into the beam/heap with all required fields.
def _push(heap: List['OpenNode'], priority: float, dist: float, node: int,
          steps: List[PathStep], last_eid: Optional[int],
          used_counts: Dict[int,int], scorer, ploss: float, bin50: int) -> None:
    heapq.heappush(heap, OpenNode(priority=priority, tie=next(_TIE),
                                   dist=dist, node=node, steps=steps,
                                   last_eid=last_eid, used_counts=used_counts,
                                   scorer=scorer, ploss=ploss, bin50=bin50))

# Squared 2D distance (no sqrt) — faster for comparisons in spatial checks.
def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx; dy = ay - by
    return dx*dx + dy*dy

# Best-first search guided by length gap and partial-profile fit. Expands edges from candidate starts until a path within the length tolerance is found, then ranks by the full scorer. Uses: U-turn blocking, edge repeat caps, spatial leash, dominance by (node,last_edge,length bin), and cached partial-profile loss.
def search_best_path(
    graph: Graph,
    start: StartPos,
    L: float,
    scorer_fwd: Optional[Callable[[List[PathStep]], float]],
    scorer_back: Optional[Callable[[List[PathStep]], float]],
    reverse_of: Dict[int, Optional[int]],
    *,
    center_xy: Optional[Tuple[float,float]] = None,
    radius_D: Optional[float] = None,
    target_ds: List[float],
    target_zs: List[float],
) -> Optional[CandidatePath]:
    eps = default_len_tolerance(L)
    EDGE_REPEAT_CAP = 3
    MAX_EXPANSIONS = 2_000_000
    BEAM_CAP = 4096
    BIN = 25.0 

    leash_r2 = None
    if center_xy is not None and radius_D is not None:
        leash_r = radius_D + 0.75 * L
        leash_r2 = leash_r * leash_r

    def within_bounds(nid: int) -> bool:
        if leash_r2 is None: return True
        n = graph.nodes[nid]
        return _dist2(n.x, n.y, center_xy[0], center_xy[1]) <= leash_r2 + 1e-6

    best_gap: Dict[Tuple[int, Optional[int], int], float] = {}
    def consider_state(dist: float, node: int, last_eid: Optional[int]) -> bool:
        b = int(dist // BIN)
        key = (node, last_eid, b)
        gap = abs(L - dist)
        prev = best_gap.get(key)
        if prev is not None and gap >= prev - 1e-9:
            return False
        best_gap[key] = gap
        return True

    openh: List[OpenNode] = []
    si_used = 0.0

    if start.at_node is not None:
        if within_bounds(start.at_node) and consider_state(0.0, start.at_node, None):
            _push(openh, priority=L, dist=0.0, node=start.at_node,
                  steps=[], last_eid=None, used_counts={}, scorer=scorer_fwd,
                  ploss=0.5, bin50=0)
    else:
        e0 = graph.edges[start.edge_id]
        si_used = float(start.si)

        dist_fwd = e0.length * (1.0 - si_used)
        if within_bounds(e0.v) and consider_state(dist_fwd, e0.v, e0.id):
            steps_fwd = [PathStep(edge_id=e0.id, frac=1.0)]
            _push(openh, priority=abs(L - dist_fwd), dist=dist_fwd, node=e0.v,
                  steps=steps_fwd, last_eid=e0.id, used_counts={e0.id: 1}, scorer=scorer_fwd,
                  ploss=0.5, bin50=int(dist_fwd // 50.0))

        rev_eid = reverse_of.get(e0.id)
        if rev_eid is not None:
            rev_v = graph.edges[rev_eid].v
            dist_back = e0.length * si_used
            if within_bounds(rev_v) and consider_state(dist_back, rev_v, rev_eid):
                steps_back = [PathStep(edge_id=rev_eid, frac=si_used)]
                _push(openh, priority=abs(L - dist_back), dist=dist_back, node=rev_v,
                      steps=steps_back, last_eid=rev_eid, used_counts={rev_eid: 1}, scorer=scorer_back,
                      ploss=0.5, bin50=int(dist_back // 50.0))

    best: Optional[CandidatePath] = None
    best_loss: float = float("inf")
    expansions = 0

    while openh and expansions < MAX_EXPANSIONS:
        if len(openh) > BEAM_CAP:
            openh = heapq.nsmallest(BEAM_CAP, openh)
            heapq.heapify(openh)

        st = heapq.heappop(openh)
        expansions += 1

        if (L - eps) <= st.dist <= (L + eps):
            score_fn = st.scorer or scorer_fwd
            loss = float(score_fn(st.steps)) if score_fn else abs(st.dist - L)
            if loss < best_loss:
                best_loss = loss
                best = CandidatePath(steps=list(st.steps), total_length=st.dist,
                                     si=si_used, ti=(st.steps[-1].frac if st.steps else 0.0))
        if st.dist > L + eps:
            continue

        u = st.node
        for eid in graph.adj.get(u, []):
            if st.last_eid is not None:
                rev = reverse_of.get(st.last_eid)
                if rev is not None and eid == rev:
                    continue
            if st.used_counts.get(eid, 0) >= EDGE_REPEAT_CAP:
                continue

            e = graph.edges[eid]
            v = e.v
            if not within_bounds(v):
                continue

            ndist = st.dist + e.length

            if ndist > L + eps:
                need = max(0.0, L - st.dist)
                cut = min(1.0, need / e.length)
                if cut > 0.0:
                    cut_steps = st.steps + [PathStep(edge_id=eid, frac=cut)]
                    clen = st.dist + e.length * cut
                    if (L - eps) <= clen <= (L + eps):
                        score_fn = st.scorer or scorer_fwd
                        loss = float(score_fn(cut_steps)) if score_fn else abs(clen - L)
                        if loss < best_loss:
                            best_loss = loss
                            best = CandidatePath(steps=cut_steps, total_length=clen,
                                                 si=si_used, ti=cut)
                continue

            if not consider_state(ndist, v, eid):
                continue

            nsteps = st.steps + [PathStep(edge_id=eid, frac=1.0)]
            ncounts = dict(st.used_counts)
            ncounts[eid] = ncounts.get(eid, 0) + 1

            new_bin50 = int(ndist // 50.0)
            if new_bin50 != st.bin50:
                new_ploss = partial_path_check(graph, nsteps, 0.0, target_ds, target_zs)
            else:
                new_ploss = st.ploss

            gap = abs(L - ndist)          # meters
            pr = 0.6 * gap + 0.4 * (new_ploss * L)

            _push(openh, priority=pr, dist=ndist, node=v,
                  steps=nsteps, last_eid=eid, used_counts=ncounts,
                  scorer=st.scorer, ploss=new_ploss, bin50=new_bin50)

    return best

# ========================= #
# MULTI-START               #
# ========================= #

# Looks at plausible start positions within radius D of center C. Includes (a) nodes inside the circle and (b) points along edges whose closest point to C falls inside the circle. Results are ranked by distance to C, preferring edge-interior starts on exact ties, and capped by max_nodes/max_edges.
def _candidate_starts(graph: Graph, sidx: SpatialIndex, C: Tuple[float,float], D: float,
                      max_nodes: int = 256, max_edges: int = 256) -> List[StartPos]:
    cx, cy = C
    D2 = D * D
    cands: List[Tuple[float, int, StartPos]] = []

    # Nodes within D
    for nid, n in graph.nodes.items():
        d2 = dist2(cx, cy, n.x, n.y)
        if d2 <= D2:
            cands.append((d2, nid, StartPos(at_node=nid, edge_id=None, si=0.0)))

    # Edges via grid
    seen: set[int] = set()
    cells = list(_cells_overlapping_circle(sidx, cx, cy, D)); cells.sort()
    for cell in cells:
        for eid in sorted(sidx.cell.get(cell, [])):
            if eid in seen:
                continue
            seen.add(eid)
            e = graph.edges[eid]
            (x0,y0,_),(x1,y1,_) = _edge_endpoints(graph, e)
            t, qx, qy = point_segment_project(cx, cy, x0, y0, x1, y1)
            d2 = dist2(cx, cy, qx, qy)
            if d2 <= D2:
                cands.append((d2, eid, StartPos(at_node=None, edge_id=eid, si=float(t))))

    # Sort by: (distance to C, prefer edge-interior over node on ties, deterministic tie_id)
    def _rank(item):
        d2, tie_id, sp = item
        typ_rank = 0 if sp.edge_id is not None else 1  # prefer edge-interior starts on ties
        return (d2, typ_rank, tie_id)

    cands.sort(key=_rank)

    nodes_out, edges_out = [], []
    for _, _, sp in cands:
        if sp.edge_id is None:
            if len(nodes_out) < max_nodes:
                nodes_out.append(sp)
        else:
            if len(edges_out) < max_edges:
                edges_out.append(sp)
    return nodes_out + edges_out

# Tries the search from each candidate start and returns the overall best path. Uses the full scorer for final ranking, but passes a *reduced* (≤150 pt) profile to the search for fast partial-profile guidance.
def _best_from_starts(graph: Graph, sidx: SpatialIndex, starts: List[StartPos], L: float,
                      target_ds: List[float], target_zs: List[float],
                      allow_offset: bool, reverse_of: Dict[int, Optional[int]],
                      cx: float, cy: float, D: float) -> Optional[CandidatePath]:
    best_cand: Optional[CandidatePath] = None
    best_score: float = float("inf")

    # Precompute reduced guidance profile once
    g_ds, g_zs = _reduce_profile_for_guidance(target_ds, target_zs, max_pts=150)

    for start in starts:
        si = start.si if start.edge_id is not None else 0.0
        scorer_fwd = make_profile_scorer(graph, si,
                                         target_ds=target_ds, target_zs=target_zs,
                                         allow_vertical_offset=allow_offset)
        scorer_back = None
        if start.edge_id is not None:
            scorer_back = make_profile_scorer(graph, 0.0,
                                              target_ds=target_ds, target_zs=target_zs,
                                              allow_vertical_offset=allow_offset)

        cand = search_best_path(
            graph, start, L, scorer_fwd, scorer_back, reverse_of,
            center_xy=(cx, cy), radius_D=D,
            target_ds=g_ds, target_zs=g_zs
        )
        if not cand:
            continue
        loss = scorer_fwd(cand.steps)
        if loss < best_score:
            best_score = loss
            best_cand = cand
    return best_cand

# ========================= #
# OUTPUT                    #
# ========================= #

# Deals with outputting into CLI.
def _fmt_frac(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"

def format_output(candidate: CandidatePath) -> str:
    if not candidate or not candidate.steps:
        return "0 0"
    edge_ids = [str(s.edge_id) for s in candidate.steps]
    return f"{_fmt_frac(candidate.si)} {_fmt_frac(candidate.ti)} " + " ".join(edge_ids)

# ========================= #
# MAIN                      #
# ========================= #

def main():
    ap = argparse.ArgumentParser(description="Project Profile Finder")
    ap.add_argument("graph", help="Path to JSONL road graph")
    ap.add_argument("cx", type=float, nargs="?", help="Center X (meters)")
    ap.add_argument("cy", type=float, nargs="?", help="Center Y (meters)")
    ap.add_argument("D", type=float, nargs="?", help="Radius around center (meters)")
    ap.add_argument("L", type=float, nargs="?", help="Target path length (meters)")
    ap.add_argument("--profile", type=str, default="", help="Comma-separated d:z pairs, e.g. '0:0,50:2,100:1'")
    ap.add_argument("--nooffset", action="store_true", help="Disable vertical offset when scoring")
    ap.add_argument("--batch", action="store_true", help="Read queries from stdin (with or without leading count)")
    args = ap.parse_args()

    graph = load_graph(args.graph)
    sidx = build_spatial_index(graph)
    reverse_of = build_reverse_edge_map(graph)

    # ---------- Text file mode ----------
    if args.batch:
        import sys
        lines = [l.strip() for l in sys.stdin if l.strip()]
        if not lines:
            return
        try:
            q = int(lines[0])
            query_lines = lines[1:1+q]
        except ValueError:
            query_lines = lines

        for line in query_lines:
            parts = list(map(float, line.split()))
            cx, cy, D = parts[:3]
            rest = parts[3:]
            ds, zs = [], []
            for i in range(0, len(rest), 2):
                ds.append(rest[i]); zs.append(rest[i+1])
            if len(ds) < 2 or any(ds[i] <= ds[i-1] for i in range(1, len(ds))):
                print("BAD_PROFILE"); continue
            L = ds[-1]

            starts = _candidate_starts(graph, sidx, (cx, cy), D)
            if not starts:
                print("NO_START_IN_RADIUS"); continue

            best = _best_from_starts(
                graph, sidx, starts, L,
                target_ds=ds, target_zs=zs,
                allow_offset=(not args.nooffset),
                reverse_of=reverse_of,
                cx=cx, cy=cy, D=D
            )
            print("NO_PATH_IN_WINDOW" if not best else format_output(best))
        return

    # Single-query mode
    if args.cx is None or args.cy is None or args.D is None or args.L is None:
        ap.error("cx, cy, D, and L are required unless --batch is specified")

    profile: Optional[TargetProfile] = None
    if args.profile:
        ds: List[float] = []
        zs: List[float] = []
        for tok in args.profile.split(","):
            tok = tok.strip()
            if not tok:
                continue
            d_str, z_str = tok.split(":")
            ds.append(float(d_str)); zs.append(float(z_str))
        if len(ds) < 2 or any(ds[i] <= ds[i-1] for i in range(1, len(ds))):
            ap.error("Profile distances must be strictly increasing and have ≥2 points.")
        profile = TargetProfile(ds=ds, zs=zs)

    L = profile.L if profile else args.L

    starts = _candidate_starts(graph, sidx, (args.cx, args.cy), args.D)
    if not starts:
        print("NO_START_IN_RADIUS"); return

    if profile:
        best = _best_from_starts(
            graph, sidx, starts, L,
            target_ds=profile.ds, target_zs=profile.zs,
            allow_offset=(not args.nooffset),
            reverse_of=reverse_of,
            cx=args.cx, cy=args.cy, D=args.D
        )
        print("NO_PATH_IN_WINDOW" if not best else format_output(best))
    else:
        ds = [0.0, L]; zs = [0.0, 0.0]
        best = _best_from_starts(
            graph, sidx, starts, L,
            target_ds=ds, target_zs=zs,
            allow_offset=True,
            reverse_of=reverse_of,
            cx=args.cx, cy=args.cy, D=args.D
        )
        print("NO_PATH_IN_WINDOW" if not best else format_output(best))

if __name__ == "__main__":
    main()

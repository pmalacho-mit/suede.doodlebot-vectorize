//! A faithful emulation of the networkx behaviour the routing stage relies on:
//! insertion-ordered `MultiGraph` / `Graph`, `eulerize` (BFS shortest paths +
//! Edmonds' blossom `max_weight_matching`), Hierholzer's Eulerian circuit, and
//! CPython's set iteration order for the matching edge set.

use indexmap::{IndexMap, IndexSet};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Key {
    P(usize),
    I(i64),
}

type KeyDict = IndexMap<Key, f64>;

#[derive(Clone, Default, Debug)]
pub struct MultiGraph {
    pub adj: IndexMap<usize, IndexMap<usize, KeyDict>>,
}

impl MultiGraph {
    pub fn new() -> Self {
        MultiGraph { adj: IndexMap::new() }
    }
    pub fn add_node(&mut self, u: usize) {
        self.adj.entry(u).or_default();
    }
    pub fn nodes(&self) -> Vec<usize> {
        self.adj.keys().cloned().collect()
    }
    pub fn len(&self) -> usize {
        self.adj.len()
    }
    pub fn new_edge_key(&self, u: usize, v: usize) -> Key {
        let mut key: i64 = match self.adj.get(&u).and_then(|a| a.get(&v)) {
            Some(kd) => kd.len() as i64,
            None => 0,
        };
        if let Some(kd) = self.adj.get(&u).and_then(|a| a.get(&v)) {
            while kd.contains_key(&Key::I(key)) {
                key += 1;
            }
        }
        Key::I(key)
    }
    pub fn add_edge(&mut self, u: usize, v: usize, key: Option<Key>, weight: f64) -> Key {
        self.add_node(u);
        self.add_node(v);
        let key = key.unwrap_or_else(|| self.new_edge_key(u, v));
        // adj[u][v][key] = data ; adj[v][u] shares the same keydict
        {
            let au = self.adj.get_mut(&u).unwrap();
            let kd = au.entry(v).or_default();
            kd.insert(key, weight);
        }
        if u != v {
            let av = self.adj.get_mut(&v).unwrap();
            let kd = av.entry(u).or_default();
            kd.insert(key, weight);
        }
        key
    }
    pub fn remove_edge(&mut self, u: usize, v: usize, key: Key) {
        let empty = {
            let kd = self.adj.get_mut(&u).unwrap().get_mut(&v).unwrap();
            kd.shift_remove(&key);
            kd.is_empty()
        };
        if u != v {
            let kd = self.adj.get_mut(&v).unwrap().get_mut(&u).unwrap();
            kd.shift_remove(&key);
        }
        if empty {
            self.adj.get_mut(&u).unwrap().shift_remove(&v);
            if u != v {
                self.adj.get_mut(&v).unwrap().shift_remove(&u);
            }
        }
    }
    pub fn degree(&self, v: usize) -> usize {
        let nbrs = &self.adj[&v];
        let mut deg: usize = nbrs.values().map(|kd| kd.len()).sum();
        if let Some(kd) = nbrs.get(&v) {
            deg += kd.len();
        }
        deg
    }
    pub fn number_of_edges(&self) -> usize {
        self.edges_once().len()
    }
    /// Edges as networkx's `G.edges(keys=True, data=True)` yields them.
    pub fn edges_once(&self) -> Vec<(usize, usize, Key, f64)> {
        let mut out = Vec::new();
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (&n, nbrs) in &self.adj {
            for (&nbr, kd) in nbrs {
                if !seen.contains(&nbr) {
                    for (&k, &w) in kd {
                        out.push((n, nbr, k, w));
                    }
                }
            }
            seen.insert(n);
        }
        out
    }
    /// `G.copy()` / `nx.MultiGraph(G)`: rebuild by iterating the adjacency.
    pub fn copy(&self) -> MultiGraph {
        let mut g = MultiGraph::new();
        for &n in self.adj.keys() {
            g.add_node(n);
        }
        for (&u, nbrs) in &self.adj {
            for (&v, kd) in nbrs {
                for (&k, &w) in kd {
                    g.add_edge(u, v, Some(k), w);
                }
            }
        }
        g
    }
    /// `G.subgraph(nodes).copy()`
    pub fn subgraph_copy(&self, nodes: &std::collections::HashSet<usize>) -> MultiGraph {
        let mut g = MultiGraph::new();
        for &n in self.adj.keys() {
            if nodes.contains(&n) {
                g.add_node(n);
            }
        }
        for (&u, nbrs) in &self.adj {
            if !nodes.contains(&u) {
                continue;
            }
            for (&v, kd) in nbrs {
                if !nodes.contains(&v) {
                    continue;
                }
                for (&k, &w) in kd {
                    g.add_edge(u, v, Some(k), w);
                }
            }
        }
        g
    }
    /// `nx.compose(G, H)`
    pub fn compose(g: &MultiGraph, h: &MultiGraph) -> MultiGraph {
        let mut r = MultiGraph::new();
        for &n in g.adj.keys() {
            r.add_node(n);
        }
        for &n in h.adj.keys() {
            r.add_node(n);
        }
        for (u, v, k, w) in g.edges_once() {
            r.add_edge(u, v, Some(k), w);
        }
        for (u, v, k, w) in h.edges_once() {
            r.add_edge(u, v, Some(k), w);
        }
        r
    }
    fn plain_bfs(&self, source: usize) -> std::collections::HashSet<usize> {
        let mut seen = std::collections::HashSet::new();
        let mut next = vec![source];
        seen.insert(source);
        while !next.is_empty() {
            let this = std::mem::take(&mut next);
            for v in this {
                for &w in self.adj[&v].keys() {
                    if seen.insert(w) {
                        next.push(w);
                    }
                }
            }
        }
        seen
    }
    pub fn connected_components(&self) -> Vec<std::collections::HashSet<usize>> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for &v in self.adj.keys() {
            if !seen.contains(&v) {
                let c = self.plain_bfs(v);
                seen.extend(c.iter().cloned());
                out.push(c);
            }
        }
        out
    }
    pub fn is_connected(&self) -> bool {
        if self.adj.is_empty() {
            return false;
        }
        let first = *self.adj.keys().next().unwrap();
        self.plain_bfs(first).len() == self.adj.len()
    }
    pub fn is_eulerian(&self) -> bool {
        self.adj.keys().all(|&v| self.degree(v) % 2 == 0) && self.is_connected()
    }
    pub fn has_eulerian_path(&self, source: Option<usize>) -> bool {
        if self.is_eulerian() {
            return true;
        }
        if let Some(s) = source {
            if self.degree(s) % 2 != 1 {
                return false;
            }
        }
        let odd = self.adj.keys().filter(|&&v| self.degree(v) % 2 == 1).count();
        odd == 2 && self.is_connected()
    }
    /// `nx.shortest_path(G, source, target)` (bidirectional BFS, unweighted).
    pub fn shortest_path(&self, source: usize, target: usize) -> Vec<usize> {
        if source == target {
            return vec![source];
        }
        let mut pred: IndexMap<usize, Option<usize>> = IndexMap::new();
        let mut succ: IndexMap<usize, Option<usize>> = IndexMap::new();
        pred.insert(source, None);
        succ.insert(target, None);
        let mut forward = vec![source];
        let mut reverse = vec![target];
        let mut meet: Option<usize> = None;
        'outer: while !forward.is_empty() && !reverse.is_empty() {
            if forward.len() <= reverse.len() {
                let this = std::mem::take(&mut forward);
                for v in this {
                    for &w in self.adj[&v].keys() {
                        if !pred.contains_key(&w) {
                            forward.push(w);
                            pred.insert(w, Some(v));
                        }
                        if succ.contains_key(&w) {
                            meet = Some(w);
                            break 'outer;
                        }
                    }
                }
            } else {
                let this = std::mem::take(&mut reverse);
                for v in this {
                    for &w in self.adj[&v].keys() {
                        if !succ.contains_key(&w) {
                            succ.insert(w, Some(v));
                            reverse.push(w);
                        }
                        if pred.contains_key(&w) {
                            meet = Some(w);
                            break 'outer;
                        }
                    }
                }
            }
        }
        let w = meet.expect("no path");
        let mut path = vec![w];
        let mut cur = w;
        while let Some(Some(p)) = pred.get(&cur) {
            path.push(*p);
            cur = *p;
        }
        path.reverse();
        let mut cur = *path.last().unwrap();
        while let Some(Some(s)) = succ.get(&cur) {
            path.push(*s);
            cur = *s;
        }
        path
    }
    /// `_multigraph_eulerian_circuit` on a copy of self, starting at `source`.
    pub fn eulerian_circuit_edges(&self, source: usize) -> Vec<(usize, usize, Key)> {
        let mut g = self.copy();
        let mut out = Vec::new();
        let mut stack: Vec<(usize, Option<Key>)> = vec![(source, None)];
        let mut last_vertex: Option<usize> = None;
        let mut last_key: Option<Key> = None;
        while let Some(&(current_vertex, current_key)) = stack.last() {
            if g.degree(current_vertex) == 0 {
                if let Some(lv) = last_vertex {
                    out.push((lv, current_vertex, last_key.unwrap()));
                }
                last_vertex = Some(current_vertex);
                last_key = current_key;
                stack.pop();
            } else {
                let (next_vertex, next_key) = {
                    let nbrs = &g.adj[&current_vertex];
                    let (&nv, kd) = nbrs.iter().next().unwrap();
                    let (&k, _) = kd.iter().next().unwrap();
                    (nv, k)
                };
                stack.push((next_vertex, Some(next_key)));
                g.remove_edge(current_vertex, next_vertex, next_key);
            }
        }
        out
    }
    /// `list(nx.eulerian_circuit(G, source, keys=True))`
    pub fn eulerian_circuit(&self, source: usize) -> Vec<(usize, usize, Key)> {
        self.eulerian_circuit_edges(source)
    }
    /// `list(nx.eulerian_path(G, source, keys=True))` for undirected multigraphs.
    pub fn eulerian_path(&self, source: usize) -> Vec<(usize, usize, Key)> {
        let edges = self.eulerian_circuit_edges(source);
        let mut out: Vec<(usize, usize, Key)> = edges.into_iter().map(|(u, v, k)| (v, u, k)).collect();
        out.reverse();
        out
    }
    /// `nx.eulerize(G)`
    pub fn eulerize(&self) -> MultiGraph {
        let odd: Vec<usize> = self.adj.keys().cloned().filter(|&v| self.degree(v) % 2 == 1).collect();
        let mut g = self.copy();
        if odd.is_empty() {
            return g;
        }
        // (first, second, path) for combinations
        let mut pairs: Vec<(usize, usize, Vec<usize>)> = Vec::new();
        for i in 0..odd.len() {
            for j in i + 1..odd.len() {
                let path = g.shortest_path(odd[i], odd[j]);
                pairs.push((odd[i], odd[j], path));
            }
        }
        let upper = g.len() as i64 + 1;
        let mut gp = Graph::new();
        for (first, second, path) in &pairs {
            if first != second {
                gp.add_edge(*second, *first, upper - path.len() as i64, path.clone());
            }
        }
        let matching = max_weight_matching(&gp);
        // best_matching = nx.Graph(list(matching)); iterate its edges
        let ordered = pyset_order_pairs(&matching);
        let mut bm = Graph::new();
        for &(u, v) in &ordered {
            bm.add_edge(u, v, 0, vec![]);
        }
        for (m, n) in bm.edges() {
            let path = gp.path(m, n).clone();
            for k in 0..path.len() - 1 {
                g.add_edge(path[k], path[k + 1], None, 1.0);
            }
        }
        g
    }
}

/// Simple graph with (int weight, path) attributes, insertion ordered.
#[derive(Clone, Default, Debug)]
pub struct Graph {
    pub adj: IndexMap<usize, IndexMap<usize, (i64, Vec<usize>)>>,
}

impl Graph {
    pub fn new() -> Self {
        Graph { adj: IndexMap::new() }
    }
    pub fn add_edge(&mut self, u: usize, v: usize, weight: i64, path: Vec<usize>) {
        self.adj.entry(u).or_default();
        self.adj.entry(v).or_default();
        self.adj.get_mut(&u).unwrap().insert(v, (weight, path.clone()));
        self.adj.get_mut(&v).unwrap().insert(u, (weight, path));
    }
    pub fn nodes(&self) -> Vec<usize> {
        self.adj.keys().cloned().collect()
    }
    pub fn neighbors(&self, v: usize) -> Vec<usize> {
        self.adj[&v].keys().cloned().collect()
    }
    pub fn weight(&self, u: usize, v: usize) -> i64 {
        self.adj[&u][&v].0
    }
    pub fn path(&self, u: usize, v: usize) -> &Vec<usize> {
        &self.adj[&u][&v].1
    }
    /// `G.edges()` iteration order.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (&n, nbrs) in &self.adj {
            for &nbr in nbrs.keys() {
                if !seen.contains(&nbr) {
                    out.push((n, nbr));
                }
            }
            seen.insert(n);
        }
        out
    }
    pub fn edges_with_weight(&self) -> Vec<(usize, usize, i64)> {
        self.edges().into_iter().map(|(u, v)| (u, v, self.weight(u, v))).collect()
    }
}

// ---------------------------------------------------------------------------
// CPython set-of-tuples iteration order emulation

fn py_int_hash(v: i64) -> u64 {
    // CPython hash of ints: value mod (2^61 - 1), with -1 -> -2.
    let m: i128 = (1i128 << 61) - 1;
    let mut h: i128 = if v >= 0 { (v as i128) % m } else { -(((-(v as i128)) % m) as i128) };
    if h == -1 {
        h = -2;
    }
    h as i64 as u64
}

fn py_tuple2_hash(a: i64, b: i64) -> u64 {
    const XXPRIME_1: u64 = 11400714785074694791;
    const XXPRIME_2: u64 = 14029467366897019727;
    const XXPRIME_5: u64 = 2870177450012600261;
    let mut acc: u64 = XXPRIME_5;
    for lane in [py_int_hash(a), py_int_hash(b)] {
        acc = acc.wrapping_add(lane.wrapping_mul(XXPRIME_2));
        acc = acc.rotate_left(31);
        acc = acc.wrapping_mul(XXPRIME_1);
    }
    acc = acc.wrapping_add(2u64 ^ (XXPRIME_5 ^ 3527539u64));
    if acc == u64::MAX {
        return 1546275796u64;
    }
    acc
}

/// Order in which CPython iterates a `set` built by inserting the given
/// (distinct) pairs in order.
pub fn pyset_order_pairs(pairs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    const LINEAR_PROBES: usize = 9;
    const PERTURB_SHIFT: u32 = 5;
    let mut table: Vec<Option<(u64, (usize, usize))>> = vec![None; 8];
    let mut used = 0usize;
    fn insert_clean(table: &mut Vec<Option<(u64, (usize, usize))>>, hash: u64, key: (usize, usize)) {
        let mask = table.len() - 1;
        let mut perturb = hash;
        let mut i = (hash as usize) & mask;
        loop {
            if table[i].is_none() {
                table[i] = Some((hash, key));
                return;
            }
            if i + LINEAR_PROBES <= mask {
                for j in 1..=LINEAR_PROBES {
                    if table[i + j].is_none() {
                        table[i + j] = Some((hash, key));
                        return;
                    }
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb as usize)) & mask;
        }
    }
    for &(a, b) in pairs {
        let hash = py_tuple2_hash(a as i64, b as i64);
        // set_add_entry
        let mask = table.len() - 1;
        let mut perturb = hash;
        let mut i = (hash as usize) & mask;
        'find: loop {
            let probes = if i + LINEAR_PROBES <= mask { LINEAR_PROBES } else { 0 };
            let mut entry = i;
            for _ in 0..=probes {
                match &table[entry] {
                    None => {
                        table[entry] = Some((hash, (a, b)));
                        used += 1;
                        break 'find;
                    }
                    Some((h, k)) => {
                        if *h == hash && *k == (a, b) {
                            break 'find; // already present
                        }
                    }
                }
                entry += 1;
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb as usize)) & mask;
        }
        // resize check: fill*5 >= mask*3  (fill == used, no dummies)
        if used * 5 >= mask * 3 {
            let minused = if used > 50000 { used * 2 } else { used * 4 };
            let mut newsize = 8usize;
            while newsize <= minused {
                newsize <<= 1;
            }
            let old = std::mem::replace(&mut table, vec![None; newsize]);
            for e in old.into_iter().flatten() {
                insert_clean(&mut table, e.0, e.1);
            }
        }
    }
    table.into_iter().flatten().map(|(_, k)| k).collect()
}

// ---------------------------------------------------------------------------
// max_weight_matching (Galil / van Rantwijk) with integer weights,
// maxcardinality=False, reproducing networkx's iteration orders.

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum NodeRef {
    V(usize),
    B(usize),
}

struct Blossom {
    childs: Vec<NodeRef>,
    edges: Vec<(usize, usize)>,
    mybestedges: Option<Vec<(usize, usize)>>,
}

struct Matcher<'a> {
    g: &'a Graph,
    gnodes: Vec<usize>,
    blossoms: Vec<Blossom>,
    mate: IndexMap<usize, usize>,
    label: IndexMap<NodeRef, i32>,
    labeledge: IndexMap<NodeRef, Option<(usize, usize)>>,
    inblossom: IndexMap<usize, NodeRef>,
    blossomparent: IndexMap<NodeRef, Option<usize>>,
    blossombase: IndexMap<NodeRef, usize>,
    bestedge: IndexMap<NodeRef, Option<(usize, usize)>>,
    dualvar: IndexMap<usize, i64>,
    blossomdual: IndexMap<usize, i64>,
    allowedge: std::collections::HashSet<(usize, usize)>,
    queue: Vec<usize>,
}

impl<'a> Matcher<'a> {
    fn leaves(&self, b: usize) -> Vec<usize> {
        let mut stack: Vec<NodeRef> = self.blossoms[b].childs.clone();
        let mut out = Vec::new();
        while let Some(t) = stack.pop() {
            match t {
                NodeRef::B(bb) => stack.extend(self.blossoms[bb].childs.iter().cloned()),
                NodeRef::V(v) => out.push(v),
            }
        }
        out
    }
    fn slack(&self, v: usize, w: usize) -> i64 {
        self.dualvar[&v] + self.dualvar[&w] - 2 * self.g.weight(v, w)
    }
    fn lbl(&self, n: NodeRef) -> Option<i32> {
        self.label.get(&n).cloned()
    }
    fn assign_label(&mut self, w: usize, t: i32, v: Option<usize>) {
        let b = self.inblossom[&w];
        self.label.insert(NodeRef::V(w), t);
        self.label.insert(b, t);
        let le = v.map(|vv| (vv, w));
        self.labeledge.insert(NodeRef::V(w), le);
        self.labeledge.insert(b, le);
        self.bestedge.insert(NodeRef::V(w), None);
        self.bestedge.insert(b, None);
        if t == 1 {
            match b {
                NodeRef::B(bb) => {
                    let lv = self.leaves(bb);
                    self.queue.extend(lv);
                }
                NodeRef::V(vv) => self.queue.push(vv),
            }
        } else if t == 2 {
            let base = self.blossombase[&b];
            let mb = self.mate[&base];
            self.assign_label(mb, 1, Some(base));
        }
    }
    fn scan_blossom(&mut self, v0: usize, w0: usize) -> Option<usize> {
        let mut path: Vec<NodeRef> = Vec::new();
        let mut base: Option<usize> = None;
        let mut v: Option<usize> = Some(v0);
        let mut w: Option<usize> = Some(w0);
        while let Some(vv) = v {
            let b = self.inblossom[&vv];
            if self.label[&b] & 4 != 0 {
                base = Some(self.blossombase[&b]);
                break;
            }
            path.push(b);
            self.label.insert(b, 5);
            if self.labeledge[&b].is_none() {
                v = None;
            } else {
                let lv = self.labeledge[&b].unwrap().0;
                let b2 = self.inblossom[&lv];
                v = Some(self.labeledge[&b2].unwrap().0);
            }
            if w.is_some() {
                std::mem::swap(&mut v, &mut w);
            }
        }
        for b in path {
            self.label.insert(b, 1);
        }
        base
    }
    fn add_blossom(&mut self, base: usize, v0: usize, w0: usize) {
        let bb = self.inblossom[&base];
        let mut bv = self.inblossom[&v0];
        let mut bw = self.inblossom[&w0];
        let bidx = self.blossoms.len();
        self.blossoms.push(Blossom { childs: vec![], edges: vec![(v0, w0)], mybestedges: None });
        let b = NodeRef::B(bidx);
        self.blossombase.insert(b, base);
        self.blossomparent.insert(b, None);
        self.blossomparent.insert(bb, Some(bidx));
        let mut path: Vec<NodeRef> = Vec::new();
        let mut edgs: Vec<(usize, usize)> = vec![(v0, w0)];
        let mut v = v0;
        while bv != bb {
            self.blossomparent.insert(bv, Some(bidx));
            path.push(bv);
            edgs.push(self.labeledge[&bv].unwrap());
            v = self.labeledge[&bv].unwrap().0;
            bv = self.inblossom[&v];
        }
        let _ = v;
        path.push(bb);
        path.reverse();
        edgs.reverse();
        let mut w = w0;
        while bw != bb {
            self.blossomparent.insert(bw, Some(bidx));
            path.push(bw);
            let le = self.labeledge[&bw].unwrap();
            edgs.push((le.1, le.0));
            w = le.0;
            bw = self.inblossom[&w];
        }
        let _ = w;
        self.label.insert(b, 1);
        let lbb = self.labeledge[&bb];
        self.labeledge.insert(b, lbb);
        self.blossomdual.insert(bidx, 0);
        self.blossoms[bidx].childs = path.clone();
        self.blossoms[bidx].edges = edgs;
        for lv in self.leaves(bidx) {
            if self.label[&self.inblossom[&lv]] == 2 {
                self.queue.push(lv);
            }
            self.inblossom.insert(lv, b);
        }
        let mut bestedgeto: IndexMap<NodeRef, (usize, usize)> = IndexMap::new();
        for &bv in &path {
            let nblist: Vec<(usize, usize)> = match bv {
                NodeRef::B(bvi) => {
                    if let Some(mbe) = self.blossoms[bvi].mybestedges.take() {
                        mbe
                    } else {
                        let mut l = Vec::new();
                        for lv in self.leaves(bvi) {
                            for w in self.g.neighbors(lv) {
                                if lv != w {
                                    l.push((lv, w));
                                }
                            }
                        }
                        l
                    }
                }
                NodeRef::V(vv) => self.g.neighbors(vv).into_iter().filter(|&w| vv != w).map(|w| (vv, w)).collect(),
            };
            for k in nblist {
                let (mut i, mut j) = k;
                if self.inblossom[&j] == b {
                    std::mem::swap(&mut i, &mut j);
                }
                let bj = self.inblossom[&j];
                if bj != b && self.lbl(bj) == Some(1) {
                    let better = match bestedgeto.get(&bj) {
                        None => true,
                        Some(&(a, c)) => self.slack(i, j) < self.slack(a, c),
                    };
                    if better {
                        bestedgeto.insert(bj, k);
                    }
                }
            }
            self.bestedge.insert(bv, None);
        }
        let mbe: Vec<(usize, usize)> = bestedgeto.values().cloned().collect();
        let mut mybestedge: Option<(usize, usize)> = None;
        let mut mybestslack = 0i64;
        for &k in &mbe {
            let ks = self.slack(k.0, k.1);
            if mybestedge.is_none() || ks < mybestslack {
                mybestedge = Some(k);
                mybestslack = ks;
            }
        }
        self.blossoms[bidx].mybestedges = Some(mbe);
        self.bestedge.insert(b, mybestedge);
    }
    fn expand_blossom(&mut self, b: usize, endstage: bool) {
        let childs = self.blossoms[b].childs.clone();
        for &s in &childs {
            self.blossomparent.insert(s, None);
            match s {
                NodeRef::B(sb) => {
                    if endstage && self.blossomdual[&sb] == 0 {
                        self.expand_blossom(sb, endstage);
                    } else {
                        for v in self.leaves(sb) {
                            self.inblossom.insert(v, s);
                        }
                    }
                }
                NodeRef::V(sv) => {
                    self.inblossom.insert(sv, s);
                }
            }
        }
        let bref = NodeRef::B(b);
        if !endstage && self.lbl(bref) == Some(2) {
            let le = self.labeledge[&bref].unwrap();
            let entrychild = self.inblossom[&le.1];
            let nch = childs.len() as i64;
            let mut j = childs.iter().position(|&c| c == entrychild).unwrap() as i64;
            let jstep: i64;
            if j & 1 != 0 {
                j -= nch;
                jstep = 1;
            } else {
                jstep = -1;
            }
            let idx = |j: i64, n: i64| -> usize { (((j % n) + n) % n) as usize };
            let (mut v, mut w) = le;
            while j != 0 {
                let (p, q);
                if jstep == 1 {
                    let e = self.blossoms[b].edges[idx(j, nch)];
                    p = e.0;
                    q = e.1;
                } else {
                    let e = self.blossoms[b].edges[idx(j - 1, nch)];
                    q = e.0;
                    p = e.1;
                }
                self.label.shift_remove(&NodeRef::V(w));
                self.label.shift_remove(&NodeRef::V(q));
                self.assign_label(w, 2, Some(v));
                self.allowedge.insert((p, q));
                self.allowedge.insert((q, p));
                j += jstep;
                if jstep == 1 {
                    let e = self.blossoms[b].edges[idx(j, nch)];
                    v = e.0;
                    w = e.1;
                } else {
                    let e = self.blossoms[b].edges[idx(j - 1, nch)];
                    w = e.0;
                    v = e.1;
                }
                self.allowedge.insert((v, w));
                self.allowedge.insert((w, v));
                j += jstep;
            }
            let bw = childs[idx(j, nch)];
            self.label.insert(NodeRef::V(w), 2);
            self.label.insert(bw, 2);
            self.labeledge.insert(NodeRef::V(w), Some((v, w)));
            self.labeledge.insert(bw, Some((v, w)));
            self.bestedge.insert(bw, None);
            j += jstep;
            while childs[idx(j, nch)] != entrychild {
                let bv = childs[idx(j, nch)];
                if self.lbl(bv) == Some(1) {
                    j += jstep;
                    continue;
                }
                let vsel: Option<usize> = match bv {
                    NodeRef::B(bvi) => {
                        let lv = self.leaves(bvi);
                        let mut last = None;
                        for v in lv {
                            last = Some(v);
                            if self.lbl(NodeRef::V(v)).is_some() {
                                break;
                            }
                        }
                        last
                    }
                    NodeRef::V(vv) => Some(vv),
                };
                if let Some(v) = vsel {
                    if self.lbl(NodeRef::V(v)).is_some() {
                        self.label.shift_remove(&NodeRef::V(v));
                        let mb = self.mate[&self.blossombase[&bv]];
                        self.label.shift_remove(&NodeRef::V(mb));
                        let lv = self.labeledge[&NodeRef::V(v)].unwrap().0;
                        self.assign_label(v, 2, Some(lv));
                    }
                }
                j += jstep;
            }
        }
        self.label.shift_remove(&bref);
        self.labeledge.shift_remove(&bref);
        self.bestedge.shift_remove(&bref);
        self.blossomparent.shift_remove(&bref);
        self.blossombase.shift_remove(&bref);
        self.blossomdual.shift_remove(&b);
    }
    fn augment_blossom(&mut self, b: usize, v: usize) {
        let mut t = NodeRef::V(v);
        while self.blossomparent[&t] != Some(b) {
            t = NodeRef::B(self.blossomparent[&t].unwrap());
        }
        if let NodeRef::B(tb) = t {
            self.augment_blossom(tb, v);
        }
        let childs = self.blossoms[b].childs.clone();
        let nch = childs.len() as i64;
        let i = childs.iter().position(|&c| c == t).unwrap() as i64;
        let mut j = i;
        let jstep: i64;
        if i & 1 != 0 {
            j -= nch;
            jstep = 1;
        } else {
            jstep = -1;
        }
        let idx = |j: i64, n: i64| -> usize { (((j % n) + n) % n) as usize };
        while j != 0 {
            j += jstep;
            let t = childs[idx(j, nch)];
            let (w, x);
            if jstep == 1 {
                let e = self.blossoms[b].edges[idx(j, nch)];
                w = e.0;
                x = e.1;
            } else {
                let e = self.blossoms[b].edges[idx(j - 1, nch)];
                x = e.0;
                w = e.1;
            }
            if let NodeRef::B(tb) = t {
                self.augment_blossom(tb, w);
            }
            j += jstep;
            let t = childs[idx(j, nch)];
            if let NodeRef::B(tb) = t {
                self.augment_blossom(tb, x);
            }
            self.mate.insert(w, x);
            self.mate.insert(x, w);
        }
        let iu = i as usize;
        let bl = &mut self.blossoms[b];
        let mut nc = bl.childs[iu..].to_vec();
        nc.extend_from_slice(&bl.childs[..iu]);
        bl.childs = nc;
        let mut ne = bl.edges[iu..].to_vec();
        ne.extend_from_slice(&bl.edges[..iu]);
        bl.edges = ne;
        let first = self.blossoms[b].childs[0];
        let nb = self.blossombase[&first];
        self.blossombase.insert(NodeRef::B(b), nb);
    }
    fn augment_matching(&mut self, v: usize, w: usize) {
        for (s0, j0) in [(v, w), (w, v)] {
            let mut s = s0;
            let mut j = j0;
            loop {
                let bs = self.inblossom[&s];
                if let NodeRef::B(bsi) = bs {
                    self.augment_blossom(bsi, s);
                }
                self.mate.insert(s, j);
                if self.labeledge[&bs].is_none() {
                    break;
                }
                let t = self.labeledge[&bs].unwrap().0;
                let bt = self.inblossom[&t];
                let (ns, nj) = self.labeledge[&bt].unwrap();
                s = ns;
                j = nj;
                if let NodeRef::B(bti) = bt {
                    self.augment_blossom(bti, j);
                }
                self.mate.insert(j, s);
            }
        }
    }
}

pub fn max_weight_matching(g: &Graph) -> Vec<(usize, usize)> {
    let gnodes = g.nodes();
    if gnodes.is_empty() {
        return vec![];
    }
    let mut maxweight = 0i64;
    for (i, j, wt) in g.edges_with_weight() {
        if i != j && wt > maxweight {
            maxweight = wt;
        }
    }
    let mut m = Matcher {
        g,
        gnodes: gnodes.clone(),
        blossoms: Vec::new(),
        mate: IndexMap::new(),
        label: IndexMap::new(),
        labeledge: IndexMap::new(),
        inblossom: gnodes.iter().map(|&v| (v, NodeRef::V(v))).collect(),
        blossomparent: gnodes.iter().map(|&v| (NodeRef::V(v), None)).collect(),
        blossombase: gnodes.iter().map(|&v| (NodeRef::V(v), v)).collect(),
        bestedge: IndexMap::new(),
        dualvar: gnodes.iter().map(|&v| (v, maxweight)).collect(),
        blossomdual: IndexMap::new(),
        allowedge: std::collections::HashSet::new(),
        queue: Vec::new(),
    };
    loop {
        m.label.clear();
        m.labeledge.clear();
        m.bestedge.clear();
        for bi in 0..m.blossoms.len() {
            if m.blossomdual.contains_key(&bi) {
                m.blossoms[bi].mybestedges = None;
            }
        }
        m.allowedge.clear();
        m.queue.clear();
        let gn = m.gnodes.clone();
        for &v in &gn {
            if !m.mate.contains_key(&v) && m.lbl(m.inblossom[&v]).is_none() {
                m.assign_label(v, 1, None);
            }
        }
        let mut augmented = false;
        loop {
            while !m.queue.is_empty() && !augmented {
                let v = m.queue.pop().unwrap();
                for w in g.neighbors(v) {
                    if w == v {
                        continue;
                    }
                    let bv = m.inblossom[&v];
                    let bw = m.inblossom[&w];
                    if bv == bw {
                        continue;
                    }
                    let mut kslack = 0;
                    if !m.allowedge.contains(&(v, w)) {
                        kslack = m.slack(v, w);
                        if kslack <= 0 {
                            m.allowedge.insert((v, w));
                            m.allowedge.insert((w, v));
                        }
                    }
                    if m.allowedge.contains(&(v, w)) {
                        if m.lbl(bw).is_none() {
                            m.assign_label(w, 2, Some(v));
                        } else if m.lbl(bw) == Some(1) {
                            let base = m.scan_blossom(v, w);
                            match base {
                                Some(b) => m.add_blossom(b, v, w),
                                None => {
                                    m.augment_matching(v, w);
                                    augmented = true;
                                    break;
                                }
                            }
                        } else if m.lbl(NodeRef::V(w)).is_none() {
                            m.label.insert(NodeRef::V(w), 2);
                            m.labeledge.insert(NodeRef::V(w), Some((v, w)));
                        }
                    } else if m.lbl(bw) == Some(1) {
                        let cur = m.bestedge.get(&bv).cloned().flatten();
                        if cur.is_none() || kslack < m.slack(cur.unwrap().0, cur.unwrap().1) {
                            m.bestedge.insert(bv, Some((v, w)));
                        }
                    } else if m.lbl(NodeRef::V(w)).is_none() {
                        let cur = m.bestedge.get(&NodeRef::V(w)).cloned().flatten();
                        if cur.is_none() || kslack < m.slack(cur.unwrap().0, cur.unwrap().1) {
                            m.bestedge.insert(NodeRef::V(w), Some((v, w)));
                        }
                    }
                }
            }
            if augmented {
                break;
            }
            // compute delta
            let mut deltatype = 1;
            let mut delta: i64 = *m.dualvar.values().min().unwrap();
            let mut deltaedge: Option<(usize, usize)> = None;
            let mut deltablossom: Option<usize> = None;
            for &v in &gnodes {
                if m.lbl(m.inblossom[&v]).is_none() {
                    if let Some(Some(be)) = m.bestedge.get(&NodeRef::V(v)) {
                        let d = m.slack(be.0, be.1);
                        if d < delta {
                            delta = d;
                            deltatype = 2;
                            deltaedge = Some(*be);
                        }
                    }
                }
            }
            let bp_keys: Vec<NodeRef> = m.blossomparent.keys().cloned().collect();
            for b in bp_keys {
                if m.blossomparent[&b].is_none() && m.lbl(b) == Some(1) {
                    if let Some(Some(be)) = m.bestedge.get(&b) {
                        let kslack = m.slack(be.0, be.1);
                        let d = kslack.div_euclid(2);
                        if d < delta {
                            delta = d;
                            deltatype = 3;
                            deltaedge = Some(*be);
                        }
                    }
                }
            }
            let bd_keys: Vec<usize> = m.blossomdual.keys().cloned().collect();
            for b in bd_keys {
                if m.blossomparent[&NodeRef::B(b)].is_none() && m.lbl(NodeRef::B(b)) == Some(2) && m.blossomdual[&b] < delta {
                    delta = m.blossomdual[&b];
                    deltatype = 4;
                    deltablossom = Some(b);
                }
            }
            for &v in &gnodes {
                match m.lbl(m.inblossom[&v]) {
                    Some(1) => *m.dualvar.get_mut(&v).unwrap() -= delta,
                    Some(2) => *m.dualvar.get_mut(&v).unwrap() += delta,
                    _ => {}
                }
            }
            let bd_keys: Vec<usize> = m.blossomdual.keys().cloned().collect();
            for b in bd_keys {
                if m.blossomparent[&NodeRef::B(b)].is_none() {
                    match m.lbl(NodeRef::B(b)) {
                        Some(1) => *m.blossomdual.get_mut(&b).unwrap() += delta,
                        Some(2) => *m.blossomdual.get_mut(&b).unwrap() -= delta,
                        _ => {}
                    }
                }
            }
            if deltatype == 1 {
                break;
            } else if deltatype == 2 || deltatype == 3 {
                let (v, w) = deltaedge.unwrap();
                m.allowedge.insert((v, w));
                m.allowedge.insert((w, v));
                m.queue.push(v);
            } else if deltatype == 4 {
                m.expand_blossom(deltablossom.unwrap(), false);
            }
        }
        if !augmented {
            break;
        }
        let bd_keys: Vec<usize> = m.blossomdual.keys().cloned().collect();
        for b in bd_keys {
            if !m.blossomdual.contains_key(&b) {
                continue;
            }
            if m.blossomparent[&NodeRef::B(b)].is_none() && m.lbl(NodeRef::B(b)) == Some(1) && m.blossomdual[&b] == 0 {
                m.expand_blossom(b, true);
            }
        }
    }
    // matching_dict_to_set
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut set: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for (&u, &v) in &m.mate {
        if set.contains(&(v, u)) || set.contains(&(u, v)) {
            continue;
        }
        set.insert((u, v));
        edges.push((u, v));
    }
    edges
}

#[allow(dead_code)]
fn _unused(_: IndexSet<usize>) {}

# app.py
from flask import Flask, request, render_template_string, jsonify
from collections import deque
import heapq
import math

app = Flask(__name__)

# ----------------------------
# Graphs & algorithms (backend)
# ----------------------------
GRAPH = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [],
    "E": [],
    "F": []
}

GRAPH_A = {
    "A": [("B", 1), ("C", 4)],
    "B": [("D", 1)],
    "C": [("D", 1)],
    "D": []
}
HEURISTIC = {"A": 3, "B": 1, "C": 1, "D": 0}

def bfs_order(graph, start, goal=None):
    visited = set()
    q = deque([start])
    order = []
    while q:
        node = q.popleft()
        if node in visited:
            continue
        order.append(node)
        if node == goal:
            return order, True
        visited.add(node)
        for nb in graph.get(node, []):
            if nb not in visited:
                q.append(nb)
    return order, False

def dfs_order(graph, start, goal=None):
    visited = set()
    order = []
    found = False
    def _dfs(u):
        nonlocal found
        if found:
            return
        visited.add(u)
        order.append(u)
        if u == goal:
            found = True
            return
        for nb in graph.get(u, []):
            if nb not in visited:
                _dfs(nb)
    _dfs(start)
    return order, found

def a_star(graph, h, start, goal):
    pq = []
    heapq.heappush(pq, (h[start], 0, start, [start]))
    closed = set()
    while pq:
        f, g, node, path = heapq.heappop(pq)
        if node == goal:
            return path, True, g
        if node in closed:
            continue
        closed.add(node)
        for nb, cost in graph.get(node, []):
            if nb in closed:
                continue
            new_g = g + cost
            new_f = new_g + h.get(nb, 0)
            heapq.heappush(pq, (new_f, new_g, nb, path + [nb]))
    return [], False, math.inf

# ----------------------------
# Fuzzy logic (no numpy)
# ----------------------------
def triangular(x, a, b, c):
    if a == b and x == a:
        return 1.0
    if b == c and x == c:
        return 1.0
    if x <= a or x >= c:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    if b <= x < c:
        return (c - x) / (c - b)
    return 0.0

# Food
def food_bad(x): return triangular(x, 0, 0, 5)
def food_good(x): return triangular(x, 5, 10, 10)

# Service
def service_poor(x): return triangular(x, 0, 0, 5)
def service_excellent(x): return triangular(x, 5, 10, 10)

# Tip
def tip_low(x): return triangular(x, 0, 0, 10)
def tip_high(x): return triangular(x, 10, 20, 20)

def defuzzify_centroid(low_strength, high_strength, steps=400):
    xs = [i * (20.0 / (steps - 1)) for i in range(steps)]
    num = 0.0
    den = 0.0
    for x in xs:
        mu_low = min(low_strength, tip_low(x))
        mu_high = min(high_strength, tip_high(x))
        mu = max(mu_low, mu_high)
        num += x * mu
        den += mu
    return (num / den) if den != 0 else 0.0

def compute_fuzzy(food, service):
    mu_bad = food_bad(food)
    mu_good = food_good(food)
    mu_poor = service_poor(service)
    mu_exc = service_excellent(service)

    low_strength = max(mu_poor, mu_bad)
    high_strength = min(mu_exc, mu_good)
    tip = defuzzify_centroid(low_strength, high_strength)
    return {
        "mu_bad": mu_bad,
        "mu_good": mu_good,
        "mu_poor": mu_poor,
        "mu_exc": mu_exc,
        "low_strength": low_strength,
        "high_strength": high_strength,
        "tip_percent": tip
    }

# ----------------------------
# Routes (JSON endpoints for front-end)
# ----------------------------
@app.route("/")
def index():
    # path to uploaded image (from your session). We'll render it as-is.
    uploaded = "/mnt/data/45f5282c-7667-4c70-99be-db90ac5722e5.png"
    return render_template_string(TEMPLATE, uploaded_image=uploaded)

@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.json or {}
    algo = data.get("algo", "bfs")
    start = data.get("start", "A")
    goal = data.get("goal", None)
    if algo == "bfs":
        order, found = bfs_order(GRAPH, start, goal)
    else:
        order, found = dfs_order(GRAPH, start, goal)
    # also return graph nodes/edges for visualization
    nodes = [{"id": n, "label": n} for n in GRAPH.keys()]
    edges = []
    for u, vs in GRAPH.items():
        for v in vs:
            edges.append({"from": u, "to": v})
    return jsonify({"order": order, "found": found, "nodes": nodes, "edges": edges})

@app.route("/api/astar", methods=["POST"])
def api_astar():
    data = request.json or {}
    start = data.get("start", "A")
    goal = data.get("goal", "D")
    path, found, cost = a_star(GRAPH_A, HEURISTIC, start, goal)
    nodes = [{"id": n, "label": n} for n in GRAPH_A.keys()]
    edges = []
    for u, vs in GRAPH_A.items():
        for v, c in vs:
            edges.append({"from": u, "to": v, "label": str(c)})
    return jsonify({"path": path, "found": found, "cost": cost, "nodes": nodes, "edges": edges})

@app.route("/api/fuzzy", methods=["POST"])
def api_fuzzy():
    data = request.json or {}
    food = float(data.get("food", 7))
    service = float(data.get("service", 3))
    res = compute_fuzzy(food, service)
    return jsonify(res)

# ----------------------------
# Frontend template (vis-network + Chart.js)
# ----------------------------
TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Search & Fuzzy — Visual App</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <!-- vis-network for graph visualization -->
  <script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js">
/* -------------------------
   Fuzzy Tip Triangular Curve (PATCH)
   ------------------------- */

function tri(x, a, b, c) {
  if (x <= a || x >= c) return 0;
  if (a < x && x < b) return (x - a) / (b - a);
  if (b <= x && x < c) return (c - x) / (c - b);
  return 0;
}

let xs_tri = [];
let lowTri = [];
let highTri = [];
for (let x = 0; x <= 20; x++) {
  xs_tri.push(x);
  lowTri.push(tri(x, 0, 0, 10));
  highTri.push(tri(x, 10, 20, 20));
}

let tipTriangleCtx = document.getElementById('tipTriangle').getContext('2d');
let tipTriangleChart = new Chart(tipTriangleCtx, {
  type: 'line',
  data: {
    labels: xs_tri,
    datasets: [
      {
        label: 'Tip Low (0,0,10)',
        data: lowTri,
        borderWidth: 2,
        fill: false
      },
      {
        label: 'Tip High (10,20,20)',
        data: highTri,
        borderWidth: 2,
        fill: false
      }
    ]
  },
  options: {
    responsive: true,
    scales: {
      x: { title: { display: true, text: 'Tip (%)' } },
      y: { min: 0, max: 1, title: { display: true, text: 'Membership' } }
    }
  }
});

</script>
  <link href="https://unpkg.com/vis-network@9.1.2/styles/vis-network.min.css" rel="stylesheet" />
  <!-- Chart.js for fuzzy plots -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js">
/* -------------------------
   Fuzzy Tip Triangular Curve (PATCH)
   ------------------------- */

function tri(x, a, b, c) {
  if (x <= a || x >= c) return 0;
  if (a < x && x < b) return (x - a) / (b - a);
  if (b <= x && x < c) return (c - x) / (c - b);
  return 0;
}

let xs_tri = [];
let lowTri = [];
let highTri = [];
for (let x = 0; x <= 20; x++) {
  xs_tri.push(x);
  lowTri.push(tri(x, 0, 0, 10));
  highTri.push(tri(x, 10, 20, 20));
}

let tipTriangleCtx = document.getElementById('tipTriangle').getContext('2d');
let tipTriangleChart = new Chart(tipTriangleCtx, {
  type: 'line',
  data: {
    labels: xs_tri,
    datasets: [
      {
        label: 'Tip Low (0,0,10)',
        data: lowTri,
        borderWidth: 2,
        fill: false
      },
      {
        label: 'Tip High (10,20,20)',
        data: highTri,
        borderWidth: 2,
        fill: false
      }
    ]
  },
  options: {
    responsive: true,
    scales: {
      x: { title: { display: true, text: 'Tip (%)' } },
      y: { min: 0, max: 1, title: { display: true, text: 'Membership' } }
    }
  }
});

</script>

  <style>
    :root{--card:#ffffff;--muted:#6b7280;--accent:#2563eb;--bg:#f3f4f6}
    body{font-family:Inter,Segoe UI,Helvetica,Arial,sans-serif;background:var(--bg);margin:0;padding:24px;color:#111}
    .container{max-width:1100px;margin:0 auto}
    header{display:flex;gap:16px;align-items:center;margin-bottom:18px}
    h1{margin:0;font-size:1.4rem}
    .grid{display:grid;grid-template-columns:1fr 420px;gap:18px}
    .card{background:var(--card);padding:14px;border-radius:12px;box-shadow:0 6px 18px rgba(15,23,42,0.06)}
    .card h3{margin:0 0 8px 0}
    label{display:block;font-size:0.9rem;margin-bottom:6px;color:var(--muted)}
    input[type=text], input[type=number], select{width:100%;padding:8px;border-radius:8px;border:1px solid #e5e7eb}
    button{background:var(--accent);color:white;padding:8px 12px;border-radius:8px;border:0;cursor:pointer}
    button.secondary{background:#f3f4f6;color:#111}
    #network{height:340px;border-radius:8px;border:1px solid #e5e7eb}
    .trace-list{font-family:monospace;white-space:pre-wrap; margin-top:10px; color:#374151}
    .small{font-size:0.85rem;color:var(--muted)}
    .flex{display:flex;gap:10px;align-items:center}
    .right {text-align:right}
    .img-snap{max-width:320px;border-radius:8px;border:1px solid #e5e7eb}
    footer{margin-top:18px;color:var(--muted);font-size:0.85rem}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <img src="{{ uploaded_image }}" alt="upload" class="img-snap" onerror="this.style.display='none'">
      <div>
        <h1>Search & Fuzzy — Visual App</h1>
      </div>
    </header>

    <div class="grid">
      <div>
        <!-- Graph Card -->
        <div class="card" id="graph-card">
          <h3>Blind Search</h3>
          <div class="flex">
            <div style="flex:1">
              <label>Mulai</label>
              <input id="startNode" value="A">
            </div>
            <div style="width:120px">
              <label>Tujuan</label>
              <input id="goalNode" value="F">
            </div>
            <div style="width:140px">
              <label>Algoritma</label>
              <select id="algoSelect">
                <option value="bfs">BFS</option>
                <option value="dfs">DFS</option>
              </select>
            </div>
            <div style="align-self:end">
              <button id="runSearchBtn">Jalankan</button>
            </div>
          </div>

          <div id="network" style="margin-top:12px"></div>
          <div class="trace-list" id="trace"></div>
        </div>

        <!-- A* Card -->
        <div class="card" style="margin-top:14px;">
          <h3>A* Heuristic Search</h3>
          <div class="flex">
            <div style="flex:1">
              <label>Mulai</label><input id="astart" value="A">
            </div>
            <div style="width:120px">
              <label>Tujuan</label><input id="agoal" value="D">
            </div>
            <div style="align-self:end">
              <button id="runAStarBtn" class="">Jalankan A*</button>
            </div>
          </div>
          <div id="network_astar" style="height:260px;margin-top:12px;border-radius:8px;border:1px solid #e5e7eb"></div>
          <div class="trace-list" id="astarTrace"></div>
        </div>
      </div>

      <!-- Right column: Fuzzy -->
      <div>
        <div class="card">
          <h3>Kalkulator Fuzzy Tip</h3>
          <div class="small">Makanan & Layanan pada [0-10] → Tip pada [0-20%]</div>

          <div style="margin-top:10px">
            <label>Kualitas Makanan</label>
            <input id="foodInput" type="number" min="0" max="10" step="0.1" value="7">
            <label style="margin-top:8px">Kualitas Layanan</label>
            <input id="serviceInput" type="number" min="0" max="10" step="0.1" value="3">
            <div style="margin-top:10px" class="flex">
              <button id="runFuzzyBtn">Hitung Tip</button>
              <button id="exampleBtn" class="secondary">Gunakan Contoh (7,3)</button>
            </div>
          </div>

          <canvas id="membershipChart" style="margin-top:12px; height:180px"></canvas>
          <div style="margin-top:10px">
            <label>Tip Hasil Defuzzifikasi</label>
            <div id="tipValue" style="font-size:1.6rem;font-weight:600;color:var(--accent)">— %</div>
            <canvas id="tipBar" style="height:80px;margin-top:6px"></canvas>
<canvas id="tipTriangle" style="margin-top:12px; height:160px"></canvas>
          </div>
        </div>

        <div class="card" style="margin-top:14px">
          <h3>Catatan</h3>
          <p class="small">Aturan yang Diterapkan:
            <ol>
              <li>Jika Layanan buruk atau Makanan tidak enak maka TIP rendah</li>
              <li>Jika Layanan luar biasa dan Makanan enak maka TIP tinggi</li>
            </ol>
          </p>
          <p class="small">Segitiga: Makanan Buruk (0,0,5), 
          Makanan Baik (5,10,10), Layanan Buruk (0,0,5), 
          Layanan Sangat Baik (5,10,10), Tip Rendah (0,0,10), Tip Tinggi (10,20,20)
          </p>
        </div>
      </div>
    </div>

    <footer>Made with Flask • vis-network • Chart.js</footer>
  </div>

<script>
/* -------------------------
   Helper: draw vis network
   ------------------------- */
function createNetwork(container, nodes, edges, options){
  const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
  return new vis.Network(container, data, options || {});
}

/* -------------------------
   Initialize default network
   ------------------------- */
let defaultNodes = [{"id":"A","label":"A"},{"id":"B","label":"B"},{"id":"C","label":"C"},{"id":"D","label":"D"},{"id":"E","label":"E"},{"id":"F","label":"F"}];
let defaultEdges = [{"from":"A","to":"B"},{"from":"A","to":"C"},{"from":"B","to":"D"},{"from":"B","to":"E"},{"from":"C","to":"F"}];

let network = createNetwork(document.getElementById('network'), defaultNodes, defaultEdges, {
  layout: { hierarchical: { enabled: true, direction: 'UD', sortMethod: 'directed' } },
  edges: { arrows: 'to' },
  physics: { enabled: false }
});

let networkA = createNetwork(document.getElementById('network_astar'),
  [{"id":"A","label":"A"},{"id":"B","label":"B"},{"id":"C","label":"C"},{"id":"D","label":"D"}],
  [{"from":"A","to":"B","label":"1"},{"from":"A","to":"C","label":"4"},{"from":"B","to":"D","label":"1"},{"from":"C","to":"D","label":"1"}],
  { layout:{ hierarchical:{ enabled:true, direction:'LR' }}, edges:{arrows:'to'} , physics:false }
);

/* -------------------------
   Animate traversal
   ------------------------- */
function animateTraversal(net, order, highlightColor="#ff7a59", delay=700){
  const nodesDS = net.body.data.nodes;
  // reset colors
  nodesDS.forEach(n => nodesDS.update({id: n.id, color: undefined}));
  let i=0;
  function step(){
    if(i >= order.length) return;
    const id = order[i];
    nodesDS.update({id: id, color: {background: highlightColor}});
    i++;
    setTimeout(step, delay);
  }
  step();
}

/* -------------------------
   BFS/DFS button
   ------------------------- */
document.getElementById("runSearchBtn").addEventListener("click", async ()=>{
  const start = document.getElementById("startNode").value || "A";
  const goal = document.getElementById("goalNode").value || null;
  const algo = document.getElementById("algoSelect").value;
  const res = await fetch("/api/search", {
    method:"POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({start, goal, algo})
  });
  const j = await res.json();
  // rebuild network with returned nodes/edges
  const nodes = j.nodes.map(n => ({id:n.id, label:n.label}));
  const edges = j.edges.map(e => ({from:e.from, to:e.to}));
  network.setData({nodes: nodes, edges: edges});
  // animate
  document.getElementById("trace").innerText = "Visited order: " + j.order.join(", ");
  animateTraversal(network, j.order, "#60a5fa", 700);
});

/* -------------------------
   A* button
   ------------------------- */
document.getElementById("runAStarBtn").addEventListener("click", async ()=>{
  const start = document.getElementById("astart").value || "A";
  const goal = document.getElementById("agoal").value || "D";
  const res = await fetch("/api/astar", {
    method:"POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({start, goal})
  });
  const j = await res.json();
  // rebuild A* network
  const nodes = j.nodes.map(n => ({id:n.id, label:n.label}));
  const edges = j.edges.map(e => ({from:e.from, to:e.to, label:e.label}));
  networkA.setData({nodes: nodes, edges: edges});
  // highlight path
  const path = j.path;
  document.getElementById("astarTrace").innerText = "Path: " + path.join(" → ") + "  |  Cost: " + j.cost;
  // color nodes in path
  const nodesDS = networkA.body.data.nodes;
  nodesDS.forEach(n => nodesDS.update({id: n.id, color: undefined}));
  path.forEach((id, idx) => {
    setTimeout(()=> nodesDS.update({id: id, color: {background: "#34d399"}}), idx*500);
  });
});

/* -------------------------
   Fuzzy: Chart.js setup
   ------------------------- */
let membershipCtx = document.getElementById('membershipChart').getContext('2d');
let membershipChart = new Chart(membershipCtx, {
  type: 'bar',
  data: {
    labels: ['Makanan Tidak Enak','Makanan Enak','Service Buruk','Service Luar Biasa','Kekuatan Rendah','Kekuatan Tinggi'],
    datasets: [{
      label: 'Membership (0..1)',
      data: [0,0,0,0,0,0],
      borderRadius:6
    }]
  },
  options: {
    indexAxis: 'y',
    scales: { x: { min:0, max:1 } },
    plugins: { legend: { display:false } }
  }
});

let tipCtx = document.getElementById('tipBar').getContext('2d');
let tipChart = new Chart(tipCtx, {
  type: 'bar',
  data: {
    labels: ['Tip %'],
    datasets: [{ label: 'Tip', data:[0], borderRadius:6 }]
  },
  options: {
    scales: { y: { min:0, max:20 } },
    plugins: { legend: { display:false } }
  }
});

/* -------------------------
   Fuzzy compute
   ------------------------- */
async function runFuzzy(food, service){
  const res = await fetch("/api/fuzzy", {
    method:"POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({food: food, service: service})
  });
  const j = await res.json();
  // update membership chart
  membershipChart.data.datasets[0].data = [
    j.mu_bad, j.mu_good, j.mu_poor, j.mu_exc, j.low_strength, j.high_strength
  ];
  membershipChart.update();
  // update tip chart & numeric
  document.getElementById("tipValue").innerText = j.tip_percent.toFixed(3) + " %";
  tipChart.data.datasets[0].data = [j.tip_percent];
  tipChart.update();
}

/* Buttons */
document.getElementById("runFuzzyBtn").addEventListener("click", ()=>{
  const food = parseFloat(document.getElementById("foodInput").value) || 0;
  const service = parseFloat(document.getElementById("serviceInput").value) || 0;
  runFuzzy(food, service);
});
document.getElementById("exampleBtn").addEventListener("click", ()=>{
  document.getElementById("foodInput").value = 7;
  document.getElementById("serviceInput").value = 3;
  runFuzzy(7,3);
});

/* run example on load */
window.addEventListener("load", ()=> {
  runFuzzy(7,3);
});

/* -------------------------
   Fuzzy Tip Triangular Curve (PATCH)
   ------------------------- */

function tri(x, a, b, c) {
  if (x <= a || x >= c) return 0;
  if (a < x && x < b) return (x - a) / (b - a);
  if (b <= x && x < c) return (c - x) / (c - b);
  return 0;
}

let xs_tri = [];
let lowTri = [];
let highTri = [];
for (let x = 0; x <= 20; x++) {
  xs_tri.push(x);
  lowTri.push(tri(x, 0, 0, 10));
  highTri.push(tri(x, 10, 20, 20));
}

let tipTriangleCtx = document.getElementById('tipTriangle').getContext('2d');
let tipTriangleChart = new Chart(tipTriangleCtx, {
  type: 'line',
  data: {
    labels: xs_tri,
    datasets: [
      {
        label: 'Tip Rendah (0,0,10)',
        data: lowTri,
        borderWidth: 2,
        fill: false
      },
      {
        label: 'Tip Tinggi (10,20,20)',
        data: highTri,
        borderWidth: 2,
        fill: false
      }
    ]
  },
  options: {
    responsive: true,
    scales: {
      x: { title: { display: true, text: 'Tip (%)' } },
      y: { min: 0, max: 1, title: { display: true, text: 'Membership' } }
    }
  }
});

</script>
</body>
</html>
"""

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)

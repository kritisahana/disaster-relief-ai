from flask import Flask, render_template, request
import os
import random
import math
from heapq import heappush, heappop
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from heapq import heappush, heappop

app = Flask(__name__)

# ---------------------- Genetic Algorithm ---------------------- 
def ga_optimizer(areas, resources, generations=30, population_size=30, elite_size=2, mutation_rate=0.15):
    
    total_food = max(1, int(resources.get("Food", 0)))
    total_water = max(1, int(resources.get("Water", 0)))
    total_med = max(1, int(resources.get("Medical", 0)))

    def random_plan(): 
        plan = []
        total_population = sum(a["Population"] for a in areas) or 1  # avoid divide-by-zero

        for a in areas: # so resources arent wasted
            pop = a["Population"]
            pop_ratio = pop / total_population     # for example 100/300 = 0.33, 0.33*300 = 100 (we have 300 area a - 100 )
            max_food = int(pop_ratio * total_food)
            max_water = int(pop_ratio * total_water)
            max_med = int(pop_ratio * total_med)

            # random allocation bounded by population share
            plan.append({
                "Area": a["Area"],
                "Allocated": {
                    "Food": random.randint(0, max_food),
                    "Water": random.randint(0, max_water),
                    "Medical": random.randint(0, max_med)
                }
            })
        return plan


    def fitness(plan):
        fitness_score = 0.0
        delivery_penalty = 0.0

        for p, a in zip(plan, areas):
            need = (a.get("Need") or "").lower()
            

            food = p["Allocated"]["Food"]
            water = p["Allocated"]["Water"]
            medical = p["Allocated"]["Medical"]

            if need == "food":
                # heavily reward food allocation; penalize if too little
                score = (0.7 * food) + (0.2 * water) + (0.1 * medical)
                if food < 0.1 * total_food / len(areas):  # if food is less than 10% of fair share
                    score -= 200  # strong penalty
            elif need == "water":
                score = (0.7 * water) + (0.2 * food) + (0.1 * medical)
                if water < 0.1 * total_water / len(areas):
                    score -= 200
            elif need == "medical":
                score = (0.7 * medical) + (0.2 * food) + (0.1 * water)
                if medical < 0.1 * total_med / len(areas):
                    score -= 200
            else:
                score = 0.33 * (food + water + medical)

            fitness_score += score

        return fitness_score 


    population = [random_plan() for _ in range(population_size)]
    best_fitness_over_time = []
    avg_fitness_over_time = []

    for gen in range(generations):
        # --- Evaluate fitness of all individuals ---
        scored = [(fitness(ind), ind) for ind in population]
        scored.sort(key=lambda x: x[0], reverse=True)
        fitness_values = [s for s, _ in scored]

        
        best = scored[0][0]
        avg = sum(fitness_values) / len(fitness_values)
        best_fitness_over_time.append(best)
        avg_fitness_over_time.append(avg)

        
        elites = [ind for _, ind in scored[:elite_size]]

        # --- Normalize fitness values into probabilities --- (So it doesnt go beyond limit)
        min_fit = min(fitness_values)
        shift = -min_fit + 1.0 if min_fit < 0 else 0
        fitness_positive = [f + shift for f in fitness_values]
        total_fp = sum(fitness_positive)
        probs = [fp / total_fp if total_fp > 0 else 1.0 / len(fitness_positive) for fp in fitness_positive]

        def pick_parent():
            r = random.random()
            cum = 0.0
            for (p, _), prob in zip(scored, probs):
                cum += prob
                if r <= cum:
                    return _
            return scored[0][1]

        # --- Child generation ---
        children = elites.copy()

        # --- Resource constraint --- total from mutation is 450 total food we have is 400 
        def normalize_resources(child):
            total_allocated = {"Food": 0, "Water": 0, "Medical": 0}
            for area in child:
                for r in total_allocated:
                    total_allocated[r] += area["Allocated"][r]

            for r, total in total_allocated.items():
                limit = total_food if r == "Food" else (total_water if r == "Water" else total_med)
                if total > limit:
                    # Scale down proportionally 
                    scale = limit / total
                    for area in child:
                        area["Allocated"][r] = int(area["Allocated"][r] * scale)
            return child

        # --- Reproduction loop --- 

        while len(children) < population_size:
            p1 = pick_parent()
            p2 = pick_parent()

            # --- Crossover ---
            if len(areas) > 1:
                cp = random.randint(1, len(areas) - 1)
                child = [dict(x) for x in (p1[:cp] + p2[cp:])]
            else:
                child = [dict(x) for x in p1]

            # --- Mutation ---
            if random.random() < mutation_rate:
                area_idx = random.randint(0, len(areas) - 1)
                res_type = random.choice(["Food", "Water", "Medical"])
                cur_val = child[area_idx]["Allocated"].get(res_type, 0)
                delta = int(max(1, round(0.10 * (total_food if res_type == "Food"
                                                else (total_water if res_type == "Water"
                                                    else total_med)))))
                change = random.randint(-delta, delta)
                new_val = max(0, min(cur_val + change,
                                    total_food if res_type == "Food"
                                    else total_water if res_type == "Water"
                                    else total_med))
                child[area_idx]["Allocated"][res_type] = new_val

            # --- Normalize to keep within resource limits ---
            child = normalize_resources(child)
            children.append(child)

        population = children

    # --- After all generations ---
    final_scored = sorted([(fitness(ind), ind) for ind in population],
                        key=lambda x: x[0], reverse=True)
    best_fitness, best_plan = final_scored[0]

    # --- Plot Fitness Progress ---
    os.makedirs("static", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, generations + 1), best_fitness_over_time, marker='o', label='Best')
    plt.plot(range(1, generations + 1), avg_fitness_over_time, marker='x', label='Average')
    plt.title('GA: Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/fitness_plot.png')
    plt.close()

    return best_plan, best_fitness


# ---------- Coordinate & graph helpers ----------
def area_coords_lookup(areas):
    return {a["Area"]: (float(a["x"]), float(a["y"])) for a in areas}

def euclid(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

def build_graph(node_names, coords, blocked_edges):
    graph = {n: [] for n in node_names}
    n = len(node_names)
    for i in range(n):
        for j in range(i+1, n):
            u, v = node_names[i], node_names[j]
            if (u, v) in blocked_edges or (v, u) in blocked_edges:
                continue
            w = euclid(coords[u], coords[v])
            graph[u].append((v, w))
            graph[v].append((u, w))
    return graph

# ---------- A* shortest path ----------
def astar_path(graph, coords, start, goal):
    if start not in graph or goal not in graph:
        return None, float('inf')
    def h(n): return euclid(coords[n], coords[goal])
    open_heap = []
    heappush(open_heap, (h(start), 0.0, start, None))
    came_from, g_score = {}, {start: 0.0}
    while open_heap:
        f, g, node, parent = heappop(open_heap)
        if node in came_from: continue
        came_from[node] = parent
        if node == goal:
            path, cur = [], node
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            return path[::-1], g_score[node]
        for nbr, w in graph[node]:
            t = g + w
            if nbr not in g_score or t < g_score[nbr]:
                g_score[nbr] = t
                heappush(open_heap, (t + h(nbr), t, nbr, node))
    return None, float('inf')

# ---------- Route planner using A* ----------
def plan_route_with_astar(areas, blocked_edges, hq_name="HQ", hq_coord=(0.0,0.0)):
    coords = area_coords_lookup(areas)
    coords[hq_name] = hq_coord
    nodes = [hq_name] + [a["Area"] for a in areas]
    graph = build_graph(nodes, coords, blocked_edges)
    unvisited = set([a["Area"] for a in areas])
    current, full_path, visited, total, unreachable = hq_name, [hq_name], [], 0.0, []
    while unvisited:
        best_next, best_path, best_dist = None, None, float('inf')
        for cand in list(unvisited):
            path, dist = astar_path(graph, coords, current, cand)
            if path and dist < best_dist:
                best_next, best_path, best_dist = cand, path, dist
        if not best_next:
            unreachable.extend(sorted(unvisited))
            break
        full_path += best_path[1:]
        total += best_dist
        visited.append(best_next)
        unvisited.remove(best_next)
        current = best_next
    return {
        "full_path": full_path,
        "visited_order": visited,
        "total_distance": total,
        "unreachable": unreachable,
        "coords": coords
    }

# ---------- Plotting ----------
def plot_route_map_astar(route_result, areas, filename='static/route_map.png'):
    os.makedirs('static', exist_ok=True)
    coords, fp = route_result["coords"], route_result["full_path"]
    plt.figure(figsize=(7,6))
    for a in areas:
        n, (x,y) = a['Area'], (coords[a['Area']][0], coords[a['Area']][1])
        plt.scatter(x, y, s=80); plt.text(x+0.3, y+0.3, n, fontsize=9)
    hx, hy = coords['HQ']
    plt.scatter(hx, hy, marker='s', s=140)
    plt.text(hx+0.3, hy+0.3, 'HQ', fontsize=10, fontweight='bold')
    for i in range(len(fp)-1):
        x1,y1 = coords[fp[i]]; x2,y2 = coords[fp[i+1]]
        plt.plot([x1,x2],[y1,y2]); plt.arrow(x1,y1,x2-x1,y2-y1,
            length_includes_head=True, head_width=0.5, head_length=0.8)
    plt.title('A* Route Map (HQ → … → Last)'); plt.xlabel('X'); plt.ylabel('Y')
    plt.grid(True); plt.tight_layout(); plt.savefig(filename); plt.close()

# ---------------------- Combined Route + GA ----------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize_and_route', methods=['POST'])
def optimize_and_route():
    areas, idx = [], 1
    while True:
        name = request.form.get(f'area{idx}')
        if not name:
            break
        try:
            x = float(request.form.get(f'x{idx}', 0))
            y = float(request.form.get(f'y{idx}', 0))
        except:
            x, y = 0, 0
        areas.append({
            "Area": name,
            "Population": int(request.form.get(f'pop{idx}', 0)),
            "Need": request.form.get(f'need{idx}', ""),
            "Urgency": request.form.get(f'urg{idx}', "Low"),
            "x": x, "y": y
        })
        idx += 1

    if not areas:
        return render_template('index.html', message="Please add at least one area.")

    resources = {
        "Food": int(request.form.get('food', 0)),
        "Water": int(request.form.get('water', 0)),
        "Medical": int(request.form.get('medical', 0))
    }

    # --- GA (unchanged) ---
    allocations, fitness = ga_optimizer(areas, resources, generations=40, population_size=40)

    # --- Blocked routes ---
    blocked_edges, j = set(), 1
    while True:
        f = (request.form.get(f'from{j}') or '').strip()
        t = (request.form.get(f'to{j}') or '').strip()
        if not f or not t: break
        blocked_edges.update({(f,t),(t,f)}); j += 1

    # --- A* planning ---
    hq = (0.0,0.0)
    route = plan_route_with_astar(areas, blocked_edges, hq_name='HQ', hq_coord=hq)
    summary, fp, coords = [], route['full_path'], route['coords']
    for k in range(len(fp)-1):
        u,v = fp[k], fp[k+1]; d = euclid(coords[u], coords[v])
        summary.append(f"{u} → {v}: {d:.2f}")
    summary.append(f"Total distance: {route['total_distance']:.2f}")
    if route['unreachable']:
        summary.append("Unreachable: "+", ".join(route['unreachable']))
    plot_route_map_astar(route, areas, filename='static/route_map.png')

    return render_template('index.html',
        allocations=allocations,
        fitness_score=fitness,
        fitness_plot='fitness_plot.png',
        route_plot='route_map.png',
        route_summary=summary,
        route_order=route['visited_order'])

if __name__ == '__main__':
    app.run(debug=True)
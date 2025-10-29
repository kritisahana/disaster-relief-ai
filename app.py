from flask import Flask, render_template, request
import os
import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# ---------------------- Improved Genetic Algorithm ----------------------
def ga_optimizer(areas, resources, generations=30, population_size=30, elite_size=2, mutation_rate=0.15):
    """
    Genetic Algorithm with:
      - Elitism
      - Bounded mutation (±10% of total resource)
      - Best + average fitness tracking
    """
    total_food = max(1, int(resources.get("Food", 0)))
    total_water = max(1, int(resources.get("Water", 0)))
    total_med = max(1, int(resources.get("Medical", 0)))

    def random_plan():
        plan = []
        for _a in areas:
            plan.append({
                "Area": _a["Area"],
                "Allocated": {
                    "Food": random.randint(0, total_food),
                    "Water": random.randint(0, total_water),
                    "Medical": random.randint(0, total_med)
                }
            })
        return plan

    def fitness(plan):
        coverage = 0.0
        urgency_score = 0.0
        delivery_penalty = 0.0
        for p, a in zip(plan, areas):
            need = (a.get("Need") or "").lower()
            urgency = (a.get("Urgency") or "low").lower()
            blocked = (a.get("Blocked") or "No").lower()

            if need == "food":
                coverage += p["Allocated"]["Food"]
            elif need == "water":
                coverage += p["Allocated"]["Water"]
            elif need == "medical":
                coverage += p["Allocated"]["Medical"]

            if urgency == "high":
                urgency_score += 100
            elif urgency == "medium":
                urgency_score += 60
            else:
                urgency_score += 30

            if blocked == "yes":
                delivery_penalty += 40

        return 0.5 * coverage + 0.3 * urgency_score - 0.2 * delivery_penalty

    population = [random_plan() for _ in range(population_size)]
    best_fitness_over_time = []
    avg_fitness_over_time = []

    for gen in range(generations):
        scored = [(fitness(ind), ind) for ind in population]
        scored.sort(key=lambda x: x[0], reverse=True)
        fitness_values = [s for s, _ in scored]

        best = scored[0][0]
        avg = sum(fitness_values) / len(fitness_values)
        best_fitness_over_time.append(best)
        avg_fitness_over_time.append(avg)

        elites = [ind for _, ind in scored[:elite_size]]

        # Selection probabilities
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

        children = elites.copy()
        while len(children) < population_size:
            p1 = pick_parent()
            p2 = pick_parent()
            if len(areas) > 1:
                cp = random.randint(1, len(areas) - 1)
                child = [dict(x) for x in (p1[:cp] + p2[cp:])]
            else:
                child = [dict(x) for x in p1]

            if random.random() < mutation_rate:
                area_idx = random.randint(0, len(areas) - 1)
                res_type = random.choice(["Food", "Water", "Medical"])
                cur_val = child[area_idx]["Allocated"].get(res_type, 0)
                delta = int(max(1, round(0.10 * (total_food if res_type == "Food" else (total_water if res_type == "Water" else total_med)))))
                change = random.randint(-delta, delta)
                new_val = max(0, min(cur_val + change, total_food if res_type == "Food" else total_water if res_type == "Water" else total_med))
                child[area_idx]["Allocated"][res_type] = new_val

            children.append(child)
        population = children

    final_scored = sorted([(fitness(ind), ind) for ind in population], key=lambda x: x[0], reverse=True)
    best_fitness, best_plan = final_scored[0]

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

# ---------------------- Route Planning ----------------------
def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest_neighbor_route(hq_coord, target_points):
    unvisited = target_points.copy()
    route = []
    total_dist = 0.0
    current = ("HQ", hq_coord)

    while unvisited:
        nearest_idx = None
        nearest_dist = float('inf')
        for idx, (name, coord) in enumerate(unvisited):
            d = euclidean(current[1], coord)
            if d < nearest_dist:
                nearest_dist = d
                nearest_idx = idx
        name, coord = unvisited.pop(nearest_idx)
        route.append((name, coord, nearest_dist))
        total_dist += nearest_dist
        current = (name, coord)

    return route, total_dist

def plot_route_map(hq_coord, areas, route_order, filename='static/route_map.png'):
    os.makedirs("static", exist_ok=True)
    plt.figure(figsize=(7, 6))

    for a in areas:
        name, x, y = a['Area'], a['x'], a['y']
        blocked = (a.get('Blocked') or 'No').lower() == 'yes'
        if blocked:
            plt.scatter(x, y, color='red', marker='x', s=100)
            plt.text(x + 0.4, y + 0.4, f"{name} (blocked)", fontsize=9)
        else:
            plt.scatter(x, y, color='green', s=80)
            plt.text(x + 0.4, y + 0.4, name, fontsize=9)

    plt.scatter(hq_coord[0], hq_coord[1], color='blue', marker='s', s=140)
    plt.text(hq_coord[0] + 0.4, hq_coord[1] + 0.4, "HQ", fontsize=10, fontweight='bold')

    coords = [hq_coord]
    for name in route_order:
        a = next((x for x in areas if x['Area'] == name), None)
        coords.append((a['x'], a['y']))
    xs, ys = [c[0] for c in coords], [c[1] for c in coords]
    plt.plot(xs, ys, '-k', linewidth=1.8, alpha=0.8)
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        plt.arrow(x1, y1, x2 - x1, y2 - y1, length_includes_head=True, head_width=0.6, head_length=1.0, fc='k', ec='k')

    plt.title('Route Map (HQ → ... → Last Area)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

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
            "Blocked": request.form.get(f'blk{idx}', "No"),
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

    # Run Genetic Algorithm
    allocations, fitness = ga_optimizer(areas, resources, generations=40, population_size=40)

    # Route Planning
    hq = (0.0, 0.0)
    unblocked_points = [(a["Area"], (a["x"], a["y"])) for a in areas if (a["Blocked"] or "No").lower() != "yes"]
    route_info, total_dist = nearest_neighbor_route(hq, unblocked_points)
    route_order = [r[0] for r in route_info]
    route_summary = [f"{n}: dist={d:.2f}" for n, _, d in route_info]
    route_summary.append(f"Total distance: {total_dist:.2f}")
    plot_route_map(hq, areas, route_order)

    return render_template(
        'index.html',
        allocations=allocations,
        fitness_score=fitness,
        fitness_plot='fitness_plot.png',
        route_plot='route_map.png',
        route_summary=route_summary,
        route_order=route_order
    )

if __name__ == '__main__':
    app.run(debug=True)

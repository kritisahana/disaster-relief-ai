from flask import Flask, render_template, request
import os
import random
import math
import matplotlib
matplotlib.use('Agg')  # important for server/macOS
import matplotlib.pyplot as plt

app = Flask(__name__)


# ---------------------- Genetic Algorithm (resource allocation) ----------------------
def ga_optimizer(areas, resources, generations=30, population_size=30):
    total_food = resources.get("Food", 0)
    total_water = resources.get("Water", 0)
    total_med = resources.get("Medical", 0)

    def random_plan():
        plan = []
        for _a in areas:
            plan.append({
                "Area": _a["Area"],
                "Allocated": {
                    "Food": random.randint(0, max(1, total_food)),
                    "Water": random.randint(0, max(1, total_water)),
                    "Medical": random.randint(0, max(1, total_med))
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
                delivery_penalty += 40  # stronger penalty for blocked areas

        return 0.5 * coverage + 0.3 * urgency_score - 0.2 * delivery_penalty

    # initial population
    population = [random_plan() for _ in range(population_size)]
    best_fitness_over_time = []

    for _gen in range(generations):
        scored = sorted([(fitness(p), p) for p in population], key=lambda x: x[0], reverse=True)
        best_fitness_over_time.append(scored[0][0])
        best_half = [p for _, p in scored[:max(2, population_size // 2)]]

        children = []
        while len(children) < population_size:
            p1, p2 = random.sample(best_half, 2)
            # single-point crossover
            if len(areas) > 1:
                cp = random.randint(1, len(areas) - 1)
            else:
                cp = 1
            child = p1[:cp] + p2[cp:]
            # mutation
            if random.random() < 0.2:
                area_idx = random.randint(0, len(areas) - 1)
                res_type = random.choice(["Food", "Water", "Medical"])
                bound = total_food if res_type == "Food" else (total_water if res_type == "Water" else total_med)
                child[area_idx]["Allocated"][res_type] = random.randint(0, max(1, bound))
            children.append(child)
        population = children

    best_fitness, best_plan = scored[0]

    # plot fitness curve
    os.makedirs("static", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(best_fitness_over_time) + 1), best_fitness_over_time, marker='o')
    plt.title('Fitness Improvement Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/fitness_plot.png')
    plt.close()

    return best_plan, best_fitness


# ---------------------- Route planning (nearest neighbor TSP-like) ----------------------
def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_neighbor_route(hq_coord, target_points):
    """
    Simple nearest neighbor heuristic:
    Start at HQ, repeatedly go to nearest unvisited point.
    target_points: list of tuples (name, (x,y))
    Returns route list of names in visiting order and total distance.
    """
    unvisited = target_points.copy()
    route = []
    total_dist = 0.0
    current = ("HQ", hq_coord)

    while unvisited:
        # find nearest
        nearest_idx = None
        nearest_dist = float('inf')
        for idx, (name, coord) in enumerate(unvisited):
            d = euclidean(current[1], coord)
            if d < nearest_dist:
                nearest_dist = d
                nearest_idx = idx
        # move to nearest
        name, coord = unvisited.pop(nearest_idx)
        route.append((name, coord, nearest_dist))
        total_dist += nearest_dist
        current = (name, coord)

    return route, total_dist


def plot_route_map(hq_coord, areas, route_order, filename='static/route_map.png'):
    """
    areas: list of area dicts (with x,y and Blocked)
    route_order: list of names in visiting order (only unblocked ones)
    """
    os.makedirs("static", exist_ok=True)
    plt.figure(figsize=(7, 6))

    # draw all points
    for a in areas:
        name = a['Area']
        x = a.get('x', 0.0)
        y = a.get('y', 0.0)
        blocked = (a.get('Blocked') or 'No').lower() == 'yes'
        if blocked:
            plt.scatter(x, y, marker='x', color='red', s=100)
            plt.text(x + 0.5, y + 0.5, f"{name} (blocked)", fontsize=9)
        else:
            plt.scatter(x, y, marker='o', color='green', s=80)
            plt.text(x + 0.5, y + 0.5, name, fontsize=9)

    # HQ
    plt.scatter(hq_coord[0], hq_coord[1], marker='s', color='blue', s=140)
    plt.text(hq_coord[0] + 0.5, hq_coord[1] + 0.5, "HQ", fontsize=10, fontweight='bold')

    # draw lines following route_order
    coords = [hq_coord]
    for name in route_order:
        a = next((x for x in areas if x['Area'] == name), None)
        coords.append((a['x'], a['y']))
    # draw piecewise lines
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    if len(coords) >= 2:
        plt.plot(xs, ys, '-k', linewidth=1.8, alpha=0.8)

        # draw arrows for direction
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            plt.arrow(x1, y1, x2 - x1, y2 - y1, length_includes_head=True,
                      head_width=0.8, head_length=1.2, fc='k', ec='k', alpha=0.8)

    plt.title('Route Map (HQ → ... → last area)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ---------------------- Flask routes ----------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/optimize', methods=['POST'])
def optimize():
    # read variable-length areas
    areas = []
    idx = 1
    while True:
        name = request.form.get(f'area{idx}')
        if not name:
            break
        pop = request.form.get(f'pop{idx}') or "0"
        need = request.form.get(f'need{idx}') or ""
        urg = request.form.get(f'urg{idx}') or "Low"
        blk = request.form.get(f'blk{idx}') or "No"
        # coordinates may be present but not required for allocation
        x = request.form.get(f'x{idx}')
        y = request.form.get(f'y{idx}')
        try:
            x = float(x) if x not in (None, "") else 0.0
            y = float(y) if y not in (None, "") else 0.0
        except ValueError:
            x, y = 0.0, 0.0

        areas.append({
            "Area": name,
            "Population": int(pop),
            "Need": need,
            "Urgency": urg,
            "Blocked": blk,
            "x": x,
            "y": y
        })
        idx += 1

    resources = {
        "Food": int(request.form.get('food') or 0),
        "Water": int(request.form.get('water') or 0),
        "Medical": int(request.form.get('medical') or 0)
    }

    if not areas:
        return render_template('index.html', message="Please add at least one area before optimizing.")

    allocations, fitness = ga_optimizer(areas, resources)

    # return with allocations and fitness plot
    return render_template('index.html',
                           allocations=allocations,
                           fitness_score=fitness,
                           fitness_plot='fitness_plot.png')


@app.route('/route', methods=['POST'])
def route():
    # read variable-length areas (with coordinates)
    areas = []
    idx = 1
    while True:
        name = request.form.get(f'area{idx}')
        if not name:
            break
        pop = request.form.get(f'pop{idx}') or "0"
        need = request.form.get(f'need{idx}') or ""
        urg = request.form.get(f'urg{idx}') or "Low"
        blk = request.form.get(f'blk{idx}') or "No"
        x = request.form.get(f'x{idx}')
        y = request.form.get(f'y{idx}')
        try:
            x = float(x)
            y = float(y)
        except (ValueError, TypeError):
            # if coordinates missing or invalid, default to 0,0
            x, y = 0.0, 0.0

        areas.append({
            "Area": name,
            "Population": int(pop),
            "Need": need,
            "Urgency": urg,
            "Blocked": blk,
            "x": x,
            "y": y
        })
        idx += 1

    if not areas:
        return render_template('index.html', message="Please add at least one area before finding route.")

    # HQ coordinate (you can change to user input if desired)
    hq = (0.0, 0.0)

    # build list of unblocked points with coordinates
    unblocked_points = []
    for a in areas:
        if (a.get('Blocked') or 'No').lower() != 'yes':
            unblocked_points.append((a['Area'], (a['x'], a['y'])))

    if not unblocked_points:
        return render_template('index.html', message="No unblocked areas to route to.", route_plot=None)

    # compute nearest-neighbor route from HQ (stops after last area)
    route_info, total_distance = nearest_neighbor_route(hq, unblocked_points)
    # route_info is list of (name, coord, dist_from_prev)
    route_order = [entry[0] for entry in route_info]

    # create human-readable route summary with distances
    # the first distance in route_info is from HQ to first area, etc.
    route_summary = []
    for name, coord, dist in route_info:
        route_summary.append(f"{name}: distance from previous = {dist:.2f}")

    route_summary.append(f"Total distance (HQ → ... → last): {total_distance:.2f}")

    # plot map and route
    plot_route_map(hq, areas, route_order, filename='static/route_map.png')

    return render_template('index.html',
                           route_plot='route_map.png',
                           route_order=route_order,
                           route_summary=route_summary)


if __name__ == '__main__':
    app.run(debug=True)

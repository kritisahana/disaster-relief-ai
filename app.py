from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')  # Prevents macOS GUI crash
import matplotlib.pyplot as plt
import random, os

app = Flask(__name__)

def ga_optimizer(areas, resources, generations=30, population_size=20):
    total_food = resources["Food"]
    total_water = resources["Water"]
    total_med = resources["Medical"]

    def random_plan():
        plan = []
        for a in areas:
            plan.append({
                "Area": a["Area"],
                "Need": a["Need"],
                "Allocated": {
                    "Food": random.randint(0, total_food),
                    "Water": random.randint(0, total_water),
                    "Medical": random.randint(0, total_med)
                }
            })
        return plan

    def fitness(plan):
        coverage = 0
        urgency_score = 0
        delivery_penalty = 0
        for p, a in zip(plan, areas):
            need = a["Need"].lower()
            urgency = a["Urgency"].lower()
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

            if a["Blocked"].lower() == "yes":
                delivery_penalty += 20

        return 0.5 * coverage + 0.3 * urgency_score - 0.2 * delivery_penalty

    population = [random_plan() for _ in range(population_size)]
    best_fitness_over_time = []

    for _ in range(generations):
        scored = sorted([(fitness(p), p) for p in population], key=lambda x: x[0], reverse=True)
        best_fitness_over_time.append(scored[0][0])
        best_half = [p for _, p in scored[:population_size//2]]

        children = []
        while len(children) < population_size:
            p1, p2 = random.sample(best_half, 2)
            crossover_point = random.randint(0, len(areas) - 1)
            child = p1[:crossover_point] + p2[crossover_point:]
            if random.random() < 0.2:
                area_idx = random.randint(0, len(areas) - 1)
                res_type = random.choice(["Food", "Water", "Medical"])
                child[area_idx]["Allocated"][res_type] = random.randint(0, total_food)
            children.append(child)
        population = children

    best_fitness, best_plan = scored[0]

    # Plot fitness curve
    plt.figure()
    plt.plot(best_fitness_over_time, marker='o', color='blue')
    plt.title('Fitness Improvement Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    os.makedirs('static', exist_ok=True)
    plt.savefig('static/fitness_plot.png')
    plt.close()

    return best_plan, best_fitness


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    # Collect variable-length area data
    areas = []
    area_names = request.form.getlist('area[]')
    pops = request.form.getlist('population[]')
    needs = request.form.getlist('need[]')
    urgencies = request.form.getlist('urgency[]')
    blocked = request.form.getlist('blocked[]')

    for i in range(len(area_names)):
        if area_names[i].strip() != "":
            areas.append({
                "Area": area_names[i],
                "Population": int(pops[i] or 0),
                "Need": needs[i],
                "Urgency": urgencies[i],
                "Blocked": blocked[i]
            })

    resources = {
        "Food": int(request.form.get("food") or 0),
        "Water": int(request.form.get("water") or 0),
        "Medical": int(request.form.get("medical") or 0)
    }

    if not areas:
        return render_template('index.html', message="⚠️ Please add at least one area.")

    allocations, fitness = ga_optimizer(areas, resources)
    return render_template(
        'index.html',
        message=f"Optimization Complete! (Fitness Score: {fitness:.2f})",
        allocations=allocations,
        plot='fitness_plot.png'
    )


if __name__ == '__main__':
    app.run(debug=True)

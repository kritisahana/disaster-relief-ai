from flask import Flask, render_template, request   # imports first
import random

app = Flask(__name__)                               # create the app first


import random
import matplotlib
matplotlib.use('Agg')   # Use non-GUI backend
import matplotlib.pyplot as plt


import random
import matplotlib.pyplot as plt

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
        best_fitness_over_time.append(scored[0][0])  # record best fitness this generation
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

    # plot the fitness curve
    plt.figure()
    plt.plot(best_fitness_over_time, marker='o')
    plt.title('Fitness Improvement Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.savefig('static/fitness_plot.png')  # save to static folder
    plt.close()

    return best_plan, best_fitness




@app.route('/')
def home():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    areas = []
    for i in range(1, 4):
        area = request.form.get(f'area{i}')
        pop = request.form.get(f'pop{i}')
        need = request.form.get(f'need{i}')
        urg = request.form.get(f'urg{i}')
        blk = request.form.get(f'blk{i}')
        if area:
            areas.append({
                "Area": area,
                "Population": int(pop or 0),
                "Need": need,
                "Urgency": urg,
                "Blocked": blk
            })

    resources = {
        "Food": int(request.form.get("food") or 0),
        "Water": int(request.form.get("water") or 0),
        "Medical": int(request.form.get("medical") or 0)
    }

    # 🔹 Run the simple GA and display results
    allocations, fitness = ga_optimizer(areas, resources)
    plot_path = 'static/fitness_plot.png'


    return render_template(
        'index.html',
        message=f"Optimized Allocation (fitness={fitness:.2f})",
        allocations=allocations,
        plot=plot_path
    )


if __name__ == '__main__':
    app.run(debug=True)




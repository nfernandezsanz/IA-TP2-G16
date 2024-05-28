import random
import numpy
import csv
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Definir las personas, tragos, colores y decoraciones
personas = ["Julia", "María", "Pedro", "Juan"]
tragos = ["Dulcinea", "Piel de serpiente", "Elixir", "Coco"]
colores = ["marrón", "verde", "naranja", "blanco"]
decoraciones = ["hojas de menta", "hielo", "mini sombrilla", "rodajas de naranja"]

# Crear el tipo de Fitness y el individuo
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Función para crear un individuo sin repeticiones
def create_individual():
    individual = list(zip(random.sample(personas, len(personas)),
                          random.sample(tragos, len(tragos)),
                          random.sample(colores, len(colores)),
                          random.sample(decoraciones, len(decoraciones))))
    return individual

# Función fitnes
def evaluate(individual):
    score = 0
    
    for persona, trago, color, decoracion in individual:
        
        # Reglas
        if color == "marrón" and decoracion == "hojas de menta":
            score += 2
        if color == "verde" and decoracion == "hielo":
            score += 2
        if persona == "María" and trago == "Dulcinea":
            score += 2
        if persona == "Pedro" and trago == "Piel de serpiente":
            score += 2
        if trago == "Elixir" and color == "naranja":
            score += 2
        if trago == "Coco" and color == "blanco":
            score += 2
        if persona == "Juan" and decoracion == "rodajas de naranja":
            score += 2

        # Penalizaciones
        if persona == "Julia" and color == "marrón":
            score -= 4
        if persona == "Julia" and decoracion == "hojas de menta":
            score -= 4
        if persona == "María" and (color == "verde" or decoracion == "hielo"):
            score -= 4
        if persona == "Juan" and trago == "Coco":
            score -= 4
        if (trago == "Elixir" or color == "naranja") and decoracion == "mini sombrilla":
            score -= 4

    return score,

# Función para reparar individuos después de cruzamiento y mutación
def repair(individual):
    seen_personas = set()
    seen_tragos = set()
    seen_colores = set()
    seen_decoraciones = set()

    missing_personas = set(personas)
    missing_tragos = set(tragos)
    missing_colores = set(colores)
    missing_decoraciones = set(decoraciones)

    for i, (persona, trago, color, decoracion) in enumerate(individual):
        if persona in seen_personas:
            individual[i] = (None, trago, color, decoracion)
        else:
            seen_personas.add(persona)
            missing_personas.discard(persona)
        
        if trago in seen_tragos:
            individual[i] = (persona, None, color, decoracion)
        else:
            seen_tragos.add(trago)
            missing_tragos.discard(trago)
        
        if color in seen_colores:
            individual[i] = (persona, trago, None, decoracion)
        else:
            seen_colores.add(color)
            missing_colores.discard(color)
        
        if decoracion in seen_decoraciones:
            individual[i] = (persona, trago, color, None)
        else:
            seen_decoraciones.add(decoracion)
            missing_decoraciones.discard(decoracion)
    
    for i, (persona, trago, color, decoracion) in enumerate(individual):
        if persona is None:
            individual[i] = (missing_personas.pop(), trago, color, decoracion)
        if trago is None:
            individual[i] = (persona, missing_tragos.pop(), color, decoracion)
        if color is None:
            individual[i] = (persona, trago, missing_colores.pop(), decoracion)
        if decoracion is None:
            individual[i] = (persona, trago, color, missing_decoraciones.pop())

    return individual

# Función de cruzamiento que mantiene la unicidad
def mate(ind1, ind2):
    tools.cxOnePoint(ind1, ind2)
    ind1[:] = repair(ind1)
    ind2[:] = repair(ind2)
    return ind1, ind2

# Función de mutación que mantiene la unicidad
def mutate(individual):
    tools.mutShuffleIndexes(individual, indpb=0.05)
    individual[:] = repair(individual)
    return individual,

# Inicializar el toolbox
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", mate)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=4)

# Ejecutar el algoritmo genético
def main():
    random.seed(55)
    npop = 100
    ngen = 100
    population = toolbox.population(n=npop)
    halloffame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", lambda vals: min(v[0] for v in vals))
    stats.register("avg", lambda vals: sum(v[0] for v in vals) / len(vals))
    
    
    population, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=npop, lambda_=npop, cxpb=0.7, mutpb=0.3, ngen=ngen, stats=stats, halloffame=halloffame)
    
    return population, halloffame, logbook

if __name__ == "__main__":
    pop, hof, log = main()
    for ind in hof:
        for persona, trago, color, decoracion in ind:
            print(f"{persona} tiene el trago {trago} de color {color} con {decoracion}")

    
    from datetime import datetime
     
    filename = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(10,8))
    front = numpy.array([(c['gen'], c['avg']) for c in log])
    plt.plot(front[:,0][1:-1], front[:,1][1:-1])
    plt.axis("tight")
    plt.savefig(f"{filename}.png")

    with open(f'{filename}.csv', mode='w', newline='') as log_file:
        log_writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow(['Generation', 'Avg'])
        for entry in log:
            log_writer.writerow([entry['gen'], entry['avg']])
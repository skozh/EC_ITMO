from draw_log import draw_logs
from functions import rastrigin
from ga_exp import *
from pso_exp import *


if __name__ == "__main__":
    def function(x):
        res = rastrigin(x)
        return res,
    dimension = 50
    pop_size = 100
    iterations = 2000
    ga = SimpleGAExperiment(function, dimension, pop_size, iterations)
    log = ga.run()

    pso = PSOAlg(pop_size, iterations, dimension, function)
    pop, logbook, best = pso.run()
    draw_logs(log, logbook, "ga", "pso")
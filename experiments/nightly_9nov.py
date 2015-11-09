from experiments.parameter_tester import parameter_tester
from experiments.parameter_tester import CONFIG

parameter_tester('ant_init', ['random', 'weighted', 'static', 'on_global_best', 'chance_of_global_best'])
parameter_tester('q_init', [CONFIG['q_max'], CONFIG['q_min']])
parameter_tester('decay_type', ['random_type', 'grad_type'])

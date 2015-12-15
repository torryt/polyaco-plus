from experiments.parameter_tester import parameter_tester


def ant_init():
    parameter_tester('ant_init', ['weighted', 'static', 'on_global_best'], data_set='rectangle')
    parameter_tester('ant_init', ['weighted', 'static', 'on_global_best'], data_set='circle')
    parameter_tester('ant_init', ['weighted', 'static', 'on_global_best'], data_set='semicircle')
# parameter_tester('tau_init', [CONFIG['tau_max'], CONFIG['tau_min']])
# parameter_tester('decay_type', ['probabilistic', 'gradual'])


def evaporation_strategy():
    parameter_tester('decay_type', ['random_type', 'grad_type'], data_set='rectangle')
    parameter_tester('decay_type', ['random_type', 'grad_type'], data_set='circle')
    parameter_tester('decay_type', ['random_type', 'grad_type'], data_set='semicircle')

evaporation_strategy()

from experiments.parameter_tester import parameter_tester
from experiments.parameter_tester import CONFIG


def ant_init():
    parameter_tester('ant_init', ['weighted', 'static', 'on_global_best'], data_set='rectangle')
    parameter_tester('ant_init', ['weighted', 'static', 'on_global_best'], data_set='circle')
    parameter_tester('ant_init', ['weighted', 'static', 'on_global_best'], data_set='semicircle')
# parameter_tester('q_init', [CONFIG['q_max'], CONFIG['q_min']])
# parameter_tester('decay_type', ['random_type', 'grad_type'])


def evap_strategy():
    # parameter_tester('decay_type', ['random_type', 'grad_type'], data_set='rectangle')
    parameter_tester('decay_type', ['random_type', 'grad_type'], data_set='circle')
    parameter_tester('decay_type', ['random_type', 'grad_type'], data_set='semicircle')

ant_init()
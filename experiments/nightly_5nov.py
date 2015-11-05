from experiments.parameter_tester import parameter_tester
from experiments.parameter_tester import CONFIG

parameter_tester('ant_init', ['random', 'weighted', 'static', 'on_global_best'])
parameter_tester('q_init', [CONFIG['q_max'], CONFIG['q_min']])
parameter_tester('q', [0.1, 1.0, 5.0, 10.0])
parameter_tester('q_min', [0.001, 0.01, 0.02, 0.1, 0.3])
parameter_tester('q_max', [1.0, 5.0, 10.0, 20.0, 40.0])
parameter_tester('rho', [0.001, 0.01, 0.02, 0.1, 0.3])
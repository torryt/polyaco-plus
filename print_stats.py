import pstats

stream = open('cpu_profile.txt', 'w');
stats = pstats.Stats('cpu_profile.pstat', stream=stream)
stats.strip_dirs()
stats.sort_stats('tottime')
stats.print_stats(20)

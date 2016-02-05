from acoc_runner import run

runs = 10

with_multi = []
no_multi = []
for i in range(runs):
    print("\nRun {}/{}\n".format(i, runs))
    with_multi.append(run(multi_level=True))
    no_multi.append(run(multi_level=False, granularity=17))

print("\nMean best result with multi-leveling: {}".format(sum(with_multi)/10))
print("Mean best result without multi-leveling: {}".format(sum(no_multi)/10))
import cProfile
from main import run


def profile():
    for _ in range(10):
        run()

cProfile.run('profile()', filename='profile_output', sort='cumtime')
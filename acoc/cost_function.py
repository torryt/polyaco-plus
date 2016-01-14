from numba import cuda


@cuda.jit
def cost_function_cuda(P, E, result):
    result[0] = 1

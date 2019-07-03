import multiprocessing


def perform_multiprocessed(initialize, init_args, function, fn_args, finalize, nproc=multiprocessing.cpu_count()):
    """Performs a given function multiprocessed

    :param initialize: a function which produces an initial state. Called by 
        each worker once when spawned
    :param init_args: arguments to pass to the initialization function
    :param function: a function to be performed many times in parallel. 
        (fn_arg) -> result_item 
    :param fn_args: a generator which produces the arguments for each call of
        function. yields fn_arg
    :param finalize: a function to use to wrap up the final results.
        (result_item[]) -> result
    :param nproc: number of processes. Defaults to the number of cpus
    :returns: result
    """
    with multiprocessing.Pool(nproc, initializer=initialize, initargs=init_args, maxtasksperchild=1000) as pool:
        result = pool.map(function, fn_args)
    return finalize(result)

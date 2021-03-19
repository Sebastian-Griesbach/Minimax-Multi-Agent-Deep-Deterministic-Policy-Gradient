def func1(param1, param2):
    return f"func_1_{param1}_{param2}"
def func2(param1, param2):
    return f"func_1_{param1}_{param2}"

func_list = [func1, func2]

list_param1 = [[1,2],[3,4]]
list_param2 = [["eins","zwei"],["drei", "vier"]]

def _execute_function_param_pairs(function_list, params):
    execute = lambda func, args: func(*args)

    return list(map(execute, function_list, params))

_execute_function_param_pairs(func_list, zip(list_param1, list_param2))
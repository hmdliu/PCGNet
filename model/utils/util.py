
def get_param_ids(module_list):
    param_ids = []
    for mo in module_list:
        ids = list(map(id, mo.parameters()))
        param_ids = param_ids + ids
    return param_ids
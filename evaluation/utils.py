from addict import Dict


METRIC_INDICATOR = {
    # camera pose metrics
    "auc": "↑",

    # depth metrics
    "Abs Rel": "↓",
    "Sq Rel": "↓",
    "RMSE": "↓",
    "δ": "↑",

    # point-cloud metrics
    "acc": "↓",
    "comp": "↓",
    "overall": "↓",
    "precision": "↑",
    "recall": "↑",
    "fscore": "↑",

    # efficiency metrics
    "latency": "↓",
    "mem": "↓",
}

def get_indicator(metric_name):
    for m, ind in METRIC_INDICATOR.items():
        if m in metric_name:
            return ind
    return ""


# ============================================================== #
# ========================== Printing ========================== #
# ============================================================== #
def print_dict(d: Dict, precision=4, indent=0, sort_keys=False):
    """
    Pretty print dict / addict.Dict with aligned keys and values.
    Each line: <key padded> : <value padded>
    """
    if not isinstance(d, dict):
        raise TypeError(f"Expected dict-like object, got {type(d)}")

    keys = sorted(d.keys()) if sort_keys else list(d.keys())
    pad = " " * indent

    # key width
    max_key_len = max(len(str(k)) for k in keys)

    # format float
    float_fmt = f"{{:.{precision}f}}"
    values_str = []

    # pre-format values to compute width
    for k in keys:
        v = d[k]
        if isinstance(v, (int, float)):
            s = float_fmt.format(float(v))
        else:
            s = str(v)
        values_str.append(s)

    max_val_len = max(len(s) for s in values_str)

    # print
    for k, v_str in zip(keys, values_str):
        print(f"{pad}{str(k):<{max_key_len}} : {v_str:>{max_val_len}}")


def add_indicator(d: Dict, remove_keys=[]):
    new_d = Dict()
    for k, v in d.items():
        # remove keys that are not metrics for display
        if k in remove_keys:
            continue
        
        # add indicator
        flag = False
        for m, ind in METRIC_INDICATOR.items():
            if m in k:
                new_k, flag = f"{k} ({ind})", True
        new_d[k if not flag else new_k] = v

    return new_d

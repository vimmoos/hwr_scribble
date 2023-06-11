from Levenshtein import distance as lev

from pathlib import Path


def eval_res(y_dir: str, target_dir: str = "data/test_results"):
    target_dir = Path(target_dir)
    y_dir = Path(y_dir)
    res = []

    for x in Path(target_dir).glob("*.txt"):
        if not x.is_file():
            continue
        with open(target_dir / x.name, "r") as f:
            s_target = f.read()
        with open(y_dir / x.name, "r") as f:
            s_y = f.read()
        res.append(lev(s_y, s_target))
    return res

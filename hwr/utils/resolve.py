import importlib


def split_var_spec(qualified_spec: str):
    if ":" not in qualified_spec:
        raise Exception(
            f"Invalid specifier '{qualified_spec}' does not contain ':'"
        )

    parts = qualified_spec.split(":")
    if len(parts) != 2:
        raise Exception(
            f"Invalid specifier '{qualified_spec}' has {len(parts)} parts"
        )

    (module, var) = parts

    if not var:
        raise Exception(
            f"You must specify <module>:<var>, '{qualified_spec}' is not valid!"
        )

    return module, var


def resolve_var(qualified_spec: str):
    module, var = split_var_spec(qualified_spec)
    _mod = importlib.import_module(module)

    if not hasattr(_mod, var):
        raise Exception(f"Module {_mod} does not contain var {var}")
    return getattr(_mod, var)


class VarRef(str):
    def __int__(self, value):
        super().__init__(value)
        # checks validity and throws if not
        split_var_spec(self)

    def resolve(self):
        return resolve_var(self)

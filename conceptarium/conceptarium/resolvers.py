import ast

from omegaconf import OmegaConf

from env import CACHE


def math_eval(node):
    # adapted from https://stackoverflow.com/a/9558001
    import ast
    import operator

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    match node:
        case ast.Constant(value) if isinstance(value, (int, float)):
            return value  # integer
        case ast.BinOp(left, op, right):
            return operators[type(op)](math_eval(left), math_eval(right))
        case ast.UnaryOp(op, operand):  # e.g., -1
            return operators[type(op)](math_eval(operand))
        case _:
            raise TypeError(node)


def register_custom_resolvers():
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))
    OmegaConf.register_new_resolver(
        "math",
        lambda expr: math_eval(ast.parse(expr, mode="eval").body),
    )
    OmegaConf.register_new_resolver(
        "cache", lambda path: str(CACHE.joinpath(path).absolute())
    )

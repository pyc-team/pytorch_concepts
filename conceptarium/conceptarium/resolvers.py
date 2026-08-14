"""Custom OmegaConf resolvers for Hydra configurations.

This module registers custom resolvers that can be used in YAML configuration
files to perform operations like math evaluation, tuple creation, and path 
resolution at configuration time.
"""

import ast

from omegaconf import OmegaConf

from env import CACHE


def math_eval(node):
    """Evaluate mathematical expressions from AST nodes.

    Safely evaluates mathematical expressions parsed as AST nodes. Supports
    basic arithmetic operations: +, -, *, /, //, **, unary minus, and the
    conditional ``a if cond else b``.

    Args:
        node: AST node representing a mathematical expression.

    Returns:
        int or float: Result of the evaluated expression.

    Raises:
        TypeError: If the node contains unsupported operations.

    Note:
        Adapted from https://stackoverflow.com/a/9558001
        This is safer than eval() as it only supports arithmetic operations.

    Example:
        >>> import ast
        >>> expr = ast.parse("2 + 3 * 4", mode="eval").body
        >>> math_eval(expr)
        14

        The conditional lets a config size something off a boolean flag.
        OmegaConf renders a bool as Python's ``True``/``False``, so an
        interpolated flag parses straight into this branch:

        >>> math_eval(ast.parse("(2 + 1) * (16 if True else 0)", mode="eval").body)
        48
        >>> math_eval(ast.parse("(2 + 1) * (16 if False else 0)", mode="eval").body)
        0
    """
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
        # bool is a subclass of int, so an interpolated `True`/`False` lands here
        # and can drive the conditional below.
        case ast.Constant(value) if isinstance(value, (int, float)):
            return value  # integer
        case ast.BinOp(left, op, right):
            return operators[type(op)](math_eval(left), math_eval(right))
        case ast.UnaryOp(op, operand):  # e.g., -1
            return operators[type(op)](math_eval(operand))
        case ast.IfExp(test, body, orelse):  # e.g., 16 if True else 0
            return math_eval(body) if math_eval(test) else math_eval(orelse)
        case _:
            raise TypeError(node)


def register_custom_resolvers():
    """Register custom OmegaConf resolvers for use in YAML configurations.
    
    Registers three custom resolvers:
    - as_tuple: Convert arguments to a tuple, e.g., ${as_tuple:1,2,3} -> (1,2,3)
    - math: Evaluate math expressions, e.g., ${math:"2 + 3 * 4"} -> 14
    - cache: Resolve paths relative to CACHE directory, 
             e.g., ${cache:models/checkpoints} -> /path/to/cache/models/checkpoints
    
    Example:
        In a YAML config file after calling register_custom_resolvers():
        
        >>> # config.yaml
        >>> dimensions: ${as_tuple:64,128,256}  # (64, 128, 256)
        >>> batch_size: ${math:"2 ** 5"}  # 32
        >>> checkpoint_dir: ${cache:checkpoints}  # /cache/path/checkpoints
    """
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))
    OmegaConf.register_new_resolver(
        "math",
        lambda expr: math_eval(ast.parse(expr, mode="eval").body),
    )
    OmegaConf.register_new_resolver(
        "cache", lambda path: str(CACHE.joinpath(path).absolute())
    )

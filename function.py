from latex2sympy2 import latex2sympy
from sympy import symbols, sympify, lambdify
from typing import Union


class Function:
    def __init__(self, f: Union[callable, str], name: str = None):
        if isinstance(f, str):
            f = self.traduce_LaTeX(f)
        self.f = f
        self.name = name

    def __call__(self, z):
        return self.f(z)

    def __repr__(self):
        return self.name if self.name else self.f.__name__

    def __str__(self) -> str:
        return self.__repr__()

    def traduce_LaTeX(self, latex: str) -> str:
        z, i = symbols("z i")
        expr = latex2sympy(latex)
        expr = sympify(expr).subs(i, 1j)
        return lambdify(z, expr)

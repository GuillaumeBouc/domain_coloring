from PIL import Image
from typing import List
import numpy as np
import timeit

from make_circle import generate_grid, map_grid_to_colors
from function import Function


def domain_coloring(
    func: Function,
    size: int = 500,
    intervals: List[List[float]] = [[-1, 1], [-1, 1]],
    value_intervals: List[float] = [0.1, 0.2, 0.4, 0.8, 1.6],
    value_low: float = 0.5,
    value_high: float = 1.0,
    alpha: float = 0.5,
) -> Image:
    """
    Perform domain coloring for a given complex function.

    Args:
        func (Function): Complex function to be visualized.
        size (int): Size of the grid.
        value_intervals (List[float]): List of radii for the linear increase intervals.
        value_low (float): Low value for interpolation.
        value_high (float): High value for interpolation.
        alpha (float): Alpha value for blending white lines.

    Returns:
        Image: Domain-colored image.
    """
    # Generate grid of complex numbers
    grid = generate_grid(size, intervals)

    # Apply the function to each element of the grid
    complex_grid = np.vectorize(func.f)(grid)

    # Map grid of complex numbers to RGBA colors
    image_data = map_grid_to_colors(
        complex_grid, value_intervals, value_low, value_high, alpha
    )

    # Convert to PIL image
    image = Image.fromarray((image_data * 255).astype(np.uint8), mode="RGBA")
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(f"images/{func.name}.png")
    return image


expr_f1 = r"\frac{(z^2 - 1)(z - 2 - i)^2}{z^2 + 2 + i}"


def f1(z):
    return (z**2 - 1) * (z - 2 - 1j) ** 2 / (z**2 + 2 + 1j)


expr_f2 = r"\frac{z^3 + i}{z^2 + i}"


def f2(z):
    return (z**3 + 1j) / (z**2 + 1j)


expr_f3 = r"\frac{e^{z^2} + z^3}{\sinh(z + i)}"


def f3(z):
    return (np.exp(z**2) + z**3) / np.sinh(z + 1j)


expr_f4 = r"e^{\sin(z)} + \cos(z) - z"


def f4(z):
    return np.exp(np.sin(z)) + np.cos(z) - z


expr_f5 = r"z^2"


def f5(z):
    return z**2


timer = timeit.Timer(
    lambda: domain_coloring(
        Function(expr_f5, "f5"),
        1024,
        [[-5, 5], [-5, 5]],
        [2**n for n in range(-3, 8, 1)],
        alpha=0.8,
    )
)
print(f"Time taken: {timer.timeit(1):.2f} seconds")

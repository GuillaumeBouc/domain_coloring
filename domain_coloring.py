from PIL import Image
from typing import List
import numpy as np
import time

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


time_start = time.time()

expr = r"\frac{(z^2 - 1)(z - 2 - i)^2}{z^2 + 2 + i}"

domain_coloring(
    Function(expr, "f1"),
    2048,
    [[-2.5, 2.5], [-2.5, 2.5]],
    [2**n for n in range(-3, 8, 1)],
    alpha=0.8,
)
print(f"Time taken: {time.time() - time_start:.2f} seconds")

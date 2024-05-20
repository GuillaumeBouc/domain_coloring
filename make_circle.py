import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def calculate_value(
    magnitude: float, radii: List[float], value_low: float, value_high: float
) -> float:
    """
    Calculates value based on magnitude with multiple linear increases.

    Args:
        magnitude (float): Magnitude of the complex number.
        radii (List[float]): List of radii for the linear increase intervals.
        value_low (float): Low value for interpolation.
        value_high (float): High value for interpolation.

    Returns:
        float: Calculated value.
    """
    value = value_low
    for radius in radii:
        if magnitude <= radius:
            value = value_low + (magnitude / radius) * (value_high - value_low)
            break
        else:
            value = value_high
    return value


def complex_to_rgba(
    z: complex, radii: List[float], value_low: float, value_high: float, alpha: float
) -> np.ndarray:
    """
    Converts a complex number to an RGBA color based on its angle (hue).

    Args:
        z (complex): Complex number.
        radii (List[float]): List of radii for the linear increase intervals.
        value_low (float): Low value for interpolation.
        value_high (float): High value for interpolation.
        alpha (float): Alpha value for blending white lines.

    Returns:
        np.ndarray: RGBA color.
    """
    hue = np.angle(z) / (2 * np.pi) + 0.5  # Map angle to [0, 1]
    magnitude = np.abs(z)
    value = calculate_value(magnitude, radii, value_low, value_high)
    saturation = 1.0  # Keep saturation constant at 1 for simplicity

    hsv = np.array([hue, saturation, value])
    rgb = hsv_to_rgb(hsv)
    rgba = np.append(rgb, 1.0)  # Set alpha to 1.0 by default

    angle = np.angle(z)
    angle_degrees = np.degrees(angle) % 360
    if np.any(np.isclose(angle_degrees % 30, 0, atol=2)):
        rgba = blend_with_white(rgba, alpha)

    return rgba


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """
    Converts an HSV color to an RGB color.

    Args:
        hsv (np.ndarray): HSV color.

    Returns:
        np.ndarray: RGB color.
    """
    h, s, v = hsv
    i = int(h * 6.0)  # Assume hue is [0, 1)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0:
        return np.array([v, t, p])
    if i == 1:
        return np.array([q, v, p])
    if i == 2:
        return np.array([p, v, t])
    if i == 3:
        return np.array([p, q, v])
    if i == 4:
        return np.array([t, p, v])
    if i == 5:
        return np.array([v, p, q])
    return np.array([1, 1, 1])


def blend_with_white(rgba: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blends the given RGBA color with white based on the specified alpha.

    Args:
        rgba (np.ndarray): Original RGBA color.
        alpha (float): Alpha value for blending.

    Returns:
        np.ndarray: Blended RGBA color.
    """
    white = np.array([1, 1, 1, 1])
    return rgba * (1 - alpha) + white * alpha


def generate_grid(
    size: int, intervals: List[List[float]] = [[-1, 1], [-1, 1]]
) -> np.ndarray:
    """
    Generates a grid of complex numbers.

    Args:
        size (int): Size of the grid.

    Returns:
        np.ndarray: Grid of complex numbers.
    """
    x = np.linspace(intervals[0][0], intervals[0][1], size)
    y = np.linspace(intervals[1][0], intervals[1][1], size)
    xv, yv = np.meshgrid(x, y)
    return xv + 1j * yv


def map_grid_to_colors(
    grid: np.ndarray,
    radii: List[float],
    value_low: float,
    value_high: float,
    alpha: float,
) -> np.ndarray:
    """
    Maps a grid of complex numbers to RGBA colors.

    Args:
        grid (np.ndarray): Grid of complex numbers.
        radii (List[float]): List of radii for the linear increase intervals.
        value_low (float): Low value for interpolation.
        value_high (float): High value for interpolation.
        alpha (float): Alpha value for blending white lines.

    Returns:
        np.ndarray: Grid of RGBA colors.
    """

    shape = grid.shape
    image = np.zeros((shape[0], shape[1], 4))  # Include alpha channel
    for i in range(shape[0]):
        for j in range(shape[1]):
            image[i, j] = complex_to_rgba(
                grid[i, j], radii, value_low, value_high, alpha
            )
    return image


def display_chromatic_circle(
    size: int = 500,
    intervals: List[List[float]] = [[-1, 1], [-1, 1]],
    radii: List[float] = [0.0625, 0.125, 0.25, 0.5, 1, 2],
    value_low: float = 0.5,
    value_high: float = 1.0,
    alpha: float = 0.8,
):
    """
    Displays the chromatic circle with the given size and parameters.

    Args:
        size (int): Size of the grid.
        radii (List[float]): List of radii for the linear increase intervals.
        value_low (float): Low value for interpolation.
        value_high (float): High value for interpolation.
        alpha (float): Alpha value for blending white lines.
    """
    grid = generate_grid(size, intervals)
    image = map_grid_to_colors(grid, radii, value_low, value_high, alpha)
    plt.imshow(
        image,
        extent=(intervals[0][0], intervals[0][1], intervals[1][0], intervals[1][1]),
    )
    plt.title("Chromatic Circle for Domain Coloring with Blended White Lines")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.axis("off")  # Optional: Turn off axis
    plt.show()


if __name__ == "__main__":
    display_chromatic_circle()

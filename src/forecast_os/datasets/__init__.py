"""Bundled datasets: seeded synthetic generators and AirPassengers."""

from .air_passengers import load_air_passengers
from .synthetic import generate_returns, generate_series

__all__ = ["generate_series", "generate_returns", "load_air_passengers"]

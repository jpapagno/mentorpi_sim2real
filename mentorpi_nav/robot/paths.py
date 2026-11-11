"""Paths relative to the checked-out project, on host or in the ROS container."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"deployment/config"

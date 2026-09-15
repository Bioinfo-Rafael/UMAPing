"""凍結reference trajectoryを読み取るrepulsion teacher。"""
from .base import Teacher, TrainingQueries, exact_field
from .uniform import UniformMC
from .dual_importance import DualImportance, hubness_vector
from .dual_topl import DualTopL
from .barnes_hut import BarnesHut
from .grid_field import GridField

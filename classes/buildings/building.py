from shapes import *
from dataclasses import dataclass

@dataclass(frozen=True, order=True)   
class Building:
    position: tuple
    shape: Shape
    resources = [0]*8
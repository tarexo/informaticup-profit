from shapes import *
from .building import Building

class Obstacle(Building):
    def __init__(self, position, width, height):
        super().__init__(position, obstacle_shape(width, height))
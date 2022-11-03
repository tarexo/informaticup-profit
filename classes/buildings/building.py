from shapes import *


class Building:
    """The base class for all objects of the game (excluding the grid itself)

    Attributes
    ----------
    position : tuple
        The position of the building in (x,y)
    shape : Shape
        The shape of the building
    resources : list
        The resources currently held by the building
    """

    def __init__(self, position, shape):
        self.position = position
        self.shape = shape
        self.resources = [0] * 8
        self.connections = []

    def get_output_positions(self):
        return self.get_element_positions("-")

    def get_input_positions(self):
        return self.get_element_positions("+")

    def get_element_positions(self, target_element):
        element_positions = []
        for (x_offset, y_offset, element) in iter(self.shape):
            if element == target_element:
                pos = (self.position[0] + x_offset, self.position[1] + y_offset)
                element_positions.append(pos)

        return element_positions

    def add_connection(self, building):
        self.connections.append(building)
        # print(f"connecting {building} to {self}")

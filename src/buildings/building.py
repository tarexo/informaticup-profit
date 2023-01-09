import numpy as np


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

    # PlacableBuilding Constructor
    def __init__(self, position, subtype, shape=None):
        self.x = position[0]
        self.y = position[1]
        self.subtype = subtype

        if shape is None:
            from helper.building_shapes import BUILDING_SHAPES

            shape = BUILDING_SHAPES[type(self)][subtype]

        self.shape = shape
        self.resources = np.array([0] * 8)
        self.resource_cache = np.array([0] * 8)
        self.clear_connections()

    def is_placeable(self):
        return True

    def get_output_positions(self):
        return self.get_element_positions("-")

    def get_input_positions(self):
        return self.get_element_positions("+")

    def get_element_positions(self, target_element):
        element_positions = []
        for (x_offset, y_offset, element) in iter(self.shape):
            if element == target_element:
                pos = (self.x + x_offset, self.y + y_offset)
                element_positions.append(pos)

        return np.array(element_positions)

    def clear_connections(self):
        self.connections = []

    def add_connection(self, other_building):
        self.connections.append(other_building)

    def remove_connection(self, other_building):
        self.connections.remove(other_building)

    @classmethod
    def from_input_position(BuildingClass, x, y, subtype):
        """Create a Building using its input position (not its center)
        This function is invalid for Obstacles and Deposits as they do not have an input element
        TODO multiple combiner/factory inputs possible (only the very first one is used for now)

        Args:
            BuildingClass (cls): the class of the building that shall be placed
            x (int): x-position where the input element shall be placed
            y (int): y-position where the input element shall be placed
            subtype (int): its subtype

        Returns:
            building: a building placed at input position x and y
        """
        from helper.building_shapes import BUILDING_SHAPES

        shape = BUILDING_SHAPES[BuildingClass][subtype]

        for (x_offset, y_offset, element) in iter(shape):
            if element == "+":
                center_x = x - x_offset
                center_y = y - y_offset
                return BuildingClass((center_x, center_y), subtype)

        raise RuntimeError("Unexpected Behavior in 'Building.from_input_position()'")

    def __iter__(self):
        """iterate over the individual non-empty shape elements starting from the left upper corner

        Returns:
            Iterator(Shape): shape iterator
        """
        self._shape_iterator = iter(self.shape)
        return self

    def __next__(self):
        """return the next non-empty(!) element of the shape iterator and increment the offsets according to the next element's position

        Raises:
            StopIteration: stops after all elements have been iterated over

        Returns:
            tuple: (x, y, element) tile position in the grid
        """
        (x_offset, y_offset, element) = next(self._shape_iterator)
        return (self.x + x_offset, self.y + y_offset, element)

    def __str__(self) -> str:
        return f"{type(self).__name__}_{self.subtype} at x={self.x}, y={self.y}, \n{self.shape}\n"

    def to_json(self):
        raise NotImplementedError

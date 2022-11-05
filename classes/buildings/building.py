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
    def __init__(self, x, y, subtype, shape=None):
        self.x = x
        self.y = y
        self.subtype = subtype

        if shape is None:
            from helper.dicts.building_shapes import BUILDING_SHAPES

            shape = BUILDING_SHAPES[type(self)][subtype]

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
                pos = (self.x + x_offset, self.y + y_offset)
                element_positions.append(pos)

        return element_positions

    def add_connection(self, building):
        self.connections.append(building)
        # print(f"connecting {building} to {self}")

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
        from helper.dicts.building_shapes import BUILDING_SHAPES

        shape = BUILDING_SHAPES[BuildingClass][subtype]

        for (x_offset, y_offset, element) in iter(shape):
            if element == "+":
                center_x = x - x_offset
                center_y = y - y_offset
                return BuildingClass(center_x, center_y, subtype)

        raise RuntimeError("Unexpected Behavior in 'Building.from_input_position()'")

    def __repr__(self) -> str:
        return f"{type(self).__name__}_{self.subtype} at x={self.x}, y={self.y}, \n{self.shape}\n"

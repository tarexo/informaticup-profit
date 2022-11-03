from dataclasses import dataclass
import numpy as np


@dataclass(order=True)
class Shape:
    """The shape function defines the position aswell as the shape (width, height) of any building.

    Attributes
    ----------
    center : tuple
        The position of the object in (x,y) (Is not neccessarily the center of the object!)
    elements : np.ndarray
        The shape of the object, how it is presented in the grid
    """

    center: tuple
    elements: np.ndarray

    def __iter__(self):
        """iterate over the individual non-empty shape elements starting from the left upper corner

        Returns:
            Iterator(Shape): shape iterator
        """
        self.x_offset = 0
        self.y_offset = 0
        return self

    def __next__(self):
        """return the next non-empty(!) element of the shape iterator and increment the offsets according to the next element's position

        Raises:
            StopIteration: stops after all elements have been iterated over

        Returns:
            tuple: (x_offset, y_offset, element) grid position offsets relative to the center of the building
        """
        height, width = self.elements.shape
        if self.y_offset >= height:
            raise StopIteration

        # current tile position and element
        tile = (
            self.x_offset - self.center[0],
            self.y_offset - self.center[1],
            self.elements[self.y_offset, self.x_offset],
        )

        # increment offsets for next iteration
        if self.x_offset >= width - 1:
            self.x_offset = 0
            self.y_offset += 1
        else:
            self.x_offset += 1

        # if tile is empty return next element instead
        if tile[-1] == " ":
            return self.__next__()

        return tile

    def __repr__(self):
        return f"\n{self.elements}"

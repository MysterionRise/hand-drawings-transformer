import numpy as np
from corner_finder import closest_node


def test_closest_node():
    # testing that the function returns the correct node
    node = closest_node((0, 0), [(0, 0), (1, 1), (2, 2)])
    np.testing.assert_array_equal(node, np.asarray((0, 0)))

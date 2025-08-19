from tilelang import tvm as tvm
from tvm.tir import PrimExpr


def index_to_coordinates(index, shape) -> list[PrimExpr]:
    """
    Convert a flat (linear) index to multi-dimensional coordinates for a given shape.

    Example:
        shape = (4, 5, 6)
        index = 53
        index_to_coordinates(53, (4, 5, 6)) -> [1, 3, 5]
        # Explanation:
        # 53 // (5*6) = 1  (1st coordinate)
        # 53 % (5*6) = 23
        # 23 // 6 = 3      (2nd coordinate)
        # 23 % 6 = 5       (3rd coordinate)

    Args:
        index (int): The flat index to convert.
        shape (tuple or list of int): The shape of the multi-dimensional array.

    Returns:
        list: A list of coordinates corresponding to each dimension.
    """
    coordinates = []
    dims = len(shape)
    for i in range(dims):
        coordinates.append(index % shape[dims - i - 1])
        index = index // shape[dims - i - 1]
    coordinates.reverse()
    return coordinates


def linear_index(*args: PrimExpr) -> PrimExpr:
    """
    Convert a list of coordinates to a flat (linear) index using strides.

    Usage examples:
        linear_index(i)                         -> i
        linear_index(i, j)                      -> i * stride + j
        linear_index(i, j, stride_j)            -> i * stride_j + j
        linear_index(i, j, k, stride_j, stride_k)
                                                -> i * stride_j * stride_k + j * stride_k + k

        Example for index = i * threads * local_size + tx * local_size + v:
        Suppose you have i, tx, v as coordinates, and threads, local_size as strides:
        linear_index(i, tx, v, threads, local_size) == i * threads * local_size + tx * local_size + v
    """
    n = len(args)
    if n == 0:
        raise ValueError("At least one index is required")

    if n == 1:
        return args[0]

    # The first part is indices, the second part is strides (starting from the second dimension)
    # A simpler way: the number of strides = total number of arguments - number of indices
    # Actually, the args are designed as indices... + strides..., and the number of strides = number of indices - 1
    num_coords = (n + 1) // 2
    coords = args[:num_coords]
    strides = args[num_coords:]

    if len(strides) != len(coords) - 1:
        raise ValueError("Stride count must be one less than coordinate count")

    linear = coords[0]
    for idx, stride in zip(coords[1:], strides):
        linear = linear * stride + idx
    return linear

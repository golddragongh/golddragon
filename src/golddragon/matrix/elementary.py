import torch

def rowswap(matrix, row1, row2):
    my_matrix = matrix.clone().float()

    temp = my_matrix[row1].clone()
    my_matrix[row1] = my_matrix[row2]
    my_matrix[row2] = temp

    return my_matrix


def rowscale(matrix, row, scale):
    my_matrix = matrix.clone().float()
    my_matrix[row] = scale * my_matrix[row]

    return my_matrix


def rowreplacement(matrix, source_row, target_row, j, k):
    my_matrix = matrix.clone().float()

    my_matrix[target_row] = (
        j * my_matrix[source_row]
        + k * my_matrix[target_row]
    )

    return my_matrix


def rref(matrix):
    my_matrix = matrix.clone().float()

    number_of_rows = my_matrix.shape[0]
    number_of_columns = my_matrix.shape[1]

    pivot_row = 0

    for column in range(number_of_columns):

        if pivot_row >= number_of_rows:
            break

        row_with_pivot = pivot_row

        while (row_with_pivot < number_of_rows and my_matrix[row_with_pivot, column] == 0):
            row_with_pivot += 1

        if row_with_pivot == number_of_rows:
            continue

        if row_with_pivot != pivot_row:
            my_matrix = rowswap(my_matrix, pivot_row, row_with_pivot)

        pivot = my_matrix[pivot_row, column]

        my_matrix = rowscale(my_matrix, pivot_row, 1 / pivot)

        for row in range(number_of_rows):

            if row != pivot_row:
                value = my_matrix[row, column]

                my_matrix = rowreplacement(
                    my_matrix,
                    pivot_row,
                    row,
                    -value,
                    1
                )

        pivot_row += 1

    return my_matrix

# I use the three elementary row operations to compute the RREF of a matrix. The three elementary row operations are:
# 1. Row swapping: Swap two rows of the matrix.
# 2. Row scaling: Multiply a row by a non-zero scalar.
# 3. Row replacement: Replace a row by the sum of that row and a scalar
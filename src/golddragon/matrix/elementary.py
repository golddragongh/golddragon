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

if __name__ == "__main__":

    matrix = torch.tensor([
        [1, 3, 0, 0, 3],
        [0, 0, 1, 0, 9],
        [0, 0, 0, 1, -4]
    ])

    print("Original matrix:")
    print(matrix)

    matrix1 = rowswap(matrix, 0, 1)
    print('\n')
    print("After rowsap")
    print(matrix1)

    matrix2 = rowscale(matrix1, 0, 1/3)
    print('\n')
    print("After R1 = (1/3)R1:")
    print(matrix2)

    matrix3 = rowreplacement(matrix2, 0, 2, -3, 1)
    print('\n')
    print("After R3 = -3R1 + R3:")
    print(matrix3)

    print('\n')
    print("RREF of original matrix:")
    print(rref(matrix))
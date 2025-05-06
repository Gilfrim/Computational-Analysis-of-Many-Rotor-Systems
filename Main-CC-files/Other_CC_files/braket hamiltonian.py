from itertools import product


def generate_matrix(site, state):
    states = ["".join(map(str, digits)) for digits in product(range(state), repeat=site)]
    matrix = []

    for row in states:
        row_entries = [f"{row}|{col}" for col in states]
        matrix.append(row_entries)

    return matrix


def print_matrix(matrix):
    for row in matrix:
        print(" ".join(row))


# Example usage
site = 3
state = 2
matrix = generate_matrix(site, state)
print_matrix(matrix)

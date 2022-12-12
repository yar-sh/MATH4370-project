from decimal import *
from dataclasses import dataclass, field
import random
import math
from typing import Union, List, Optional

HIGH_PRECISION = 8
LOW_PRECISION = 4

getcontext().rounding = ROUND_HALF_EVEN

def D(x: Union[float, Decimal, int, str]) -> Decimal:
    return getcontext().create_decimal_from_float(float(x))

@dataclass
class Matrix:
    rows: int
    cols: int
    data: List[List[Decimal]] = field(default_factory=list)

@dataclass
class System:
    size: int
    a: Matrix
    x: Matrix
    b: Matrix

def get_rand_nonzero_coef() -> Decimal:
    while (k := D(random.randrange(-9, 10))) == D(0):
        pass

    return k

def mat_print(m: Matrix) -> None:
    print(f'{m.rows}x{m.cols} matrix:')
    for i in range(m.rows):
        print(''.join([str(x).ljust(HIGH_PRECISION+6) for x in m.data[i]]))
    print('')

def mat_zero(rows: int, cols: int) -> Matrix:
    r = Matrix(rows, cols)

    r.data = [[D(0) for j in range(cols)] for i in range(rows)]

    return r

def mat_id(size: int) -> Matrix:
    r = mat_zero(size, size)

    for i in range(size):
        r.data[i][i] = D(1)

    return r

def mat_create(values: List[List[Decimal]]) -> Matrix:
    r = Matrix(len(values), len(values[0]))

    r.data = [[D(values[i][j]) for j in range(r.cols)] for i in range(r.rows)]

    return r

def mat_random(rows: int, cols: int, seed: Optional[int]=None) -> Matrix:
    if seed:
        random.seed(seed)

    r = Matrix(rows, cols)

    r.data = [[D(random.randrange(-999, 1000)) for j in range(cols)] for i in range(rows)]

    return r

def mat_copy(m: Matrix) -> Matrix:
    # WILL NOT RETURN THE EXACT COPY!!! It re-casts all Decimals to match the current precision
    r = Matrix(m.rows, m.cols)

    r.data = [[D(m.data[i][j]) for j in range(m.cols)] for i in range(m.rows)]

    return r

def mat_diag(m: Matrix) -> Matrix:
    # Assumes square matrix
    r = mat_zero(m.rows, m.rows)

    for i in range(m.rows):
        r.data[i][i] = m.data[i][i]

    return r

def mat_lower(m: Matrix) -> Matrix:
    # Assumes square matrix
    r = mat_zero(m.rows, m.rows)

    for i in range(m.rows):
        for j in range(i + 1):
            r.data[i][j] = m.data[i][j]

    return r

def mat_upper(m: Matrix) -> Matrix:
    # Assumes square matrix
    r = mat_zero(m.rows, m.rows)

    for i in range(m.rows):
        for j in range(i, m.rows):
            r.data[i][j] = m.data[i][j]

    return r

def mat_transpose(m: Matrix) -> Matrix:
    r = Matrix(m.cols, m.rows)

    r.data = [[m.data[j][i] for j in range(m.rows)] for i in range(m.cols)]

    return r

def mat_mult_scalar(m: Matrix, k: D) -> Matrix:
    r = Matrix(m.rows, m.cols)

    r.data = [[k * m.data[i][j] for j in range(m.cols)] for i in range(m.rows)]

    return r

def mat_add(m1: Matrix, m2: Matrix) -> Matrix:
    r = Matrix(m1.rows, m1.cols)

    r.data = [[
        m1.data[i][j] + m2.data[i][j]
        for j in range(m1.cols)]
    for i in range(m1.rows)]

    return r

def mat_mult(m1: Matrix, m2: Matrix) -> Matrix:
    r = Matrix(m1.rows, m2.cols)

    r.data = [[
        sum([m1.data[i][k] * m2.data[k][j] for k in range(m1.cols)])
        for j in range(m2.cols)]
    for i in range(m1.rows)]

    return r

def mat_elementary(size: int, seed: Optional[int]=None) -> Matrix:
    # Source: https://en.wikipedia.org/wiki/Elementary_matrix#Elementary_row_operations
    if seed:
        random.seed(seed)

    r = mat_id(size)

    def make_row_switch_transformation() -> None:
        r1, r2 = random.choices(range(size), k=2)

        r.data[r1][r1] = D(0)
        r.data[r2][r2] = D(0)
        r.data[r1][r2] = D(1)
        r.data[r2][r1] = D(1)

    def make_row_multiply_transformation() -> None:
        r1 = random.randrange(size)
        r.data[r1][r1] = get_rand_nonzero_coef()

    def make_row_addition_transformation() -> None:
        r1, r2 = random.choices(range(size), k=2)
        r.data[r1][r2] = get_rand_nonzero_coef()

    # We are not doing row swaps so that we can avoid 0 entries on the diagonal - lead to better
    # examples
    random.choice([
        #make_row_switch_transformation,
        make_row_multiply_transformation,
        make_row_addition_transformation,
    ])()

    return r

def sys_random(size: int, seed: Optional[int]=None) -> System:
    if seed:
        random.seed(seed)

    a = mat_id(size)
    x = mat_random(size, 1)

    num_elementaries = random.randrange(4 * size, 6 * size)
    for i in range(num_elementaries):
        a = mat_mult(a, mat_elementary(size))

    b = mat_mult(a, x)

    return System(size, a, x, b)

def sys_print(s: System) -> None:
    print(f'System of {s.size} equations:')

    # Always print systems in high precision
    prev_prec = getcontext().prec
    getcontext().prec = HIGH_PRECISION

    mat_print(s.a)

    print('Exact solution x:')
    mat_print(mat_transpose(s.x))

    print('Vector b:')
    mat_print(mat_transpose(s.b))

    getcontext().prec = prev_prec

def sys_augmented(s: System) -> Matrix:
    r = mat_copy(s.a)
    r.cols += 1

    for i in range(s.size):
        r.data[i].append(D(s.b.data[i][0]))

    return r

def backwards_substitution(m: Matrix) -> Matrix:
    r = mat_zero(m.rows, 1)
    for j in range(m.rows - 1, -1, -1):
        r.data[j][0] = m.data[j][m.rows] / m.data[j][j]

        for k in range(j - 1, -1, -1):
            m.data[k][m.rows] -= m.data[k][j] * r.data[j][0]

    return r

def sys_solve_gaussian_elimination(s: System) -> Matrix:
    # Source: https://en.wikipedia.org/wiki/Gaussian_elimination#Pseudocode
    r = sys_augmented(s)
    h = 0
    k = 0

    while h < r.rows and k < r.cols:
        # If the pivot is 0 we swap it with first row with non-zero pivot
        if r.data[h][k] == D(0):
            for i in range(h + 1, r.rows):
                if r.data[i][k] != D(0):
                    temp = r.data[i]
                    r.data[i] = r.data[h]
                    r.data[h] = temp
                    break

        if r.data[h][k] == D(0):
            raise Exception('Potential division by 0 due to precision loss. Terminating.')

        for i in range(h + 1, r.rows):
            f = r.data[i][k] / r.data[h][k]

            r.data[i][k] = D(0)
            for j in range(k + 1, r.cols):
                r.data[i][j] = r.data[i][j] - r.data[h][j] * f

        print('Next gaussian step:')
        mat_print(r)

        h += 1
        k += 1

    return backwards_substitution(r)

def sys_solve_partial_pivoting(s: System) -> Matrix:
    r = sys_augmented(s)
    h = 0
    k = 0

    while h < r.rows and k < r.cols:
        swap_idx = 0
        for i in range(h + 1, r.rows):
            pivot = D(math.fabs(r.data[i][k]))
            if pivot != D(0) and (not swap_idx or D(math.fabs(r.data[swap_idx][k])) > pivot):
                swap_idx = i

        if swap_idx:
            temp = r.data[swap_idx]
            r.data[swap_idx] = r.data[h]
            r.data[h] = temp

        if r.data[h][k] == D(0):
            raise Exception('Potential division by 0 due to precision loss. Terminating.')

        for i in range(h + 1, r.rows):
            f = r.data[i][k] / r.data[h][k]

            r.data[i][k] = D(0)
            for j in range(k + 1, r.cols):
                r.data[i][j] = r.data[i][j] - r.data[h][j] * f

        print('Next gaussian step:')
        mat_print(r)

        h += 1
        k += 1

    return backwards_substitution(r)

def sys_solve_scaled_partial_pivoting(s: System) -> Matrix:
    r = sys_augmented(s)
    h = 0
    k = 0
    perm_vec = [i for i in range(r.rows)]
    s_vec = [ max([abs(r.data[i][j]) for j in range(r.rows)]) for i in range(r.rows)]

    while h < r.rows and k < r.cols:
        swap_idx = h
        for i in range(h + 1, r.rows):
            pivot = D(abs(r.data[i][k]) / s_vec[perm_vec[i]])
            if pivot != D(0) and pivot > D(abs(r.data[swap_idx][k]) / s_vec[perm_vec[swap_idx]]):
                swap_idx = i

        if swap_idx and swap_idx != h:
            temp = r.data[swap_idx]
            r.data[swap_idx] = r.data[h]
            r.data[h] = temp
            mat_print(r)

            temp = perm_vec[swap_idx]
            perm_vec[swap_idx] = perm_vec[h]
            perm_vec[h] = temp


        if r.data[h][k] == D(0):
            raise Exception('Potential division by 0 due to precision loss. Terminating.')

        for i in range(h + 1, r.rows):
            f = r.data[i][k] / r.data[h][k]

            r.data[i][k] = D(0)
            for j in range(k + 1, r.cols):
                r.data[i][j] = r.data[i][j] - r.data[h][j] * f

        print('Next gaussian step:')
        mat_print(r)

        h += 1
        k += 1

    return backwards_substitution(r)

def error(x: Matrix, x_prev: Matrix) -> Decimal:
    diff = mat_add(x, mat_mult_scalar(x_prev, D(-1)))

    abs_error = D(0)
    for i in range(diff.rows):
        if abs(diff.data[i][0]) > abs_error:
            abs_error = abs(diff.data[i][0])

    norm_x = D(0)
    for i in range(x.rows):
        if abs(x.data[i][0]) > norm_x:
            norm_x = abs(x.data[i][0])

    return abs_error / norm_x

def sys_solve_iterative_jacobi(s: System, epsilon: Decimal) -> Matrix:
    a = mat_copy(s.a)

    d = mat_diag(a)
    l = mat_lower(a)
    u = mat_upper(a)

    d_inv = mat_zero(a.rows, a.rows)
    for i in range(a.rows):
        # Safe to divide, diagonal doesn't have 0 entries
        d_inv.data[i][i] = D(1) / d.data[i][i]

    t = mat_mult(d_inv, mat_add(mat_mult_scalar(a, D(-1)), d))
    c = mat_mult(d_inv, s.b)

    print('T is:')
    mat_print(t)

    print('C is:')
    mat_print(c)

    # x^{0} is a zero vector
    x = mat_zero(a.rows, 1)
    while True:
        x_temp = mat_add(mat_mult(t, x), c)
        err = error(x_temp, x)
        x = x_temp

        if err < epsilon:
            break

    return x

def sys_solve_iterative_gauss_seidal(s: System, epsilon: Decimal) -> Matrix:
    a = mat_copy(s.a)

    d = mat_diag(a)
    l = mat_lower(a)
    u = mat_upper(a)

    l_inv = mat_zero(a.rows, a.rows)

    # Computing the inverse of a lower triangular matrix
    for i in range(a.rows):
        # Safe to divide, diagonal doesn't have 0 entries
        l_inv.data[i][i] = D(1)/a.data[i][i]
        for j in range(0, i):
            temp = sum([l.data[i][k] * l_inv.data[k][j] for k in range(j, i)])
            l_inv.data[i][j] = -temp * l_inv.data[i][i]

    t = mat_mult(l_inv, mat_add(d, mat_mult_scalar(u, D(-1))))
    c = mat_mult(l_inv, s.b)

    print('T is:')
    mat_print(t)

    print('C is:')
    mat_print(c)

    # x^{0} is a zero vector
    x = mat_zero(a.rows, 1)
    while True:
        x_temp = mat_add(mat_mult(t, x), c)
        err = error(x_temp, x)
        x = x_temp

        if err < epsilon:
            break

    return x

###
#
# High precision for creating systems of linear of equations
#
###
getcontext().prec = HIGH_PRECISION

s = sys_random(4,24)
sys_print(s)

###
#
# Low precision for solving systems of linear equations
#
###
getcontext().prec = LOW_PRECISION

x1 = sys_solve_gaussian_elimination(s)
x2 = sys_solve_partial_pivoting(s)
x3 = sys_solve_scaled_partial_pivoting(s)
mat_print(x1)
mat_print(x2)
mat_print(x3)

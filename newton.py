import argparse
import sympy as sym
import numpy as np

def derivative(func):
    my_func = sym.parse_expr(func)
    return sym.diff(my_func)

def newton_method(x0, iteration_number, f, df, accuracy, step):

    x = x0
    for i in range(int(iteration_number)):
        y = f(x)
        y_prime = df(x)
        x1 = x - y / y_prime
        # x1 = x - y / (y_prime * step) # if you want step uncomment this
        if np.abs(f(x1)) <= accuracy:
            return x1, f(x1)
        x = x1
    return x, f(x)


parser = argparse.ArgumentParser(description='Newtons method.')
parser.add_argument('func', type=str, action='store', help='Function.')
parser.add_argument('-start_point', type=int, action='store', help='Starting point')
parser.add_argument('-number_of_steps', type=int, action='store', help='Num of steps')
parser.add_argument('-accuracy', type=float, action='store', help='Accuracy')
parser.add_argument('-step_size', type=float, action='store', help='Step (0..1)')

args = parser.parse_args()
x = sym.Symbol('x')
f = sym.parse_expr(args.func)


print('Function: ' + args.func)
df = derivative(args.func)
print('Derivative: ' + str(df))

f = sym.lambdify(x, f, 'numpy')
df = sym.lambdify(x, df, 'numpy')


solution, y = newton_method(args.start_point, args.number_of_steps, f, df, args.accuracy, args.step_size)
print("x: ",solution,"y: ", y)

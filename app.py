from flask import Flask, render_template, request
from sympy import sympify, symbols, diff, lambdify
import math

app = Flask(__name__)
x = symbols('x')


def get_function_from_string(expr_str):
    sym_expr = sympify(expr_str)
    f = lambdify(x, sym_expr, 'math')
    d_expr = diff(sym_expr, x)
    df = lambdify(x, d_expr, 'math')
    return f, df, sym_expr, d_expr


def bisection(f, a, b, max_iter, tol):
    rows = []
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")

    c_old = None
    for i in range(1, max_iter + 1):
        c = 0.5 * (a + b)
        fc = f(c)
        err = abs(c - c_old) if c_old is not None else None
        rows.append((i, a, b, c, fc, err))

        if abs(fc) == 0 or (c_old is not None and abs(c - c_old) < tol) or (abs(b - a) / 2 < tol):
            return c, abs(fc), i, rows

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        c_old = c

    return c, abs(f(c)), max_iter, rows


def regula_falsi(f, a, b, max_iter, tol):
    rows = []
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")

    c_old = None
    for i in range(1, max_iter + 1):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        err = abs(c - c_old) if c_old is not None else None
        rows.append((i, a, b, c, fc, err))

        if abs(fc) == 0 or (c_old is not None and abs(c - c_old) < tol):
            return c, abs(fc), i, rows

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        c_old = c

    return c, abs(f(c)), max_iter, rows


def secant(f, x0, x1, max_iter, tol):
    rows = []
    for i in range(1, max_iter + 1):
        f0 = f(x0)
        f1 = f(x1)
        if (f1 - f0) == 0:
            raise ValueError("Zero denominator in Secant.")

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = f(x2)
        err = abs(x2 - x1)
        rows.append((i, x0, x1, x2, f2, err))

        if abs(f2) == 0 or err < tol:
            return x2, abs(f2), i, rows

        x0, x1 = x1, x2

    return x2, abs(f(x2)), max_iter, rows


def newton_raphson(f, df, x0, max_iter, tol):
    rows = []
    for i in range(1, max_iter + 1):
        fx = f(x0)
        dfx = df(x0)
        if dfx == 0:
            raise ValueError("Derivative zero in Newton-Raphson.")

        x1 = x0 - fx / dfx
        err = abs(x1 - x0)
        rows.append((i, x0, fx, dfx, x1, err))

        if abs(fx) == 0 or err < tol:
            return x1, abs(f(x1)), i, rows

        x0 = x1

    return x1, abs(f(x1)), max_iter, rows


def fixed_point_iteration(g_func, x0, max_iter, tol):
    rows = []
    for i in range(1, max_iter + 1):
        x1 = g_func(x0)
        err = abs(x1 - x0)
        rows.append((i, x0, x1, err))

        if err < tol:
            return x1, err, i, rows

        x0 = x1

    return x1, err, max_iter, rows


def modified_secant(f, x0, delta, max_iter, tol):
    rows = []
    for i in range(1, max_iter + 1):
        fx = f(x0)
        denom = f(x0 + delta * x0) - fx
        if denom == 0:
            raise ValueError("Denominator zero in Modified Secant.")

        x1 = x0 - (delta * x0 * fx) / denom
        err = abs(x1 - x0)
        rows.append((i, x0, fx, x1, err))

        if err < tol:
            return x1, abs(f(x1)), i, rows

        x0 = x1

    return x1, abs(f(x1)), max_iter, rows


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None

    if request.method == 'POST':
        method = request.form.get('method')
        func_expr = request.form.get('func').strip()
        max_iter = int(request.form.get('max_iter'))
        tol = float(request.form.get('tol'))

        try:
            f, df, _, _ = get_function_from_string(func_expr)
        except Exception as e:
            return render_template('index.html', error=f"Invalid function: {e}")

        try:

            if method == 'bisection':
                a = float(request.form.get('a'))
                b = float(request.form.get('b'))
                root, final_err, iters, rows = bisection(f, a, b, max_iter, tol)
                headers = ["Iter", "a", "b", "c", "f(c)", "Error"]

                result = {
                    "method": "Bisection",
                    "root": root,
                    "final_err": final_err,
                    "iters": iters,
                    "rows": rows,
                    "headers": headers
                }

            elif method == 'regula':
                a = float(request.form.get('a'))
                b = float(request.form.get('b'))
                root, final_err, iters, rows = regula_falsi(f, a, b, max_iter, tol)
                headers = ["Iter", "a", "b", "c", "f(c)", "Error"]

                result = {
                    "method": "Regula Falsi",
                    "root": root,
                    "final_err": final_err,
                    "iters": iters,
                    "rows": rows,
                    "headers": headers
                }

            elif method == 'secant':
                x0 = float(request.form.get('x0'))
                x1 = float(request.form.get('x1'))
                root, final_err, iters, rows = secant(f, x0, x1, max_iter, tol)
                headers = ["Iter", "x0", "x1", "x2", "f(x2)", "Error"]

                result = {
                    "method": "Secant",
                    "root": root,
                    "final_err": final_err,
                    "iters": iters,
                    "rows": rows,
                    "headers": headers
                }

            elif method == 'newton':
                x0 = float(request.form.get('x0'))
                root, final_err, iters, rows = newton_raphson(f, df, x0, max_iter, tol)
                headers = ["Iter", "x", "f(x)", "f'(x)", "x_new", "Error"]

                result = {
                    "method": "Newton-Raphson",
                    "root": root,
                    "final_err": final_err,
                    "iters": iters,
                    "rows": rows,
                    "headers": headers
                }

            elif method == 'fixed':
                g_expr = request.form.get('gfunc').strip()
                g_func = lambdify(x, sympify(g_expr), 'math')
                x0 = float(request.form.get('x0'))

                root, final_err, iters, rows = fixed_point_iteration(g_func, x0, max_iter, tol)
                headers = ["Iter", "x_old", "x_new", "Error"]

                result = {
                    "method": "Fixed Point",
                    "root": root,
                    "final_err": final_err,
                    "iters": iters,
                    "rows": rows,
                    "headers": headers
                }

            elif method == 'modified_secant':
                x0 = float(request.form.get('x0'))
                delta = float(request.form.get('delta'))
                root, final_err, iters, rows = modified_secant(f, x0, delta, max_iter, tol)
                headers = ["Iter", "x", "f(x)", "x_new", "Error"]

                result = {
                    "method": "Modified Secant",
                    "root": root,
                    "final_err": final_err,
                    "iters": iters,
                    "rows": rows,
                    "headers": headers
                }

        except Exception as e:
            error = f"Error during computation: {e}"

    return render_template('index.html', result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

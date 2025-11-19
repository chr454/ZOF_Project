#!/usr/bin/env python3
"""
ZOF_CLI.py
Command Line Zero-Of-Function (root-finding) solver
Supports: Bisection, Regula Falsi, Secant, Newton-Raphson, Fixed Point, Modified Secant

Usage: run with Python (interactive prompts will guide you).
"""
import math
from sympy import sympify, symbols, diff, lambdify
import sys

x = symbols('x')

def get_function_from_string(expr_str):
    try:
        sym_expr = sympify(expr_str)
        f = lambdify(x, sym_expr, 'math')
        # also return derivative sympy expression
        d_expr = diff(sym_expr, x)
        df = lambdify(x, d_expr, 'math')
        return f, df, sym_expr, d_expr
    except Exception as e:
        raise ValueError(f"Invalid function expression: {e}")

def safe_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except Exception:
            print("Enter a valid number (like 1.5 or -2).")

def safe_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except Exception:
            print("Enter a valid integer.")

def bisection(f, a, b, max_iter, tol):
    fa = f(a); fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Bisection.")
    print("\nIter |    a       |    b       |    c       |  f(c)       |  Error")
    c_old = None
    for i in range(1, max_iter + 1):
        c = 0.5 * (a + b)
        fc = f(c)
        err = abs(c - c_old) if c_old is not None else float('nan')
        print(f"{i:4d} | {a:10.6f} | {b:10.6f} | {c:10.6f} | {fc:12.6e} | {err:10.6e}")
        if abs(fc) == 0 or (c_old is not None and abs(c - c_old) < tol) or (abs(b-a)/2 < tol):
            return c, abs(fc), i
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        c_old = c
    return c, abs(f(c)), max_iter

def regula_falsi(f, a, b, max_iter, tol):
    fa = f(a); fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Regula Falsi.")
    print("\nIter |    a       |    b       |    c       |  f(c)       |  Error")
    c_old = None
    c = a
    for i in range(1, max_iter + 1):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        err = abs(c - c_old) if c_old is not None else float('nan')
        print(f"{i:4d} | {a:10.6f} | {b:10.6f} | {c:10.6f} | {fc:12.6e} | {err:10.6e}")
        if abs(fc) == 0 or (c_old is not None and abs(c - c_old) < tol):
            return c, abs(fc), i
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        c_old = c
    return c, abs(f(c)), max_iter

def secant(f, x0, x1, max_iter, tol):
    print("\nIter |    x0       |    x1       |    x2       |  f(x2)       |  Error")
    for i in range(1, max_iter + 1):
        f0 = f(x0); f1 = f(x1)
        if (f1 - f0) == 0:
            raise ZeroDivisionError("Division by zero in Secant denominator.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = f(x2)
        err = abs(x2 - x1)
        print(f"{i:4d} | {x0:11.6f} | {x1:11.6f} | {x2:11.6f} | {f2:12.6e} | {err:10.6e}")
        if abs(f2) == 0 or err < tol:
            return x2, abs(f2), i
        x0, x1 = x1, x2
    return x2, abs(f(x2)), max_iter

def newton_raphson(f, df, x0, max_iter, tol):
    print("\nIter |    x_n       |   f(x_n)     |  f'(x_n)   |   x_{n+1}    |  Error")
    for i in range(1, max_iter+1):
        fx = f(x0); dfx = df(x0)
        if dfx == 0:
            raise ZeroDivisionError("Zero derivative encountered in Newton-Raphson.")
        x1 = x0 - fx/dfx
        err = abs(x1 - x0)
        print(f"{i:4d} | {x0:11.6f} | {fx:12.6e} | {dfx:10.6e} | {x1:11.6f} | {err:10.6e}")
        if abs(fx) == 0 or err < tol:
            return x1, abs(f(x1)), i
        x0 = x1
    return x1, abs(f(x1)), max_iter

def fixed_point_iteration(g_func, x0, max_iter, tol):
    print("\nIter |    x_n       |   g(x_n)     |  Error")
    for i in range(1, max_iter+1):
        x1 = g_func(x0)
        err = abs(x1 - x0)
        print(f"{i:4d} | {x0:11.6f} | {x1:12.6e} | {err:10.6e}")
        if err < tol:
            return x1, err, i
        x0 = x1
    return x1, err, max_iter

def modified_secant(f, x0, delta, max_iter, tol):
    print("\nIter |    x_n       |   f(x_n)     |   x_n+1      |  Error")
    for i in range(1, max_iter+1):
        f_x0 = f(x0)
        x1 = x0 - (delta * x0 * f_x0) / (f(x0 + delta * x0) - f_x0)
        err = abs(x1 - x0)
        print(f"{i:4d} | {x0:11.6f} | {f_x0:12.6e} | {x1:12.6f} | {err:10.6e}")
        if err < tol:
            return x1, abs(f(x1)), i
        x0 = x1
    return x1, abs(f(x1)), max_iter

def main():
    print("=== ZOF CLI (Zero-Of-Function) Solver ===")
    print("Enter the function f(x) (use Python syntax; variable must be 'x'). Example: x**3 - 2*x - 5")
    expr = input("f(x) = ").strip()
    try:
        f, df, sym_expr, d_expr = get_function_from_string(expr)
    except Exception as e:
        print(e)
        sys.exit(1)

    print("\nChoose method (enter number):")
    print("1) Bisection")
    print("2) Regula Falsi (False Position)")
    print("3) Secant")
    print("4) Newton-Raphson")
    print("5) Fixed Point Iteration (provide g(x) such that x = g(x))")
    print("6) Modified Secant")
    choice = safe_int("Method number: ")

    max_iter = safe_int("Max iterations (e.g. 50): ")
    tol = safe_float("Tolerance (e.g. 1e-6): ")

    try:
        if choice == 1:
            a = safe_float("Enter left endpoint a: ")
            b = safe_float("Enter right endpoint b: ")
            root, final_err, iters = bisection(f, a, b, max_iter, tol)
        elif choice == 2:
            a = safe_float("Enter left endpoint a: ")
            b = safe_float("Enter right endpoint b: ")
            root, final_err, iters = regula_falsi(f, a, b, max_iter, tol)
        elif choice == 3:
            x0 = safe_float("Enter x0 (first initial guess): ")
            x1 = safe_float("Enter x1 (second initial guess): ")
            root, final_err, iters = secant(f, x0, x1, max_iter, tol)
        elif choice == 4:
            x0 = safe_float("Enter initial guess x0: ")
            root, final_err, iters = newton_raphson(f, df, x0, max_iter, tol)
        elif choice == 5:
            print("For Fixed Point Iteration, you must supply g(x) such that x = g(x).")
            g_expr = input("g(x) = ").strip()
            try:
                g_sym = sympify(g_expr)
                g_func = lambdify(x, g_sym, 'math')
            except Exception as e:
                print(f"Invalid g(x): {e}")
                sys.exit(1)
            x0 = safe_float("Enter initial guess x0: ")
            root, final_err, iters = fixed_point_iteration(g_func, x0, max_iter, tol)
        elif choice == 6:
            x0 = safe_float("Enter initial guess x0: ")
            delta = safe_float("Enter delta (small number, e.g. 1e-3): ")
            root, final_err, iters = modified_secant(f, x0, delta, max_iter, tol)
        else:
            print("Invalid method choice.")
            sys.exit(1)
    except Exception as e:
        print("Error during computation:", e)
        sys.exit(1)

    print("\n=== RESULT ===")
    print(f"Estimated root: {root:.12f}")
    print(f"Final function value (|f(root)|): {final_err:.6e}")
    print(f"Iterations used: {iters}")
    print("=== End ===")

if __name__ == "__main__":
    main()

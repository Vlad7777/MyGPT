#MyGPT_DynP

"""

Dynamic programming is a technique for solving optimization problems by breaking them down into smaller overlapping subproblems and solving each subproblem only once, saving the results for future use. Here's a simple example of dynamic programming using the classic problem of finding the nth Fibonacci number:

python
Copy code
"""

def fibonacci_recursive(n):
    if n <= 1:
        return n
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def fibonacci_dynamic_programming(n):
    fib_values = [0, 1]

    for i in range(2, n + 1):
        fib_values.append(fib_values[i - 1] + fib_values[i - 2])

    return fib_values[n]

# Example usage
n = 10

# Recursive approach (inefficient for large n)
fib_recursive = fibonacci_recursive(n)
print(f"Fibonacci({n}) using recursive approach: {fib_recursive}")

# Dynamic programming approach (more efficient)
fib_dp = fibonacci_dynamic_programming(n)
print(f"Fibonacci({n}) using dynamic programming: {fib_dp}")

"""
In this example:

fibonacci_recursive is a simple recursive function for calculating Fibonacci numbers, but it becomes inefficient for large n due to redundant calculations.
fibonacci_dynamic_programming uses dynamic programming to store intermediate results in the fib_values list, making it more efficient.
Dynamic programming is particularly useful for problems that exhibit overlapping subproblems and optimal substructure, allowing you to avoid redundant computations. The Fibonacci sequence is a classic example, but dynamic programming can be applied to various problems, such as the knapsack problem, longest common subsequence, and more.

It's important to note that dynamic programming can often be further optimized, and there are advanced topics like memoization and tabulation that can enhance its performance for specific problems.


"""




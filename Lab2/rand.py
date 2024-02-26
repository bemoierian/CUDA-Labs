import numpy as np

# Take input for rows and cols from the console
rows = int(input("Enter the number of rows: "))
cols = int(input("Enter the number of columns: "))

# Generate random numbers for Matrix A
matrix_a = np.random.rand(rows, cols) * 100

# Generate random numbers for Matrix B
matrix_b = np.random.rand(rows, cols) * 100

filename = f"q1_{rows}_{cols}.txt"
# Save matrices to a text file
with open(filename, "w") as file:
    file.write("1\n")  # Number of test cases
    file.write(f"{rows} {cols}\n")  # Matrix dimensions
    np.savetxt(file, matrix_a, fmt='%.1f', delimiter=' ')  # Matrix A
    np.savetxt(file, matrix_b, fmt='%.1f', delimiter=' ')  # Matrix B

print(f"Matrices saved to {filename}")

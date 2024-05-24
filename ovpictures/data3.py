import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Load dataset
df = pd.read_excel("./LB-LH_SHUBHAM.xlsx")

# Values
X = df.columns[1:].astype(float)
Y = np.array(df.iloc[:, 0])
Z = df.iloc[:, 1:].values

def get_user_input():
    x1 = float(input("Enter x1: "))
    y1 = float(input("Enter y1: "))
    x2 = float(input("Enter x2: "))
    y2 = float(input("Enter y2: "))
    return x1, y1, x2, y2

def select_rectangle_data(x1, y1, x2, y2):
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    x_indices = np.where((X >= x1) & (X <= x2))[0]
    y_indices = np.where((Y >= y1) & (Y <= y2))[0]

    if len(x_indices) == 0 or len(y_indices) == 0:
        print("No Data Selected")
        return

    selected_data = Z[np.ix_(y_indices, x_indices)]

    data_sum = np.sum(selected_data)
    minval = np.min(selected_data)
    maxval = np.max(selected_data)

    min_idx = np.unravel_index(np.argmin(selected_data), selected_data.shape)
    max_idx = np.unravel_index(np.argmax(selected_data), selected_data.shape)
    min_x, min_y = X[x_indices[min_idx[1]]], Y[y_indices[min_idx[0]]]
    max_x, max_y = X[x_indices[max_idx[1]]], Y[y_indices[max_idx[0]]]

    output = []
    output.append(f"Rectangle corners: (x1: {x1}, y1: {y1}) to (x2: {x2}, y2: {y2})")
    output.append("Selected data points:")
    for i, data_point in enumerate(selected_data.flatten()):
        output.append(f"Data point {i+1}: {data_point}")

    output.append(f"Average Data: {(data_sum / selected_data.size):.3f}")
    output.append(f"Min Data point is at ({min_x}, {min_y}): {minval}")
    output.append(f"Max Data point is at ({max_x}, {max_y}): {maxval}")
    output.append("\n" + "="*80 + "\n")

    # Append data to a file
    with open("selected_data.txt", "a") as file:
        for line in output:
            file.write(line + "\n")

    print("\n".join(output))

# Normalize data 
norm = Normalize(vmin=Z.min(), vmax=Z.max())

# Plot
fig, ax = plt.subplots()
im = ax.contourf(X, Y, Z, cmap='turbo', norm=norm, levels=20, linestyles='solid', origin="upper", extent=[X.min(), X.max(), Y.min(), Y.max()])
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Intensity Value (Candela)', fontsize=12)

ax.set_xlabel('Horizontal Angle', fontsize=12)
ax.set_ylabel('Vertical Angle', fontsize=12)
ax.set_title('Intensity Plot', fontsize=14)

plt.show()

# Get user input and select rectangle data
x1, y1, x2, y2 = get_user_input()
select_rectangle_data(x1, y1, x2, y2)

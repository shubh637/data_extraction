import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import RectangleSelector

# Load dataset
df = pd.read_excel("./LB-LH_SHUBHAM.xlsx")

#  values
X = df.columns[1:]
Y = np.array(df.iloc[:, 0])
Z = df.iloc[:, 1:].values


def onselect_rectangle(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    
    x1, x2 = int(np.floor(x1)), int(np.ceil(x2))
    y1, y2 = int(np.floor(y1)), int(np.ceil(y2))
    
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x2 > Z.shape[1]: x2 = Z.shape[1]
    if y2 > Z.shape[0]: y2 = Z.shape[0]

    selected_data = Z[y1:y2, x1:x2]
    
    if selected_data.size == 0:
        print("No data points selected.")
        return

    print("Selected data points:")
    data_sum, minval, maxval = compute_statistics(selected_data)

    print(f"Average Data: {(data_sum / selected_data.size):.1f}")
    print(f"Min Data point: {minval:.1f}")
    print(f"Max Data point: {maxval:.1f}")


def compute_statistics(data):
    data_sum = data.sum()
    minval = data.min()
    maxval = data.max()
    for i, data_point in enumerate(data.flatten()):
        print(f"Data point {i + 1}: {data_point:.1f}")
    return data_sum, minval, maxval


plt.figure(figsize=(10, 8))
ax = sns.heatmap(Z, cmap='turbo', xticklabels=False, yticklabels=False, cbar_kws={'label': 'Colorbar'}, linewidths=.5)


ax.set_xlabel('Horizontal', fontsize=12)
ax.set_ylabel('Vertical', fontsize=12)
ax.set_title('Heatmap with Interactive Selection', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)


rect_selector = RectangleSelector(ax.figure.gca(), onselect_rectangle, useblit=True, button=[1], minspanx=1, minspany=1, spancoords='pixels', interactive=True)



plt.show()

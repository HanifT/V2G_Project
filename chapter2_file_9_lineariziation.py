import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
# %%

def BD_deg():

    # Load the JSON file containing the points
    with open('batt_deg.json', 'r') as file:
        data = json.load(file)

    # Extract X and Y coordinates
    X = np.array(data['DOD'])
    Y = np.array(data['BD'])

    # Combine X and Y into a single matrix
    points = np.column_stack((X, Y))
    points1 = pd.DataFrame({"DOD": X, "BD": Y})


    # Fit a polynomial function to the data
    degree = 3  # You can adjust the degree of the polynomial as needed
    coefficients = np.polyfit(points1["DOD"], points1["BD"], degree)

    # Define the polynomial function
    poly_func = np.poly1d(coefficients)

    # Generate x values for plotting
    x_values = np.linspace(0, 100, 6)

    # Calculate corresponding y values using the polynomial function
    y_values = poly_func(x_values)

    points2 = pd.DataFrame({"DOD": x_values, "BD": y_values}).reset_index(drop=True)
    points2.loc[0, "BD"] = 0
    points2.loc[points2.index[-1], "BD"] = 0.0005

    return points, points1, points2
#
# # Plot the original data and the fitted polynomial function
# plt.scatter(points1["DOD"], points1["BD"], label='Original Data')
# plt.plot(x_values, y_values, color='red', label='Fitted Polynomial')
#
# # Add labels and legend
# plt.xlabel('DOD')
# plt.ylabel('BD')
# plt.title('Polynomial Fit to Data')
# plt.legend()
#
# # Show the plot
# plt.show()
# ################################################################################################################
# ################################################################################################################
# # Use K-means clustering to partition the points into 5 clusters
# kmeans = KMeans(n_clusters=6, random_state=0)
# clusters = kmeans.fit_predict(points)
#
# # Plot original points
# plt.scatter(X, Y, c=clusters, cmap='viridis', label='Original Points')
#
# # Fit linear regression lines to each cluster
# for i in range(6):
#     cluster_points = points[clusters == i]
#     reg = LinearRegression().fit(cluster_points[:, 0].reshape(-1, 1), cluster_points[:, 1])
#     line_X = np.linspace(min(cluster_points[:, 0]), max(cluster_points[:, 0]), 100)
#     line_Y = reg.predict(line_X.reshape(-1, 1))
#     plt.plot(line_X, line_Y, label=f'Line {i + 1}', color='r')
#
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Piecewise Linear Sections')
# plt.legend()
# plt.show()

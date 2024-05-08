import matplotlib.pyplot as plt
from sklearn.cluster import Birch
import open3d as o3d
import numpy as np

ply_point_cloud = o3d.data.PLYPointCloud()

pcd = o3d.io.read_point_cloud(ply_point_cloud.path, remove_nan_points=True, remove_infinite_points=True)
plane_model, inliers = pcd.segment_plane(distance_threshold=0.10, ransac_n=3, num_iterations=1000)

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)

inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

############################################################################
nase_pcd = "/home/d618/Desktop/PythonFilesStupen/pythonProject2/output.pcd"
point_cloud = o3d.io.read_point_cloud(nase_pcd, remove_nan_points=True, remove_infinite_points=True)

plane_model2, inliers2 = point_cloud.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)

inlier_cloud2 = point_cloud.select_by_index(inliers2)
outlier_cloud2 = point_cloud.select_by_index(inliers2, invert=True)

inlier_cloud2.paint_uniform_color([1, 0, 0])
#cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#outlier_cloud2.paint_uniform_color([0.6, 0.6, 0.6])

o3d.visualization.draw_geometries([inlier_cloud2, outlier_cloud2])
#o3d.io.write_point_cloud("outliner_cloud.ply", outlier_cloud2)


######################################  DBSCAN - PCD Z INTERNETU   ################################################
labels = np.array(pcd.cluster_dbscan(eps=0.034, min_points=60))
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label
if max_label > 0 else 1))

colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([pcd])

#######################################  DBSCAN - NASE PCD  #####################################
pcd_new = "/home/d618/Desktop/PythonFilesStupen/pythonProject2/outliner_cloud.ply"
point_cloud2 = o3d.io.read_point_cloud(pcd_new, remove_nan_points=True, remove_infinite_points=True)
labels = np.array(point_cloud2.cluster_dbscan(eps=0.03, min_points=60))
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label
if max_label > 0 else 1))

colors[labels < 0] = 0
point_cloud2.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([point_cloud2])

################# BIRCH CLUSTERING - INTERNET ##########################
points = np.asarray(pcd.points)

birch = Birch(threshold=0.2, n_clusters=3)
birch.fit(points)

labels = birch.labels_
unique_labels = np.unique(labels)

cluster_colors = np.random.rand(len(unique_labels), 3)

colors = np.array([cluster_colors[label] for label in labels])
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])

################# BIRCH CLUSTERING - NASE PCD ##########################
points = np.asarray(point_cloud2.points)

birch = Birch(threshold=0.10, n_clusters=5)
birch.fit(points)

labels = birch.labels_
unique_labels = np.unique(labels)

cluster_colors = np.random.rand(len(unique_labels), 3)

colors = np.array([cluster_colors[label] for label in labels])
point_cloud2.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([point_cloud2])
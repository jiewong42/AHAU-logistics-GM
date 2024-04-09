import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib

# 设置中文字体
font_path = 'HarmonyOS_Sans_SC_Black.ttf'  # 替换为你的中文字体文件路径
font = FontProperties(fname=font_path, size=12)
matplotlib.rc('font', family='SimSun')

# 从Excel文件中读取数据
data = pd.read_excel("mdl4.xlsx")

# 提取经度和纬度列数据
coordinates = data.iloc[:, 1:].values

# 设置聚类数目的范围
k_range = range(2, 93)  # 尝试聚类数目从2到93

# 存储每个K值对应的轮廓系数
silhouette_scores = []

# 计算每个K值对应的轮廓系数
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(coordinates)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(coordinates, labels))

# 设置图形的大小和分辨率
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# 绘制K值与轮廓系数的关系图
plt.plot(k_range, silhouette_scores, 'bo-')
plt.xlabel('聚类数目K', fontproperties=font)
plt.ylabel('轮廓系数', fontproperties=font)
plt.title('轮廓系数法确定最佳聚类数目K', fontproperties=font)
plt.show()

# 输出最大的轮廓系数及对应的K值
max_score = max(silhouette_scores)
best_k = k_range[silhouette_scores.index(max_score)]
print("最大的轮廓系数:", max_score)
print("对应的K值:", best_k)

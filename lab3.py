# Импорт библиотек
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation, Birch
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sklearn.cluster import Birch, AffinityPropagation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.cluster import DBSCAN, AgglomerativeClustering


# Загрузка данных
path ='WineQT.csv'


data = pd.read_csv(path)

# Предположим, что последний столбец - целевая переменная (y), а остальные - признаки (X)
# Исключение столбца Id из данных
X = data.drop(columns=['quality', 'Id'])  # 'quality' остается целевой переменной
y = data['quality']

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# 1. Birch
birch = Birch(n_clusters=3)
birch_clusters = birch.fit_predict(X_scaled)
print("Birch Silhouette Score:", silhouette_score(X_scaled, birch_clusters))

# 2. DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(X_scaled)
# Проверка метрики возможна только, если DBSCAN создал несколько кластеров
if len(set(dbscan_clusters)) > 1:
    print("DBSCAN Silhouette Score:", silhouette_score(X_scaled, dbscan_clusters))
else:
    print("DBSCAN: Недостаточно кластеров для вычисления метрики Silhouette")

# 3. Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3)
agglo_clusters = agglo.fit_predict(X_scaled)
print("Agglomerative Clustering Silhouette Score:", silhouette_score(X_scaled, agglo_clusters))


# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)
print("KNeighbors Accuracy:", accuracy_score(y_test, knn_predictions))

# 2. MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
mlp_predictions = mlp.predict(X_test)
print("MLP Accuracy:", accuracy_score(y_test, mlp_predictions))



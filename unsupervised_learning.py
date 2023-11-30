import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()
import seaborn as sns
sns.pairplot(df)
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 2)
y_predicted=km.fit_predict(df)
df['cluster']=y_predicted
df1=df[df.cluster==0]
df2 = df[df.cluster==0]
sns.pairplot(df,hue='cluster')
df = pd.DataFrame(iris.data, columns = iris.feature_names)
scaler = MinMaxScaler()
scaler.fit(df)
scaler.transform(df)
df_scaled = pd.DataFrame(scaler.transform(df), columns = iris.feature_names)
df_scaled.head()
sns.pairplot(df_scaled)
y_predicted = km.fit_predict(df_scaled)
df_scaled['cluster'] = y_predicted
df_scaled.head()
sns.pairplot(df_scaled, hue = 'cluster')
km.cluster_centers_
df = pd.DataFrame(iris.data, columns = iris.feature_names)
k_value = range(1,10)
sse = []
for k in k_value:
    km = KMeans(n_clusters = k)
    km.fit(df)
    sse.append(km.inertia_)
sse

df = pd.DataFrame(iris.data, columns = iris.feature_names)
km35 = KMeans(n_clusters = 3)
y_cluster = km35.fit_predict(df)
y_cluster
plt.plot(k_value, sse, color = 'b', label = 'not scaled')
plt.xlabel('Number of cluster, K')
plt.ylabel('Value of cost function, sse')
df['cluster'] = y_cluster
df.head()














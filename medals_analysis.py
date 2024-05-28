import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load the dataset
data = pd.read_csv('medals.csv')

# Display the first and last few rows of the dataset
print(data)

# Summary statistics
print(data.describe())

# Distribution of total medals
plt.figure(figsize=(10, 6))
sns.histplot(data['Total'], bins=15, kde=True, color='blue')
plt.title('Distribution of Total Medals')
plt.xlabel('Total Medals')
plt.ylabel('Frequency')
plt.show()

# Define the custom palette
medal_palette = {'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'}

# Extracting top 10 countries
top_10 = data.nlargest(10, 'Total')
top_10_melted = top_10.melt(id_vars='Country', value_vars=['Gold', 'Silver', 'Bronze'], var_name='Medal', value_name='Count')

# Plotting the Medal Breakdown by Type for the top 10 countries
plt.figure(figsize=(12, 8))
sns.barplot(x='Count', y='Country', hue='Medal', data=top_10_melted, palette=medal_palette)
plt.title('Medal Breakdown by Type for Top 10 Countries')
plt.xlabel('Count')
plt.ylabel('Country')
plt.legend(title='Medal')
plt.show()

# Prepare data for plotting
top_gold = data.nlargest(10, 'Gold')
top_silver = data.nlargest(10, 'Silver')
top_bronze = data.nlargest(10, 'Bronze')
top_total = data.nlargest(10, 'Total')

# Create a figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot Top 10 countries by gold medals
sns.barplot(x='Gold', y='Country', data=top_gold, ax=axes[0, 0], color='gold')
axes[0, 0].set_title('Top 10 Countries by Gold Medals')
axes[0, 0].set_xlabel('Gold Medals')
axes[0, 0].set_ylabel('Country')

# Plot Top 10 countries by silver medals
sns.barplot(x='Silver', y='Country', data=top_silver, ax=axes[0, 1], color='silver')
axes[0, 1].set_title('Top 10 Countries by Silver Medals')
axes[0, 1].set_xlabel('Silver Medals')
axes[0, 1].set_ylabel('Country')

# Plot Top 10 countries by bronze medals
sns.barplot(x='Bronze', y='Country', data=top_bronze, ax=axes[1, 0], color='brown')
axes[1, 0].set_title('Top 10 Countries by Bronze Medals')
axes[1, 0].set_xlabel('Bronze Medals')
axes[1, 0].set_ylabel('Country')

# Plot Top 10 countries by total medals
sns.barplot(x='Total', y='Country', data=top_total, ax=axes[1, 1], color='blue')
axes[1, 1].set_title('Top 10 Countries by Total Medals')
axes[1, 1].set_xlabel('Total Medals')
axes[1, 1].set_ylabel('Country')

# Adjust layout
plt.tight_layout(pad=4.0)
plt.show()

# Aggregate medals by continent
continent_medals = data.groupby('Continent')[['Gold', 'Silver', 'Bronze', 'Total']].sum().reset_index()
print(continent_medals)

# Plot total medals by continent
plt.figure(figsize=(12, 8))
sns.barplot(x='Total', y='Continent', data=continent_medals)
plt.title('Total Medals by Continent')
plt.xlabel('Total Medals')
plt.ylabel('Continent')
plt.show()

# Select features for clustering
X = data[['Gold', 'Total']]

# Normalize data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Apply Gaussian Mixture Model clustering
gmm = GaussianMixture(n_components=4)  # Adjust n_components as needed
gmm.fit(X_normalized)
clusters = gmm.predict(X_normalized)
probabilities = gmm.predict_proba(X_normalized)

# Add cluster labels to the dataset
data['Cluster'] = clusters
data['Cluster_Probabilities'] = list(probabilities)

# Print probabilities for a few data points
for i in range(5):
    print(f"Country: {data.iloc[i]['Country']}, Probabilities: {data.iloc[i]['Cluster_Probabilities']}")

# Visualize the clusters with cluster centers
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Total', y='Gold', hue='Cluster', data=data, palette='viridis', legend='full')

# Plot the cluster centers
centers = scaler.inverse_transform(gmm.means_)
plt.scatter(centers[:, 1], centers[:, 0], s=100, c='red', label='Cluster Centers', marker='x')
plt.title('Gaussian Mixture Model Clustering: Total vs Gold Medals')
plt.xlabel('Total Medals')
plt.ylabel('Gold Medals')
plt.legend()
plt.show()

# Calculate silhouette score
silhouette_avg = silhouette_score(X_normalized, clusters)
print(f'Silhouette Score for GMM: {silhouette_avg}')

# Calculate Davies-Bouldin score
db_score = davies_bouldin_score(X_normalized, clusters)
print(f'Davies-Bouldin Score for GMM: {db_score}')
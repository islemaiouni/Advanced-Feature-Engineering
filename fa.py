import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import category_encoders as ce

# Sample Data
np.random.seed(42)
data = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C'], size=100),
    'Feature1': np.random.randn(100),
    'Feature2': np.random.randn(100),
    'Target': np.random.choice([0, 1], size=100)
})

# 1. Target Encoding
encoder = ce.TargetEncoder(cols=['Category'])
data['Category_encoded'] = encoder.fit_transform(data['Category'], data['Target'])

# 2. Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(data[['Feature1', 'Feature2']])
data_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['Feature1', 'Feature2']))

# 3. Binning
binner = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
data['Feature1_binned'] = binner.fit_transform(data[['Feature1']])

# 4. Time-Based Features (Example with Date)
data['Date'] = pd.date_range(start='1/1/2023', periods=100)
data['Month'] = data['Date'].dt.month
data['DayOfWeek'] = data['Date'].dt.dayofweek

# 5. Feature Interactions
data['Feature_Interaction'] = data['Feature1'] * data['Feature2']

# 6. Embedding Representation (Using Dummy Variables as Example)
data = pd.get_dummies(data, columns=['Category'])

# 7. Clustering as Features
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['Feature1', 'Feature2']])

# 8. Dimensionality Reduction (PCA)
pca = PCA(n_components=1)
data['PCA_Feature'] = pca.fit_transform(data[['Feature1', 'Feature2']])

# 9. Feature Selection with Mutual Information
X_train, X_test, y_train, y_test = train_test_split(data[['Feature1', 'Feature2']], data['Target'], test_size=0.2, random_state=42)
mi_scores = mutual_info_classif(X_train, y_train)
mi_results = pd.DataFrame({'Feature': ['Feature1', 'Feature2'], 'Mutual_Info': mi_scores})
print(mi_results.sort_values(by='Mutual_Info', ascending=False))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\nandu\Downloads\india_road_accidents_2025_10k (1).csv")
print(df.head())
print("Dataset Shape:", df.shape)
print(df.info())
print(df.isnull().sum())

df.drop_duplicates(inplace=True)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
numeric_cols = ["Speed_Limit_KMH", "Fatalities", "Injuries"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df[col] < 0, col] = np.nan
    df[col] = df[col].fillna(df[col].median())
df.dropna(subset=["Timestamp"], inplace=True)
categorical_cols = ["State", "Road_Type", "Primary_Vehicle", "Primary_Cause", "Weather_Condition"]
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    df[col] = df[col].replace("", "Unknown")
print("\nCleaned Dataset:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())

df["Year"] = df["Timestamp"].dt.year
df["Month"] = df["Timestamp"].dt.month
df["Hour"] = df["Timestamp"].dt.hour
df["Day"] = df["Timestamp"].dt.dayofweek
df["Accident_Severity"] = np.select([df["Fatalities"] > 0, df["Injuries"] > 0], ["Fatal", "Injury"], default="No Casualty")

plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Accident_Severity")
plt.title("Accident Severity Distribution")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
df["State"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 States by Accident Records")
plt.ylabel("Accident Records")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
df["Primary_Cause"].value_counts().head(10).plot(kind="bar")
plt.title("Top Accident Causes")
plt.ylabel("Accident Records")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
df["Hour"].value_counts().sort_index().plot(kind="bar")
plt.title("Accidents by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Accident Records")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="Weather_Condition", hue="Accident_Severity")
plt.title("Accident Severity by Weather")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="Road_Type", hue="Accident_Severity")
plt.title("Accident Severity by Road Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(df[["Speed_Limit_KMH", "Fatalities", "Injuries", "Year", "Month", "Hour"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nObjective 1: Accident Severity")
print(df["Accident_Severity"].value_counts())
print("\nObjective 2: Top Accident Causes")
print(df["Primary_Cause"].value_counts().head(10))
print("\nObjective 3: Top States")
print(df["State"].value_counts().head(10))
print("\nObjective 4: Weather Analysis")
print(pd.crosstab(df["Weather_Condition"], df["Accident_Severity"]))
print("\nObjective 5: Road Type Analysis")
print(pd.crosstab(df["Road_Type"], df["Accident_Severity"]))

X_reg = df[["Speed_Limit_KMH", "Year", "Month", "Hour"]]
y_reg = df["Fatalities"]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg_model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LinearRegression())])
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)
print("\nLinear Regression")
print("MAE:", mean_absolute_error(y_test_reg, y_pred_reg))
print("MSE:", mean_squared_error(y_test_reg, y_pred_reg))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))
print("R2 Score:", r2_score(y_test_reg, y_pred_reg))

poly_model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("poly", PolynomialFeatures(degree=2, include_bias=False)), ("model", LinearRegression())])
poly_model.fit(X_train_reg, y_train_reg)
poly_pred = poly_model.predict(X_test_reg)
print("\nPolynomial Regression")
print("MAE:", mean_absolute_error(y_test_reg, poly_pred))
print("MSE:", mean_squared_error(y_test_reg, poly_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, poly_pred)))
print("R2 Score:", r2_score(y_test_reg, poly_pred))

features = ["State", "Road_Type", "Primary_Vehicle", "Primary_Cause", "Weather_Condition", "Speed_Limit_KMH", "Year", "Month", "Hour", "Day"]
y = df["Accident_Severity"]
valid_classes = y.value_counts()
valid_classes = valid_classes[valid_classes >= 2].index
data = df[y.isin(valid_classes)].copy()
X = data[features]
y = data["Accident_Severity"]
categorical_features = ["State", "Road_Type", "Primary_Vehicle", "Primary_Cause", "Weather_Condition"]
numeric_features = ["Speed_Limit_KMH", "Year", "Month", "Hour", "Day"]
preprocessor = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
models = {"Logistic Regression": LogisticRegression(max_iter=1000), "KNN": KNeighborsClassifier(n_neighbors=5), "Decision Tree": DecisionTreeClassifier(random_state=42), "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42), "SVM": SVC(probability=True, random_state=42), "Neural Network": MLPClassifier(max_iter=300, random_state=42), "AdaBoost": AdaBoostClassifier(random_state=42)}
results = []
for name, model in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average="macro", zero_division=0)
    recall = recall_score(y_test, pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    results.append({"Model": name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1})
    print("\n", name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print(classification_report(y_test, pred, zero_division=0))

results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
print("\nModel Comparison:")
print(results_df)
best_model_name = results_df.iloc[0]["Model"]
print("\nBest Model:", best_model_name)
best_pipeline = Pipeline([("preprocessor", preprocessor), ("model", models[best_model_name])])
best_pipeline.fit(X_train, y_train)
best_pred = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Best Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

cluster_features = ["Speed_Limit_KMH", "Fatalities", "Injuries"]
cluster_data = df[cluster_features].copy()
cluster_scaled = StandardScaler().fit_transform(cluster_data)
inertia = []
for k in range(1, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(cluster_scaled)
    inertia.append(km.inertia_)
plt.figure(figsize=(7, 4))
plt.plot(range(1, 8), inertia, marker="o")
plt.title("K-Means Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.tight_layout()
plt.show()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(cluster_scaled)
print("\nK-Means Cluster Analysis:")
print(df.groupby("KMeans_Cluster")[cluster_features].mean())

sample_size = min(2000, len(df))
sample_data = df[cluster_features].sample(sample_size, random_state=42)
sample_scaled = StandardScaler().fit_transform(sample_data)
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(sample_scaled)
print("\nHierarchical Clustering:")
print(pd.Series(hierarchical_labels).value_counts())

print("\nAssociation Analysis")
print("\nTop Road Types:")
print(df["Road_Type"].value_counts().head(5))
print("\nTop Accident Causes:")
print(df["Primary_Cause"].value_counts().head(5))
print("\nWeather and Accident Severity:")
print(pd.crosstab(df["Weather_Condition"], df["Accident_Severity"]))

pca = PCA(n_components=2)
pca_data = pca.fit_transform(cluster_scaled)
pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
pca_df["Cluster"] = df["KMeans_Cluster"].values
plt.figure(figsize=(7, 5))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="Set1")
plt.title("PCA of Road Accident Data")
plt.tight_layout()
plt.show()
print("\nPCA Explained Variance:")
print(pca.explained_variance_ratio_)
print("Total Explained Variance:", pca.explained_variance_ratio_.sum())

train_pred = best_pipeline.predict(X_train)
test_pred = best_pipeline.predict(X_test)
print("\nBias and Variance Check")
print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Testing Accuracy:", accuracy_score(y_test, test_pred))

with pd.ExcelWriter("Road_Accident_Model_Performance.xlsx") as writer:
    results_df.to_excel(writer, sheet_name="Model Comparison", index=False)
    df.to_excel(writer, sheet_name="Cleaned Dataset", index=False)

print("\nExcel file saved successfully!")
print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("Total Accident Records:", len(df))
print("Total Features:", df.shape[1])
print("Best Classification Model:", best_model_name)

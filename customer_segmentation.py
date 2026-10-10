# ======================================================================
# Big Bazaar - Customer Segmentation using Machine Learning
# UCS321 MST Mini Project (Problem 3)
# ----------------------------------------------------------------------
# IDEA:
#   Big Bazaar has many customers. We want to put customers who behave
#   in a similar way into the same GROUP (called a "cluster"), so that
#   each group can get its own offers.
#   We do NOT tell the computer the groups. It finds them on its own.
#   This is called "unsupervised learning" (clustering).
#
# STEPS:
#   1. Load the data
#   2. Clean the data
#   3. Make the features we need and scale them
#   4. Look at the data with charts (EDA)
#   5. Decide how many groups (k) to make
#   6. Run the clustering models and compare them
#   7. Check that the groups are stable
#   8. Describe each group and give a marketing idea
#
# DATA: marketing_campaign.csv (Kaggle: Customer Personality Analysis)
#       Keep this file in the same folder as this python file.
# ======================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")              # save the charts as picture files
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

# colours used in all charts
COLORS = ["#13315C", "#1B998B", "#C8963E", "#B5465A", "#6C7A89", "#7B5EA7"]

K = 3             # number of groups we will make (reason is explained in step 5)
YEAR = 2014       # the year the data was collected (used to find age)


# ----------------------------------------------------------------------
# STEP 1: LOAD THE DATA
# ----------------------------------------------------------------------
raw = pd.read_csv("marketing_campaign.csv", sep="\t")    # file is tab separated

print("STEP 1: LOAD THE DATA")
print("Rows and columns:", raw.shape)
print("Missing values (only columns that have them):")
print(raw.isna().sum()[raw.isna().sum() > 0])


# ----------------------------------------------------------------------
# STEP 2: CLEAN THE DATA
# ----------------------------------------------------------------------
df = raw.copy()
rows_start = len(df)

# 2a. Remove duplicate customers.
#     Same details but a different ID means it is the same record twice.
columns_without_id = list(df.columns)
columns_without_id.remove("ID")
df = df.drop_duplicates(subset=columns_without_id)
duplicates_removed = rows_start - len(df)

# 2b. Remove impossible ages (for example someone born in 1893)
df["Age"] = YEAR - df["Year_Birth"]
rows_before = len(df)
df = df[df["Age"] <= 100]
age_errors_removed = rows_before - len(df)

# 2c. Remove the one very large income (666,666). It is an outlier.
rows_before = len(df)
df = df[df["Income"].isna() | (df["Income"] < 200000)]
income_outliers_removed = rows_before - len(df)

# 2d. Fill the missing incomes with the median (the middle value)
missing_income = df["Income"].isna().sum()
middle_income = df["Income"].median()
df["Income"] = df["Income"].fillna(middle_income)

df = df.reset_index(drop=True)

print("\nSTEP 2: CLEAN THE DATA")
print("Duplicates removed       :", duplicates_removed)
print("Wrong ages removed       :", age_errors_removed)
print("Income outlier removed   :", income_outliers_removed)
print("Missing income filled    :", missing_income)
print("Customers left after cleaning:", len(df))

# chart: income before and after cleaning
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
sns.boxplot(x=raw["Income"].dropna(), ax=ax[0], color=COLORS[4])
ax[0].set_title("Income - raw (outlier at 666,666)")
sns.boxplot(x=df["Income"], ax=ax[1], color=COLORS[1])
ax[1].set_title("Income - after cleaning")
plt.tight_layout()
plt.savefig("cleaning.png", dpi=130)
plt.close()


# ----------------------------------------------------------------------
# STEP 3: MAKE THE FEATURES AND SCALE THEM
# ----------------------------------------------------------------------
# Total spend = money spent on all 6 product types
spend_columns = ["MntWines", "MntFruits", "MntMeatProducts",
                 "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
df["Total_Spend"] = df[spend_columns].sum(axis=1)

# Purchase frequency = number of purchases on web + catalogue + store
df["Total_Purchases"] = (df["NumWebPurchases"] + df["NumCatalogPurchases"]
                         + df["NumStorePurchases"])

# Visit frequency = website visits in the last month
df["Web_Visits"] = df["NumWebVisitsMonth"]

# Kids at home (used later only to describe the groups)
df["Kids"] = df["Kidhome"] + df["Teenhome"]

# A few customers spend a LOT. Taking the log makes the spending
# more balanced, so those few customers do not dominate.
df["Spend_Log"] = np.log1p(df["Total_Spend"])

# The 4 features used to make the groups
features = ["Income", "Spend_Log", "Total_Purchases", "Web_Visits"]

# Scaling: income is in thousands but web visits are 0-20.
# K-Means uses distance, so we put all features on the same scale.
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

print("\nSTEP 3: FEATURES USED FOR CLUSTERING:", features)
print(df[["Income", "Total_Spend", "Total_Purchases", "Web_Visits",
          "Recency", "Age"]].describe().round(1))


# ----------------------------------------------------------------------
# STEP 4: LOOK AT THE DATA (EDA)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))

sns.histplot(df["Income"], bins=40, ax=ax[0, 0], color=COLORS[0])
ax[0, 0].set_title("Annual income")

sns.histplot(df["Total_Spend"], bins=40, ax=ax[0, 1], color=COLORS[1])
ax[0, 1].set_title("Total spend (2 years)")

sns.histplot(df["Web_Visits"], bins=20, ax=ax[0, 2], color=COLORS[2])
ax[0, 2].set_title("Web visits per month")

sns.histplot(df["Total_Purchases"], bins=30, ax=ax[1, 0], color=COLORS[3])
ax[1, 0].set_title("Total purchases")

sns.scatterplot(x="Income", y="Total_Spend", data=df, ax=ax[1, 1],
                s=12, color=COLORS[0], alpha=0.6)
ax[1, 1].set_title("Income vs total spend")

# correlation = how strongly two columns move together (-1 to +1)
corr_columns = ["Income", "Total_Spend", "Total_Purchases", "Web_Visits",
                "Recency", "Age"]
sns.heatmap(df[corr_columns].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax[1, 2], annot_kws={"size": 7}, cbar=False)
ax[1, 2].set_title("Correlation")
ax[1, 2].tick_params(labelsize=7)

plt.tight_layout()
plt.savefig("eda.png", dpi=130)
plt.close()


# ----------------------------------------------------------------------
# STEP 5: HOW MANY GROUPS (k)?
# We try k = 2 to 8 and look at the scores for each.
# ----------------------------------------------------------------------
k_values = [2, 3, 4, 5, 6, 7, 8]
inertia_list = []        # how tight the groups are (smaller = tighter)
silhouette_list = []     # how well separated (bigger = better)
db_list = []             # overlap between groups (smaller = better)
ch_list = []             # separation score (bigger = better)

for k in k_values:
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X)
    inertia_list.append(model.inertia_)
    silhouette_list.append(silhouette_score(X, model.labels_))
    db_list.append(davies_bouldin_score(X, model.labels_))
    ch_list.append(calinski_harabasz_score(X, model.labels_))

print("\nSTEP 5: CHOOSING k")
print("k  inertia  silhouette  DB     CH")
for i in range(len(k_values)):
    print(k_values[i], " ", round(inertia_list[i], 1), " ",
          round(silhouette_list[i], 3), "     ",
          round(db_list[i], 3), " ", round(ch_list[i], 1))

best_k = k_values[silhouette_list.index(max(silhouette_list))]
print("Best k by silhouette score:", best_k)
print("We choose k =", K, "because:")
print(" - the elbow curve bends at 3")
print(" - 2 groups (rich / poor) is too simple for different offers")
print(" - the silhouette score is still reasonable at 3")

# chart: elbow curve and silhouette curve
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(k_values, inertia_list, "o-", color=COLORS[0])
ax[0].axvline(K, ls="--", color=COLORS[3])
ax[0].set_title("Elbow method")
ax[0].set_xlabel("k")
ax[0].set_ylabel("Inertia (WCSS)")
ax[1].plot(k_values, silhouette_list, "o-", color=COLORS[1])
ax[1].axvline(K, ls="--", color=COLORS[3])
ax[1].set_title("Silhouette score")
ax[1].set_xlabel("k")
ax[1].set_ylabel("Silhouette")
plt.tight_layout()
plt.savefig("elbow.png", dpi=130)
plt.close()


# ----------------------------------------------------------------------
# STEP 6: RUN THE CLUSTERING MODELS
# ----------------------------------------------------------------------
# Model 1: K-Means (our main model)
kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
kmeans.fit(X)

# K-Means gives group numbers 0,1,2 in random order.
# We renumber them so that group 0 = highest spenders, 2 = lowest.
avg_spend = pd.Series(df["Total_Spend"].values).groupby(kmeans.labels_).mean()
old_order = list(avg_spend.sort_values(ascending=False).index)

new_number = {}
for new, old in enumerate(old_order):
    new_number[old] = new

labels = np.array([new_number[old] for old in kmeans.labels_])
centroids = kmeans.cluster_centers_[old_order]    # centre of each group

# Model 2: Hierarchical clustering (Ward)
hier = AgglomerativeClustering(n_clusters=K, linkage="ward")
hier.fit(X)

# Model 3: Gaussian Mixture
gmm = GaussianMixture(n_components=K, random_state=42, n_init=5)
gmm.fit(X)

# Model 4: DBSCAN
dbscan = DBSCAN(eps=0.55, min_samples=10)
dbscan.fit(X)

# Compare the first three models with 3 scores
all_labels = {"K-Means": labels,
              "Hierarchical (Ward)": hier.labels_,
              "Gaussian Mixture": gmm.predict(X)}

print("\nSTEP 6: MODEL COMPARISON")
print("Model                  Silhouette  Davies-Bouldin  Calinski-Harabasz")
scores = {}
for name in all_labels:
    lab = all_labels[name]
    sil = silhouette_score(X, lab)
    dbi = davies_bouldin_score(X, lab)
    chi = calinski_harabasz_score(X, lab)
    scores[name] = [sil, dbi, chi]
    print(name.ljust(22), round(sil, 3), "      ", round(dbi, 3),
          "         ", round(chi, 1))

# DBSCAN: -1 means "noise" (customer that fits no group)
noise = int((dbscan.labels_ == -1).sum())
n_dbscan = len(set(dbscan.labels_)) - 1       # -1 removes the noise label
print("DBSCAN found", n_dbscan, "cluster and", noise, "noise customers")

# Try other DBSCAN settings to be fair to it
print("DBSCAN with other settings:")
for eps, min_pts in [(0.3, 5), (0.4, 5), (0.4, 10), (0.55, 10)]:
    lab = DBSCAN(eps=eps, min_samples=min_pts).fit(X).labels_
    n_clusters = len(set(lab)) - 1
    n_noise = int((lab == -1).sum())
    print("  eps =", eps, " min_samples =", min_pts, "->", n_clusters,
          "clusters,", n_noise, "noise")
print("DBSCAN gives either 1 big cluster or many tiny ones, so it is not suitable.")
print("K-Means has the best scores, so we use K-Means.")


# ----------------------------------------------------------------------
# STEP 7: ARE THE GROUPS STABLE?
# We take 30 random samples of the customers, run K-Means on each,
# and check if the groups come out the same (ARI close to 1 = same).
# ----------------------------------------------------------------------
rng = np.random.default_rng(42)
ari_list = []
for i in range(30):
    sample_rows = rng.choice(len(X), len(X), replace=True)
    sample_model = KMeans(n_clusters=K, n_init=10, random_state=0)
    sample_model.fit(X[sample_rows])
    ari = adjusted_rand_score(labels, sample_model.predict(X))
    ari_list.append(ari)

stability = np.mean(ari_list)
print("\nSTEP 7: STABILITY (average ARI over 30 samples):", round(stability, 3))


# ----------------------------------------------------------------------
# STEP 8: DESCRIBE EACH GROUP AND GIVE MARKETING IDEAS
# ----------------------------------------------------------------------
df["Cluster"] = labels

# names we give to the groups (0 = highest spending)
names = ["Premium Loyalists", "Steady Mainstream", "Budget Browsers"]

# average of each column for every group
describe_columns = ["Income", "Total_Spend", "Total_Purchases", "Web_Visits",
                    "Recency", "Age", "Kids", "Response"]
profile = df.groupby("Cluster")[describe_columns].mean().round(2)
profile["Response"] = (profile["Response"] * 100).round(1)   # make it a percentage
profile.insert(0, "Customers", df["Cluster"].value_counts().sort_index())
profile.insert(0, "Segment", names)

print("\nSTEP 8: SEGMENT PROFILES")
print(profile.to_string())

# share of customers and share of money spent
customer_share = df["Cluster"].value_counts(normalize=True).sort_index() * 100
spend_share = df.groupby("Cluster")["Total_Spend"].sum() / df["Total_Spend"].sum() * 100
print("\nShare of customers (%):", customer_share.round(1).to_dict())
print("Share of total spend (%):", spend_share.round(1).to_dict())

print("\nMARKETING IDEAS:")
print("Premium Loyalists : VIP loyalty tier, early access to new and premium ranges,")
print("                    catalogue and in-store events")
print("Steady Mainstream : Bundle offers, cross-sell between categories,")
print("                    loyalty points to lift purchase frequency")
print("Budget Browsers   : Web-only coupons and flash deals, price-match messaging,")
print("                    convert visits into purchases")


# ----------------------------------------------------------------------
# CHARTS FOR THE RESULTS
# ----------------------------------------------------------------------
# PCA squeezes 4 features into 2 so we can draw the groups on paper
pca = PCA(n_components=2)
pca.fit(X)
Z = pca.transform(X)
print("\nPCA: 2 components keep", round(100 * pca.explained_variance_ratio_.sum(), 1),
      "% of the information")

# Chart 1: groups in PCA view
plt.figure(figsize=(7, 5))
for c in range(K):
    rows = labels == c
    plt.scatter(Z[rows, 0], Z[rows, 1], s=14, color=COLORS[c],
                label=names[c], alpha=0.65)
centre_points = pca.transform(centroids)
plt.scatter(centre_points[:, 0], centre_points[:, 1], marker="X", s=200,
            color="black", label="Centroids")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means segments in PCA space (k=3)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("pca.png", dpi=130)
plt.close()

# Chart 2: income vs spend coloured by group
plt.figure(figsize=(7, 5))
for c in range(K):
    rows = df["Cluster"] == c
    plt.scatter(df.loc[rows, "Income"], df.loc[rows, "Total_Spend"], s=14,
                color=COLORS[c], label=names[c], alpha=0.65)
plt.xlabel("Annual income")
plt.ylabel("Total spend (2 years)")
plt.title("Segments: income vs total spend")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("clusters.png", dpi=130)
plt.close()

# Chart 3: dendrogram (a family tree of how customers merge into groups)
# We use 400 random customers so that the tree is readable.
sample_rows = np.random.default_rng(0).choice(len(X), 400, replace=False)
tree = linkage(X[sample_rows], method="ward")
cut_height = tree[-(K - 1), 2] - 0.000001        # height that gives 3 groups
plt.figure(figsize=(7.5, 4.5))
dendrogram(tree, truncate_mode="lastp", p=24, color_threshold=cut_height,
           above_threshold_color="#8C8C8C", leaf_rotation=90,
           leaf_font_size=8, show_leaf_counts=True)
plt.axhline(cut_height, ls="--", color=COLORS[3], lw=1)
plt.title("Dendrogram (Ward linkage, 400-customer sample)")
plt.xlabel("Cluster size")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("dendrogram.png", dpi=130)
plt.close()

# Chart 4: silhouette of every customer (long bar = fits its group well)
sil_values = silhouette_samples(X, labels)
plt.figure(figsize=(7, 5))
y_start = 10
for c in range(K):
    values = np.sort(sil_values[labels == c])
    plt.fill_betweenx(np.arange(y_start, y_start + len(values)), 0, values,
                      color=COLORS[c], alpha=0.85)
    plt.text(-0.05, y_start + len(values) / 2, names[c], fontsize=8,
             ha="right", va="center")
    y_start = y_start + len(values) + 40
plt.axvline(silhouette_score(X, labels), ls="--", color="black", lw=1)
plt.yticks([])
plt.xlabel("Silhouette coefficient")
plt.title("Silhouette analysis (dashed = average)")
plt.xlim(-0.2, 1)
plt.tight_layout()
plt.savefig("silhouette_plot.png", dpi=130)
plt.close()

# Chart 5: heatmap - how far is each group from the overall average?
# (red = above average, blue = below average)
heat_columns = ["Income", "Total_Spend", "Total_Purchases", "Web_Visits",
                "Recency", "Age", "Kids"]
group_means = df.groupby("Cluster")[heat_columns].mean()
heat = (group_means - df[heat_columns].mean()) / df[heat_columns].std()
heat.index = names
heat.columns = ["Income", "Spend", "Purchases", "Web visits",
                "Recency", "Age", "Kids"]
plt.figure(figsize=(7, 3.6))
sns.heatmap(heat, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
            vmin=-1.5, vmax=1.5, cbar_kws={"label": "Std. deviations from average"})
plt.title("Segment profile heatmap")
plt.tight_layout()
plt.savefig("segment_heatmap.png", dpi=130)
plt.close()

# Chart 6: share of customers vs share of spending
fig, ax = plt.subplots(figsize=(6.5, 3.8))
width = 0.38
positions = np.arange(K)
ax.bar(positions - width / 2, customer_share.values, width,
       color=COLORS[4], label="% of customers")
ax.bar(positions + width / 2, spend_share.values, width,
       color=COLORS[0], label="% of total spend")
ax.set_xticks(positions)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("%")
ax.set_title("Customer share vs revenue share")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("share.png", dpi=130)
plt.close()

# Save the final table with the group of every customer
df.drop(columns=["Spend_Log"]).to_csv("segmented_customers.csv", index=False)
print("\nDone. Charts and segmented_customers.csv are saved in this folder.")

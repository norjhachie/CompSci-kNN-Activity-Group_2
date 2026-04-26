# =============================================================================
# HOMEWORK: K-Nearest Neighbors (KNN) on Diabetes Dataset
# University of Southern Mindanao
# College of Engineering and Information Technology
# Department of Computing and Information Science
# Course: Computational Science for Computer Science
# Dataset: diabetes-k-nn.csv (768 patient records)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression


# =============================================================================
# PART 1: DATA UNDERSTANDING
# =============================================================================

# Load the dataset from CSV
df = pd.read_csv('diabetes-k-nn.csv')

print("=" * 60)
print("PART 1: DATA UNDERSTANDING")
print("=" * 60)

# Show basic dataset info
print(f"\nTotal records  : {len(df)}")
print(f"Total columns  : {df.shape[1]}  (8 features + 1 target)")

# Show how many patients are diabetic vs non-diabetic
print("\nClass Distribution:")
counts = df['Outcome'].value_counts().sort_index()
for label, count in counts.items():
    name = 'Non-Diabetic' if label == 0 else 'Diabetic'
    print(f"  {label} = {name}: {count} patients ({count/len(df)*100:.1f}%)")

# Show the first 5 rows so we can see what the data looks like
print("\nFirst 5 rows of the dataset:")
print(df.head().to_string())

# Show summary statistics (count, mean, min, max, etc.) for each column
print("\nSummary Statistics:")
print(df.describe().round(2).to_string())


# =============================================================================
# PART 2: DATA PREPROCESSING
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: DATA PREPROCESSING")
print("=" * 60)

# -----------------------------------------------------------------------------
# STEP 1 — Find zero values (zeros in these columns mean data is missing)
# -----------------------------------------------------------------------------

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("\n[ Step 1 ] Counting zero values in each feature:")
print(f"  {'Feature':<30} {'Zeros':>6} {'% of Data':>10}")
print("  " + "-" * 50)
for col in zero_cols:
    cnt = (df[col] == 0).sum()
    pct = cnt / len(df) * 100
    print(f"  {col:<30} {cnt:>6} {pct:>9.1f}%")

# Chart — bar chart showing how many zeros each feature has
fig, ax = plt.subplots(figsize=(8, 5))
zero_counts = [(df[col] == 0).sum() for col in zero_cols]
bars = ax.bar(zero_cols, zero_counts,
              color=['#5B9BD5', '#5B9BD5', '#E06C2E', '#C00000', '#5B9BD5'],
              edgecolor='white', width=0.6)
for bar, cnt in zip(bars, zero_counts):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4,
            f'{cnt}  ({cnt/768*100:.1f}%)',
            ha='center', fontsize=10, fontweight='bold')
ax.set_title('Zero Value Counts per Feature (Before Imputation)',
             fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel('Number of Zero Values', fontsize=11)
ax.set_ylim(0, max(zero_counts) * 1.3)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('p2_zeros.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  [Saved] p2_zeros.png")

# -----------------------------------------------------------------------------
# STEP 2 — Replace zero values using Median Imputation
# -----------------------------------------------------------------------------
# We use the median (not the mean) because features like Insulin have very high
# outliers that pull the mean too high. The median is the middle value of the
# sorted non-zero data, so it is not affected by those extreme values.

print("\n[ Step 2 ] Replacing zeros with the median of each feature:")
df_clean = df.copy()
medians_used = {}

for col in zero_cols:
    if (df_clean[col] == 0).sum() > 0:
        # Calculate median from valid (non-zero) values only
        median_val = df_clean[col].replace(0, np.nan).median()
        df_clean[col] = df_clean[col].replace(0, median_val)
        medians_used[col] = median_val
        print(f"  {col:<30}: replaced zeros with median = {median_val}")

# Confirm all zeros are gone
print("\n  Verification (zeros remaining after imputation):")
for col in zero_cols:
    print(f"  {col:<30}: {(df_clean[col] == 0).sum()} zeros left")

# Chart — grouped bar chart showing before vs after for SkinThickness and Insulin
# This is easier to read than box plots: you can directly compare the numbers
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

stats_labels = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
plot_cols = ['SkinThickness', 'Insulin']

for ax, col in zip(axes, plot_cols):
    before_vals = [
        df[col].mean(),
        df[col].median(),
        df[col].std(),
        df[col].min(),
        df[col].max()
    ]
    after_vals = [
        df_clean[col].mean(),
        df_clean[col].median(),
        df_clean[col].std(),
        df_clean[col].min(),
        df_clean[col].max()
    ]

    x = np.arange(len(stats_labels))
    width = 0.35

    bars_before = ax.bar(x - width/2, before_vals, width,
                         label='Before Imputation', color='#C0392B', alpha=0.85, edgecolor='white')
    bars_after  = ax.bar(x + width/2, after_vals,  width,
                         label='After Imputation',  color='#2E5090', alpha=0.85, edgecolor='white')

    # Add value labels on top of each bar
    for bar in bars_before:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{bar.get_height():.1f}',
                ha='center', va='bottom', fontsize=8, color='#C0392B', fontweight='bold')
    for bar in bars_after:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{bar.get_height():.1f}',
                ha='center', va='bottom', fontsize=8, color='#2E5090', fontweight='bold')

    ax.set_title(f'{col}: Before vs After Imputation',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stats_labels, fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Effect of Median Imputation — Statistics Before vs After',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('p2_imputation.png', dpi=150, bbox_inches='tight')
plt.show()
print("  [Saved] p2_imputation.png")

# -----------------------------------------------------------------------------
# STEP 3 — Feature Scaling using Z-score Standardization
# -----------------------------------------------------------------------------
# KNN measures distance between records. Without scaling, features with large
# ranges (Insulin: 14–846) would control the distance more than features with
# small ranges (DiabetesPedigreeFunction: 0.08–2.42). Scaling fixes this by
# putting all features on the same unit: mean = 0, std = 1.
# Formula: z = (value - mean) / standard_deviation

X = df_clean.drop('Outcome', axis=1)   # input features
y = df_clean['Outcome']                 # target column

print("\n[ Step 3 ] Applying Z-score Standardization:")
print(f"  {'Feature':<30} {'Mean Before':>12} {'Std Before':>11} {'Mean After':>11} {'Std After':>10}")
print("  " + "-" * 78)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

for col in X.columns:
    print(f"  {col:<30} {X[col].mean():>12.4f} {X[col].std():>11.4f} "
          f"{X_scaled[col].mean():>11.6f} {X_scaled[col].std():>10.6f}")

# Chart — before vs after scaling histograms for 4 features
show_feats = ['Glucose', 'Insulin', 'BMI', 'Age']
fig, axes = plt.subplots(2, 4, figsize=(14, 6))

for i, feat in enumerate(show_feats):
    # Top row: original values
    axes[0, i].hist(X[feat], bins=25, color='#5B9BD5', edgecolor='white', alpha=0.85)
    axes[0, i].set_title(f'{feat}\n(Before Scaling)', fontsize=10, fontweight='bold')
    axes[0, i].set_xlabel('Original Value', fontsize=9)
    if i == 0:
        axes[0, i].set_ylabel('Count', fontsize=10)

    # Bottom row: scaled z-scores
    axes[1, i].hist(X_scaled[feat], bins=25, color='#E06C2E', edgecolor='white', alpha=0.85)
    axes[1, i].set_title(f'{feat}\n(After Scaling)', fontsize=10, fontweight='bold')
    axes[1, i].set_xlabel('z-score', fontsize=9)
    if i == 0:
        axes[1, i].set_ylabel('Count', fontsize=10)

    for row in [0, 1]:
        axes[row, i].grid(alpha=0.2)
        axes[row, i].spines['top'].set_visible(False)
        axes[row, i].spines['right'].set_visible(False)

fig.suptitle('Before (Blue) vs After Scaling (Orange) — Shape stays the same, range changes',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('p2_scaling.png', dpi=150, bbox_inches='tight')
plt.show()
print("  [Saved] p2_scaling.png")

# Chart — feature distributions by Outcome (diabetic vs non-diabetic)
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, feat in zip(axes.flatten(), X.columns):
    df_clean[df_clean['Outcome'] == 0][feat].hist(
        ax=ax, bins=20, alpha=0.65, label='Non-Diabetic', color='#5B9BD5', edgecolor='white')
    df_clean[df_clean['Outcome'] == 1][feat].hist(
        ax=ax, bins=20, alpha=0.65, label='Diabetic', color='#E06C2E', edgecolor='white')
    ax.set_title(feat, fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.suptitle('Feature Distributions by Outcome (After Preprocessing)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('p2_feat_dist.png', dpi=150, bbox_inches='tight')
plt.show()
print("  [Saved] p2_feat_dist.png")


# =============================================================================
# PART 3: KNN IMPLEMENTATION
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: KNN IMPLEMENTATION")
print("=" * 60)

# -----------------------------------------------------------------------------
# STEP 1 — Split the data: 80% for training, 20% for testing
# -----------------------------------------------------------------------------
# random_state=42 makes sure we get the same split every time we run the code

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled.values, y.values, test_size=0.2, random_state=42
)

print(f"\nTraining set : {len(X_train)} records (80%)")
print(f"Test set     : {len(X_test)} records (20%)")

# -----------------------------------------------------------------------------
# STEP 2 — Train KNN models for K = 3, 5, and 7
# -----------------------------------------------------------------------------
# For each K, the model finds the K nearest neighbors of a test record
# and predicts the class that the majority of those neighbors belong to.
# Distance is measured using the Euclidean formula:
#   d(x, y) = sqrt( (x1-y1)^2 + (x2-y2)^2 + ... + (x8-y8)^2 )

k_values  = [3, 5, 7]
knn_models = {}
results    = {}
cm_dict    = {}

print("\nTraining KNN for K = 3, 5, 7:")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    knn_models[k] = knn
    results[k]    = acc
    cm_dict[k]    = cm

    print(f"\n  K = {k}")
    print(f"    Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# -----------------------------------------------------------------------------
# STEP 3 — Manual distance computation for Test Instance #0
# -----------------------------------------------------------------------------
# We pick the first test record and manually compute its distance to every
# training record to show exactly how KNN works behind the scenes.

test_inst    = X_test[0]
actual_label = y_test[0]
feat_names   = list(X_scaled.columns)

print("\n--- Manual Euclidean Distance: Test Instance #0 ---")
print(f"\n  Feature values (scaled):")
print(f"  {'Feature':<30} {'z-score':>10}")
print("  " + "-" * 42)
for fn, fv in zip(feat_names, test_inst):
    print(f"  {fn:<30} {fv:>10.4f}")
print(f"\n  Actual label: {actual_label}  ({'Diabetic' if actual_label == 1 else 'Non-Diabetic'})")

# Compute Euclidean distance to every training sample
all_distances = []
for i in range(len(X_train)):
    diff_sq = (test_inst - X_train[i]) ** 2
    dist    = np.sqrt(diff_sq.sum())
    all_distances.append((dist, int(y_train[i]), i))

all_distances.sort(key=lambda x: x[0])

# Show full step-by-step breakdown for the closest neighbor (Rank 1)
print("\n  Step-by-step calculation to Nearest Neighbor (Rank 1):")
print(f"  {'Feature':<30} {'Test (xi)':>10} {'Train (yi)':>11} {'(xi-yi)':>10} {'(xi-yi)^2':>11}")
print("  " + "-" * 76)
running_sum = 0.0
nn_features = X_train[all_distances[0][2]]
for fn, xi, yi in zip(feat_names, test_inst, nn_features):
    diff    = xi - yi
    sq      = diff ** 2
    running_sum += sq
    print(f"  {fn:<30} {xi:>10.4f} {yi:>11.4f} {diff:>10.4f} {sq:>11.4f}")
print("  " + "-" * 76)
print(f"  {'Sum of squares':<30} {'':>33} {running_sum:>11.4f}")
print(f"  {'Euclidean Distance':<30} {'sqrt(' + f'{running_sum:.4f}' + ')':>33} = {np.sqrt(running_sum):>6.4f}")

# Show the top 10 nearest neighbors
print("\n  Top 10 Nearest Neighbors:")
print(f"  {'Rank':<6} {'Distance':>10} {'Class':>8}  {'Label':<20} {'K=3':>5} {'K=5':>5} {'K=7':>5}")
print("  " + "-" * 65)
for rank, (dist, label, _) in enumerate(all_distances[:10]):
    lbl = 'Diabetic (1)' if label == 1 else 'Non-Diabetic (0)'
    print(f"  {rank+1:<6} {dist:>10.4f} {label:>8}  {lbl:<20} "
          f"{'Yes' if rank < 3 else 'No':>5} "
          f"{'Yes' if rank < 5 else 'No':>5} "
          f"{'Yes' if rank < 7 else 'No':>5}")

# Show majority vote result for each K
print("\n  Majority Vote Results:")
for k in k_values:
    votes   = [lbl for _, lbl, _ in all_distances[:k]]
    v0, v1  = votes.count(0), votes.count(1)
    pred    = 1 if v1 > v0 else 0
    correct = 'CORRECT' if pred == actual_label else 'INCORRECT'
    print(f"  K={k}: Class 0 = {v0} vote(s), Class 1 = {v1} vote(s) "
          f"→ Predicted: {pred} | Actual: {actual_label} | {correct}")


# =============================================================================
# PART 4: MODEL EVALUATION
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: MODEL EVALUATION")
print("=" * 60)

# Print accuracy table for all K values
print(f"\n  {'K':<6} {'Accuracy':>10} {'Correct/154':>12} {'TN':>5} {'FP':>5} {'FN':>5} {'TP':>5}")
print("  " + "-" * 52)
for k in k_values:
    tn, fp, fn, tp = cm_dict[k].ravel()
    print(f"  K={k:<4} {results[k]:>10.4f} {tn+tp:>10}/154  {tn:>5} {fp:>5} {fn:>5} {tp:>5}")

print("\n  TN = correctly predicted Non-Diabetic")
print("  FP = predicted Diabetic but actually Non-Diabetic")
print("  FN = predicted Non-Diabetic but actually Diabetic  ← most dangerous in medicine")
print("  TP = correctly predicted Diabetic")

# Print classification report for each K
for k in k_values:
    print(f"\n  Classification Report (K={k}):")
    print(classification_report(y_test, knn_models[k].predict(X_test),
                                target_names=['Non-Diabetic', 'Diabetic']))

# Best K answer
best_k = max(results, key=results.get)
print(f"\n  Best K = {best_k}  (Accuracy = {results[best_k]:.4f} / {results[best_k]*100:.2f}%)")

# Chart — Accuracy vs K line graph
fig, ax = plt.subplots(figsize=(6, 4))
ks   = list(results.keys())
accs = [results[k] for k in ks]
ax.plot(ks, accs, 'o-', color='#2E5090', linewidth=2.5, markersize=10,
        markerfacecolor='#E06C2E', markeredgecolor='#2E5090', markeredgewidth=1.5)
for k, a in zip(ks, accs):
    ax.annotate(f'{a*100:.2f}%', (k, a),
                textcoords='offset points', xytext=(0, 14),
                ha='center', fontsize=10, fontweight='bold', color='#2E5090')
ax.set_xticks(ks)
ax.set_xlabel('K Value', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('KNN Accuracy vs K Value\n(Test Set: 154 records)', fontsize=12, fontweight='bold')
ax.set_ylim(0.65, 0.80)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('p4_accuracy_k.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  [Saved] p4_accuracy_k.png")


# =============================================================================
# BONUS: KNN vs LOGISTIC REGRESSION
# =============================================================================

print("\n" + "=" * 60)
print("BONUS: KNN vs LOGISTIC REGRESSION")
print("=" * 60)

# Train Logistic Regression on the same scaled data and split
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc  = accuracy_score(y_test, lr_pred)

# Print comparison table
print(f"\n  {'Model':<25} {'Accuracy':>10} {'Correct/154':>12}")
print("  " + "-" * 50)
for k in k_values:
    tn, fp, fn, tp = cm_dict[k].ravel()
    marker = ' <- BEST KNN' if k == best_k else ''
    print(f"  KNN (K={k}){'':<16} {results[k]:>10.4f} {tn+tp:>10}/154{marker}")
print(f"  {'Logistic Regression':<25} {lr_acc:>10.4f} "
      f"{(confusion_matrix(y_test,lr_pred).ravel()[0] + confusion_matrix(y_test,lr_pred).ravel()[3]):>10}/154")

print(f"\n  Logistic Regression is {(lr_acc - results[best_k])*100:.2f}% more accurate than best KNN.")

# Chart — bar chart comparing all models
fig, ax = plt.subplots(figsize=(7, 4))
labels    = ['KNN\n(K=3)', 'KNN\n(K=5)', 'KNN\n(K=7)', 'Logistic\nRegression']
acc_vals  = [results[3], results[5], results[7], lr_acc]
colors    = ['#5B9BD5', '#2E5090', '#5B9BD5', '#E06C2E']
bars = ax.bar(labels, acc_vals, color=colors, edgecolor='white', width=0.5)
ax.set_ylim(0.65, 0.82)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy Comparison: KNN vs Logistic Regression', fontsize=12, fontweight='bold')
for bar, a in zip(bars, acc_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f'{a*100:.2f}%',
            ha='center', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('bonus_lr.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  [Saved] bonus_lr.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Dataset       : diabetes-k-nn.csv")
print(f"  Records       : 768  (Non-Diabetic: 500, Diabetic: 268)")
print(f"  Train / Test  : {len(X_train)} / {len(X_test)}")
print(f"\n  Preprocessing :")
print(f"    Zeros fixed : Glucose(5), BP(35), Skin(227), Insulin(374), BMI(11)")
print(f"    Imputation  : Median — Glucose=117, BP=72, Skin=29, Insulin=125, BMI=32.3")
print(f"    Scaling     : Z-score (mean=0, std=1)")
print(f"\n  KNN Results   :")
for k in k_values:
    tag = ' <- BEST' if k == best_k else ''
    print(f"    K={k} : {results[k]*100:.2f}%{tag}")
print(f"\n  Logistic Reg  : {lr_acc*100:.2f}%")
print(f"\n  Charts saved  : p2_zeros.png, p2_imputation.png,")
print(f"                  p2_scaling.png, p2_feat_dist.png,")
print(f"                  p4_accuracy_k.png, bonus_lr.png")
print("=" * 60)

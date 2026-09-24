from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from config import DATASET, SEED
import utils
 
# ---------------------------------------------------------------
# 1. Fetch the dataset (UCI repo ID 17 = Breast Cancer Wisconsin Diagnostic)
# ---------------------------------------------------------------
dataset = fetch_ucirepo(id=DATASET)
 
X = dataset.data.features
y = dataset.data.targets

# y often comes back as a DataFrame with one column; flatten to a Series
if hasattr(y, "columns"):
    y = y.iloc[:, 0]

n_vals = utils.vals


for k in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    for j in range(len(n_vals)):
    # Split training and testind data
    
    
    # Standardizing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    
    # ---------------------------------------------------------------
    # 5. Evaluate
    # ---------------------------------------------------------------
    y_pred = clf.predict(X_test_scaled)
    
    print("\nTest accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
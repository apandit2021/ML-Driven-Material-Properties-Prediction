import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
import os # Ensure this line is present
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
import json

# --- Add PyTorch Geometric Tabular GNN Model ---
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler

# --- 1. GNN Model ---
class SimpleTabularGNN(torch.nn.Module):
    def __init__(self, num_features, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, 1)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# --- 2. Wrapper for sklearn-like use ---
class TabularGNNRegressor:
    def __init__(self, epochs=300, lr=1e-2):
        self.epochs = epochs
        self.lr = lr
        self.scaler = StandardScaler()
        self.model = None

    @staticmethod
    def make_edge_index(n):
        idx = torch.arange(n)
        row, col = torch.meshgrid(idx, idx, indexing="ij")
        mask = row != col
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        return edge_index

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).view(-1, 1)
        n = X.shape[0]
        edge_index = self.make_edge_index(n)
        self.model = SimpleTabularGNN(X.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(X, edge_index).squeeze()
            loss = F.mse_loss(out, y.squeeze())
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float)
        n = X.shape[0]
        edge_index = self.make_edge_index(n)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X, edge_index).cpu().numpy()
        return np.atleast_1d(preds.squeeze())

# ------------------------------

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["OMP_NUM_THREADS"] = "48"

# --- Create results directory if it doesn't exist ---
results_dir = "results"
os.makedirs(results_dir, exist_ok=True) # Added line

# === Load and preprocess data ===
df = pd.read_csv("elastic_constants_vs_pressure_extended.dat", comment="#", sep=r"\s+")

df.columns = [
    "System",        # Sys
    "At_Num",        # At_Num
    "Pressure",      # P(GPa)
    "Lat_Const",     # Lat_Const(Å)
    "Vol",           # Vol(Å^3)
    "Bond_Len",      # Bond_L(A)
    "RWIGS",         # RWIGS(Å)
    "Energy",        # Energy(eV)
    "Enthalpy",      # Enthalpy(eV)
    "E_fermi",       # E-fermi(eV)
    "C11",           # C11(GPa)
    "C12",           # C12(GPa)
    "C44",           # C44(GPa)
    "Bulk_Modulus",  # B_Mod(GPa)
    "Young",         # Young(GPa)
    "Shear",         # Shear(GPa)
    "Poisson",       # Poisson
    "Val_e_tot",     # Val_e_tot
    "Val_e_s",       # Val_e_s
    "Val_e_p",       # Val_e_p
    "Val_e_d",       # Val_e_d
    "Val_e_f",       # Val_e_f
    "DelRho_AvgAtom",# DelRho_AvgAtom
    "DelRho_Tetra",  # DelRho_Tetra
    "DelRho_Octa",   # DelRho_Octa
    "Bader_Ch",      # Bader_Ch
    "Octa_ELF",      # Octa_ELF
    "Tetra_ELF",     # Tetra_ELF
    "Avg_ELF"        # Avg_ELF
]

df["Avg_At_Num"] = df["At_Num"].apply(lambda x: np.mean([int(i) for i in str(x).split(",")]))
df["Inv_Vol"] = 1.0 / df["Vol"]

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.drop(columns=["System", "At_Num"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix among Features", fontsize=18)
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Fig_feature_correlation_matrix_fixed.png"), dpi=300) # Modified line
plt.close()

# === Define models and targets ===
targets = [
    "Lat_Const", "Bulk_Modulus", "Young", "Shear", "Poisson", "Enthalpy", "Octa_ELF", "Tetra_ELF",
    "DelRho_AvgAtom", "DelRho_Tetra", "DelRho_Octa"
]

param_grids = {
    "Ridge": {"model__alpha": [0.01, 0.1, 1, 10]},
    "Lasso": {"model__alpha": [0.1, 1, 10]},
    "SVR": {"model__C": [1, 10], "model__epsilon": [0.01, 0.1]},
    "RandomForest": {"n_estimators": [100]},
    "GradientBoosting": {"n_estimators": [100]},
    "Poly2_LinReg": {"polynomialfeatures__degree": [2]}
}

base_models = {
    "Linear": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
    "Lasso": Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=50000))]),
    "BayesianRidge": Pipeline([("scaler", StandardScaler()), ("model", BayesianRidge())]),
    "SVR": Pipeline([("scaler", StandardScaler()), ("model", SVR())]),
    "RandomForest": RandomForestRegressor(n_jobs=-1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "Poly2_LinReg": Pipeline([("scaler", StandardScaler()), ("polynomialfeatures", PolynomialFeatures(degree=2)), ("model", LinearRegression())]),
    "MLPReg": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42),
    "GPR": Pipeline([("scaler", StandardScaler()), ("model", GaussianProcessRegressor(kernel=C(1.0) * RBF(), normalize_y=True))]),
    "TabularGNN": TabularGNNRegressor(epochs=300, lr=1e-2),
}

predictions_summary = []
r2_scores = {}
selected_features_dict = {}

model_preference = [
    "Linear",
    "Ridge",
    "Lasso",
    "BayesianRidge",
    "Poly2_LinReg",
    "SVR",
    "RandomForest",
    "GradientBoosting",
    "MLPReg",
    "TabularGNN",
    "GPR"
]

# === Main prediction loop ===
for target in targets:
    y = df[target].values
    numeric_df = df.select_dtypes(include=[np.number])
    corrs = numeric_df.corr()[target].drop(target)
    selected_corr_features = corrs[abs(corrs) > 0.5].index.tolist()
    if "Pressure" not in selected_corr_features:
        selected_corr_features.append("Pressure")

    if not selected_corr_features:
        print(f"Skipping {target}: No features with abs corr > 0.5.")
        continue

    X_all_corr = df[selected_corr_features]
    selected_features = selected_corr_features

    selected_features_dict[target] = {
        "correlated": {feat: round(corrs[feat], 4) for feat in selected_corr_features},
        "used": selected_features
    }

    X = df[selected_features].values

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, random_state=42, test_size=0.1111)

    # --- Robust model selection block ---
    best_r2 = -np.inf
    best_mae = np.inf
    best_mse = np.inf
    best_model = None
    best_model_name = ""
    best_simplicity = len(model_preference)
    model_r2_scores = {}

    for name, model in base_models.items():
        try:
            if name == "TabularGNN":
                model.fit(X_train, y_train)
                yval_pred = model.predict(X_val)
            else:
                if name in param_grids:
                    grid = GridSearchCV(model, param_grids[name], cv=3, scoring='r2', n_jobs=-1)
                    grid.fit(X_train, y_train)
                    model = grid.best_estimator_
                else:
                    model.fit(X_train, y_train)
                yval_pred = model.predict(X_val)
            r2 = r2_score(y_val, yval_pred)
            mae = mean_absolute_error(y_val, yval_pred)
            mse = mean_squared_error(y_val, yval_pred)
        except Exception as e:
            print(f"Model {name} failed: {e}")
            r2, mae, mse = -np.inf, np.inf, np.inf
        model_r2_scores[name] = r2

        # Simplicity: lower index is preferred
        simplicity = model_preference.index(name) if name in model_preference else len(model_preference)

        is_better = False
        if r2 > best_r2:
            is_better = True
        elif r2 == best_r2:
            if mae < best_mae:
                is_better = True
            elif mae == best_mae:
                if mse < best_mse:
                    is_better = True
                elif mse == best_mse:
                    if simplicity < best_simplicity:
                        is_better = True

        if is_better:
            best_r2 = r2
            best_mae = mae
            best_mse = mse
            best_model = model
            best_model_name = name
            best_simplicity = simplicity

    y_pred_all = best_model.predict(X)
    mae_score = mean_absolute_error(y, y_pred_all)
    r2_scores[target] = model_r2_scores

    for idx, row in df.iterrows():
        X_pred = row[selected_features].values.reshape(1, -1)
        pred = best_model.predict(X_pred)[0]
        predictions_summary.append((target, row["System"], row["Pressure"], pred, row[target], best_model_name, best_r2, mae_score))

# === Save improved .dat format ===
with open(os.path.join(results_dir, "selected_features_per_target.dat"), "w") as f: # Modified line
    f.write("# Target : Correlated_Features (|r|>0.5) | RFECV_Selected_Features Used in Model\n")
    for target, feats in selected_features_dict.items():
        f.write(f"{target} :\n")
        f.write("  Corr : " + "  ".join([f"{k}({v:+.3f})" for k, v in feats["correlated"].items()]) + "\n")
        f.write("  Used : " + "  ".join(feats["used"]) + "\n\n")

# === Extend predictions beyond known pressure ===
def fit_pressure_models(df, feature_list):
    models = {}
    for feature in feature_list:
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(df[["Pressure"]])
        model = LinearRegression().fit(X_poly, df[feature])
        models[feature] = (model, poly)
    return models

feature_models_by_system = {}
features_to_estimate = ["Vol", "Lat_Const", "C11", "C12", "RWIGS", "Young", "Shear", "Poisson", "E_fermi"]

for system in df["System"].unique():
    df_sys = df[df["System"] == system]
    models = fit_pressure_models(df_sys, features_to_estimate)
    feature_models_by_system[system] = models
for target in targets:
    if target not in selected_features_dict:
        continue

    features = selected_features_dict[target]["used"]
    model_name = max(r2_scores[target], key=r2_scores[target].get)
    best_model = base_models[model_name]

    if model_name in param_grids and model_name != "TabularGNN":
        grid = GridSearchCV(best_model, param_grids[model_name], cv=3, scoring='r2', n_jobs=-1)
        grid.fit(df[features], df[target])
        best_model = grid.best_estimator_
    else:
        best_model.fit(df[features], df[target])

    for system in df["System"].unique():
        df_sys = df[df["System"] == system]
        max_p = df_sys["Pressure"].max()
        base_row = df_sys[df_sys["Pressure"] == max_p].iloc[0]
        
        for extra_p in range(int(max_p + 20), int(max_p + 101), 20):
            new_row = base_row.copy()
            new_row["Pressure"] = extra_p
            models = feature_models_by_system[system]
            for feat in features_to_estimate:
                if feat == target:
                    continue  # Do not overwrite the feature we are trying to predict
                model, poly = models[feat]
                X_poly_ex = poly.transform([[extra_p]])
                new_row[feat] = model.predict(X_poly_ex)[0]
            new_row["Inv_Vol"] = 1.0 / new_row["Vol"]  # Recompute derived features
            df_extended = pd.DataFrame([new_row])
            X_ex = df_extended[features].values
            pred = best_model.predict(X_ex)[0]
            predictions_summary.append((target, system, extra_p, pred, np.nan, model_name, r2_scores[target][model_name], np.nan))

# === Save predictions summary ===
pred_df = pd.DataFrame(predictions_summary, columns=[
    "Target", "System", "Pressure", "Predicted_Value", "Actual_Value", "Model", "R2_Score", "MAE"
])

# === Plot R2 heatmap ===
r2_df = pd.DataFrame(r2_scores).T
plt.figure(figsize=(14, 8))
sns.heatmap(r2_df, annot=True, cmap="viridis", fmt=".2f", cbar_kws={"shrink": 0.8})
plt.xlabel("Models", fontsize=16)
plt.ylabel("Target", fontsize=16)
plt.title("R² Scores (All Models)", fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Fig_R2_model_heatmap_fixed.png"), dpi=300) # Modified line
plt.close()

# === Generate plots per target ===
for target in pred_df["Target"].unique():
    df_target = pred_df[pred_df["Target"] == target]
    model_name = df_target["Model"].iloc[0]
    r2 = df_target["R2_Score"].iloc[0]
    mae = df_target["MAE"].iloc[0]

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x="Actual_Value", y="Predicted_Value", hue="System", data=df_target.dropna(), s=80, edgecolor='black')
    min_val = min(df_target["Actual_Value"].min(), df_target["Predicted_Value"].min())
    max_val = max(df_target["Actual_Value"].max(), df_target["Predicted_Value"].max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='black', linewidth=1.5)
    plt.xlabel(f"Actual {target}", fontsize=16)
    plt.ylabel(f"Predicted {target}", fontsize=16)
    plt.title(f"{target}: Actual vs Predicted\nModel: {model_name}, R2: {r2:.3f}, MAE: {mae:.3f}", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"Fig_{target}_actual_vs_predicted_fixed.png"), dpi=300) # Modified line
    plt.close()

    plt.figure(figsize=(10, 7))
    for system in df_target["System"].unique():
        sub_df = df_target[df_target["System"] == system].sort_values("Pressure")
        split_pressure = df[df["System"] == system]["Pressure"].max()
        plt.plot(sub_df["Pressure"], sub_df["Predicted_Value"], marker='o', label=system)
        plt.axvline(x=split_pressure, color='red', linestyle='--', linewidth=1.5,
                     label='Prediction beyond data' if system == df_target["System"].unique()[0] else None)
    plt.xlabel("Pressure (GPa)", fontsize=16)
    plt.ylabel(f"Predicted {target}", fontsize=16)
    plt.title(f"Predicted {target} vs Pressure\nModel: {model_name}, R2: {r2:.3f}, MAE: {mae:.3f}", fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(results_dir, f"Fig_predicted_{target}_vs_pressure_fixed.png"), dpi=300) # Modified line
    plt.close()

# === PRESSURE-WISE PARITY PLOTS AND R2/MAE TABLE ===
pressurewise_metrics = []

for target in pred_df["Target"].unique():
    df_target = pred_df[(pred_df["Target"] == target) & (pred_df["Actual_Value"].notna())]
    pressures = sorted(df_target["Pressure"].unique())
    for pressure in pressures:
        df_press = df_target[df_target["Pressure"] == pressure]
        if len(df_press) < 2:
            continue  # Need at least 2 points for R2/MAE

        x = df_press["Actual_Value"].values
        y = df_press["Predicted_Value"].values
        systems = df_press["System"].values
        r2 = r2_score(x, y)
        mae = mean_absolute_error(x, y)
        model_name = df_target["Model"].iloc[0]
        pressurewise_metrics.append({
            "Target": target,
            "Pressure": pressure,
            "R2": r2,
            "MAE": mae,
            "N_systems": len(systems),
            "Model": model_name
        })

        plt.figure(figsize=(7, 6))
        for xi, yi, sys in zip(x, y, systems):
            plt.scatter(xi, yi, label=sys, s=80)
            plt.text(xi, yi, sys, fontsize=9, ha='left', va='bottom')
        minmax = [min(x.min(), y.min()), max(x.max(), y.max())]
        plt.plot(minmax, minmax, 'k--', lw=1.5, label='Ideal')
        plt.xlabel(f"Actual {target}")
        plt.ylabel(f"Predicted {target}")
        plt.title(f"All systems at {pressure:.2f} GPa\nTarget: {target}, Model: {model_name}")
        plt.grid(True)
        plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.text(0.05, 0.95, f"R²={r2:.3f}\nMAE={mae:.3f}",
                 ha='left', va='top', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"Fig_{target}_parity_at_{pressure:.2f}GPa.png"), dpi=300) # Modified line
        plt.close()

# Save pressurewise R²/MAE table
pressurewise_metrics_df = pd.DataFrame(pressurewise_metrics)
pressurewise_metrics_df.to_csv(os.path.join(results_dir, "pressurewise_r2_mae.csv"), index=False) # Modified line

print("All ML modeling, prediction, and plotting complete!")

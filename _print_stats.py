import pandas as pd

df = pd.read_csv("results/results.csv")
df["recovery_rate"] = df["recovery_rate"].fillna(0.0)
df["detection_f1"] = df["detection_f1"].fillna(0.0)

methods = ["no_clean","rule_based","agent_full","agent_no_reviewer","agent_no_planner","agent_zero_shot","agent_few_shot"]
print("=== Method averages (9 datasets) ===")
print(f"{'Method':20s} {'AUC':>8s} {'RR':>8s} {'F1':>8s}")
print("-"*48)
for m in methods:
    sub = df[df["method"]==m]
    au = sub["auc"].mean()
    rr = sub["recovery_rate"].mean()
    f1 = sub["detection_f1"].mean()
    print(f"{m:20s} {au:8.4f} {rr:8.4f} {f1:8.4f}")

print()
print("=== CleanML datasets: gap ≤ 0.01 (not useful) ===")
cleanml = df[df["dataset_id"].isin(["Credit","EEG","Marketing"])]
for ds in ["Credit","EEG","Marketing"]:
    sub = df[df["dataset_id"]==ds]
    cu = sub[sub["method"]=="clean_upper"]["auc"].values[0]
    nc = sub[sub["method"]=="no_clean"]["auc"].values[0]
    print(f"  {ds:15s} clean_upper={cu:.4f} no_clean={nc:.4f} gap={cu-nc:.4f}")

print()
print("=== Real_base datasets: meaningful gap! ===")
for ds in ["breast_cancer","digits","wine","pima_diabetes"]:
    sub = df[df["dataset_id"]==ds]
    cu = sub[sub["method"]=="clean_upper"]["auc"].values[0]
    nc = sub[sub["method"]=="no_clean"]["auc"].values[0]
    agents = sub[sub["method"].str.startswith("agent")]
    best = agents.loc[agents["recovery_rate"].idxmax()]
    print(f"  {ds:15s} clean={cu:.4f} dirty={nc:.4f} gap={cu-nc:.4f}  best={best['method']:20s} RR={best['recovery_rate']:.4f} F1={best['detection_f1']:.4f}")

print()
print("=== Best agent ablation per dataset ===")
for ds in df["dataset_id"].unique():
    sub = df[(df["dataset_id"]==ds) & (df["method"].str.startswith("agent"))]
    best = sub.loc[sub["recovery_rate"].idxmax()]
    print(f"  {ds:20s} {best['method']:20s} RR={best['recovery_rate']:.4f}")

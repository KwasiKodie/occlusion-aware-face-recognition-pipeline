import pandas as pd, numpy as np, matplotlib.pyplot as plt

df = pd.read_csv(r"E:\publication_test\utilities_2\log_pair_demo\runtime_matches.csv")
df["mask_bin"] = pd.cut(df["mask_prob"], bins=[0,0.25,0.75,1.0],
                        labels=["<=low","try-both",">=high"], include_lowest=True)

summary = (df.groupby(["mask_bin","mask_route"])["decision"]
             .mean().reset_index())

fig, ax = plt.subplots(figsize=(6,4), dpi=150)
for route, sub in summary.groupby("mask_route"):
    ax.bar(sub["mask_bin"], sub["decision"], width=0.25, label=route)
    ax.set_xlabel("Mask probability region")
    ax.set_ylabel("Acceptance rate")
    ax.set_ylim(0,1)
    ax.legend(title="Route")
    ax.set_title("Routing vs Acceptance — Misrouting Diagnostic")
    fig.tight_layout()
    fig.savefig("misrouting_diagnostic.png")

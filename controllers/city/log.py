import pandas as pd
import matplotlib.pyplot as plt

# plt.plot(df_eval["episode"], df_eval["total_reward"], label="Total Reward")
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.title("Learning Curve")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("learning_curve.png")
# plt.show()

# R_structured = df_structured["total_reward"].mean()
# R_unstructured = df_unstructured["total_reward"].mean()
# generalization_gap = R_structured - R_unstructured

# print(f"Generalization Gap ΔR = {generalization_gap:.2f}")


import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# -------- SETTINGS -------- #
log_folder = "logs"
N_eval = 2
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)
print("1")

# -------- COLLECT FILES -------- #
csv_files = glob.glob(os.path.join(log_folder, "*.csv"))

# Group CSVs by shared model ID (ignore structured/unstructured)
model_groups = {}
for file in csv_files:
    filename = os.path.basename(file)
    model_id = filename.split("__")[0]  # before __S or __U
    if model_id not in model_groups:
        model_groups[model_id] = []
    model_groups[model_id].append(file)
    print("2")

# -------- PROCESS AND PLOT -------- #
for model_id, files in model_groups.items():
    structured_file = [f for f in files if "__S_" in f]
    unstructured_file = [f for f in files if "__U_" in f]

    if not structured_file or not unstructured_file:
        print(f"Skipping {model_id} (missing files)")
        continue

    df_struct = pd.read_csv(structured_file[0])
    df_unstruct = pd.read_csv(unstructured_file[0])

    # --- Plot Learning Curves ---
    plt.figure()
    plt.plot(df_struct["episode"], df_struct["total_reward"], label="Structured", marker="o")
    plt.plot(df_unstruct["episode"], df_unstruct["total_reward"], label="Unstructured", marker="s")
    plt.title(f"Learning Curve: {model_id}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{model_id}_learning_curve.png"))
    plt.close()

    # --- Compute Generalization Gap ---
    R_structured = df_struct["total_reward"].mean()
    R_unstructured = df_unstruct["total_reward"].mean()
    delta_R = R_structured - R_unstructured

    print(f"[{model_id}] ΔR (Generalization Gap): {delta_R:.2f}")

print("\n✅ All plots saved to:", output_folder)
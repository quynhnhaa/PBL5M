# plot_weights.py
import csv
import matplotlib.pyplot as plt

weights_data = []
with open("train_eval_metrics/widerface_run/weights.csv", mode='r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        weights_data.append({
            "Epoch": int(row["Epoch"]),
            "Stem_Mean": float(row["Stem_Mean"]),
            "Stem_Std": float(row["Stem_Std"]),
            "Stage3_Mean": float(row["Stage3_Mean"]),
            "Stage3_Std": float(row["Stage3_Std"]),
            "Stage5_Mean": float(row["Stage5_Mean"]),
            "Stage5_Std": float(row["Stage5_Std"]),
            "Neck_P5_to_P4_Mean": float(row["Neck_P5_to_P4_Mean"]),
            "Neck_P5_to_P4_Std": float(row["Neck_P5_to_P4_Std"]),
            "Head_P3_Mean": float(row["Head_P3_Mean"]),
            "Head_P3_Std": float(row["Head_P3_Std"])
        })

epochs = [d["Epoch"] for d in weights_data]

# Vẽ biểu đồ Mean
plt.figure(figsize=(12, 6))
plt.plot(epochs, [d["Stem_Mean"] for d in weights_data], label="Stem Mean")
plt.plot(epochs, [d["Stage3_Mean"] for d in weights_data], label="Stage3 Mean")
plt.plot(epochs, [d["Stage5_Mean"] for d in weights_data], label="Stage5 Mean")
plt.plot(epochs, [d["Neck_P5_to_P4_Mean"] for d in weights_data], label="Neck P5_to_P4 Mean")
plt.plot(epochs, [d["Head_P3_Mean"] for d in weights_data], label="Head P3 Mean")
plt.xlabel("Epoch")
plt.ylabel("Mean Weight Magnitude")
plt.title("Mean Weight Magnitude Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("weights_mean_plot.png")
plt.close()

# Vẽ biểu đồ Std
plt.figure(figsize=(12, 6))
plt.plot(epochs, [d["Stem_Std"] for d in weights_data], label="Stem Std")
plt.plot(epochs, [d["Stage3_Std"] for d in weights_data], label="Stage3 Std")
plt.plot(epochs, [d["Stage5_Std"] for d in weights_data], label="Stage5 Std")
plt.plot(epochs, [d["Neck_P5_to_P4_Std"] for d in weights_data], label="Neck P5_to_P4 Std")
plt.plot(epochs, [d["Head_P3_Std"] for d in weights_data], label="Head P3 Std")
plt.xlabel("Epoch")
plt.ylabel("Weight Standard Deviation")
plt.title("Weight Standard Deviation Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("weights_std_plot.png")
plt.close()

print("Biểu đồ đã được lưu tại: weights_mean_plot.png và weights_std_plot.png")
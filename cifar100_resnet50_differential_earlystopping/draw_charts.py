import os
import json
import matplotlib.pyplot as plt

result_path = os.path.join("experiment_results", "models")

output_path = os.path.join("experiment_results", "charts")

if not os.path.exists(output_path):
    os.makedirs(output_path)

top_result_path = os.path.join(result_path, "topology_stopped")
val_result_path = os.path.join(result_path, "val_stopped")

top_models = set(os.listdir(top_result_path))
val_models = set(os.listdir(val_result_path))

for model in top_models & val_models:
    top_data_path = os.path.join(top_result_path, model, f"{model}_topology_data.json")
    
    with open(top_data_path, "r") as top_f:
        top_d = json.load(top_f)

    epochs = sorted([int(x) for x in top_d.keys()])
    top_vals = [top_d[str(v)] for v in epochs]
    top_change = [abs(top_vals[i]-top_vals[i-1]) for i in epochs[1:]]

    output_file = os.path.join(output_path, f"{model}_topology.png")

    plt.plot(epochs, top_vals, label="Mean number of homologies")
    plt.plot(epochs[1:], top_change, label="Absolute rate of change of mean number of topologies")
    plt.title("Topological overview - stage 4, homology H1")
    plt.legend()
    plt.savefig(output_file)
    plt.close()




    


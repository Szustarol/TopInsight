import pickle
import numpy as np
import ripserplusplus as rpp_py
# from torch_topological.nn import VietorisRipsComplex

# def evaluate_h1_count(activations):
#     rips_layer = VietorisRipsComplex(dim=1, p=2)
#     activations = activations.reshape((activations.size(0), -1))
#     persistence_information = rips_layer(activations)
#     dim_1 = persistence_information[1].diagram
#     return len(dim_1)

def evaluate_h1_count(activations):
    activations = activations.reshape((activations.size(0), -1)).numpy()
    rpp_result = rpp_py.run("--format point-cloud --dim 1", activations)
    return len(
        np.array([[h[0], h[1]] for h in rpp_result[1]])
    )


for idx in range(0, 45):
    with open(f"activations_{idx}.pickle", 'rb') as p_f:
        activations = pickle.load(p_f)
    count = evaluate_h1_count(activations)
    print("IDX:\t", idx, "\tCount:\t", count)
import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    max_accuracy = []
    for i in range(times):
        max_accuracy.append(test_acc[i].max())

    print("std for best accuracy:", np.std(max_accuracy))
    print("mean for best accuracy:", np.mean(max_accuracy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    if not os.path.exists(file_path):
        # Fallback: flexible match — file_name is like "Dataset_Algo_test"
        # but actual files may be "Dataset_Model_Algo_test__accX.xxxx.h5"
        results_dir = os.path.dirname(file_path) or "../results"
        if not os.path.isdir(results_dir):
            results_dir = "../results"
        parts = file_name.split('_')
        # Expect at least: [dataset, algo, goal] or similar
        candidates = [f for f in os.listdir(results_dir)
                      if f.endswith('.h5') and all(p in f for p in parts)]
        if candidates:
            file_path = os.path.join(results_dir, sorted(candidates)[0])

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc
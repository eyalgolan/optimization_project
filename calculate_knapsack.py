import numpy as np

"""
To calculate a result of the knapsack problem where the input is:
* An array of item values
* An array of item sizes matching the values array
* A capacity represented by a number
Run the following to get the algorithm's result:
print(calculate_knapsack(values_array, sizes_array, number))
"""


class Estimator:
    def __init__(self, values, sizes):
        self.value = values
        self.size = sizes
        self.num_of_values = len(values)

    def get_best_relaxed_value(self, index, capacity):
        """
        Performs an estimation of the largest value of the sum of all values that can be put in the bag, according to
        the ratio we previously calculated.
        """
        best_value = 0
        capa = 0
        for i in range(index, self.num_of_values):
            capa += self.size[i]
            best_value += self.value[i]
            if capa >= capacity:
                return int(best_value - self.value[i] * (capa - capacity) / self.size[i])
        best_value = int(best_value)
        return best_value


def build_efficiency_dict(values, sizes, num_values):
    """
    sort value and size by decreasing efficiency.
    """
    efficiency = [(values[index], sizes[index], index, values[index] / sizes[index]) for index in range(num_values)]
    dtype = [('value', float), ('size', float), ('index', int), ('ratio', float)]
    efficiency = np.array(efficiency, dtype=dtype)
    efficiency = np.sort(efficiency, order='ratio')
    efficiency = efficiency[::-1]

    return efficiency


def build_sorted_and_hash_table(num_values, efficiency):
    """
    Build a new list where the i'th item holds the value, size and index that represents it and the additional
    calculated param.
    """
    sorted_values = []
    sorted_size = []
    hash_table = []
    for index in range(num_values):
        sorted_values.append(efficiency[index][0])
        sorted_size.append(efficiency[index][1])
        hash_table.append(efficiency[index][2])
    return sorted_values, sorted_size, hash_table


def cut_invalid_capacity(weight, capacity, value, index, sorted_values, dp_second, estimator, iteration, best_solution,
                         num_values):
    if weight <= capacity:
        new_value = value[0] + sorted_values[index] * iteration
        # merge branches with same weight
        if weight not in dp_second or new_value > dp_second[weight][0]:
            if weight in dp_second:
                estimate = dp_second[weight][1] - dp_second[weight][0] + new_value
            else:
                estimate = estimator.get_best_relaxed_value(index + 1, capacity - weight) + new_value
            # cut bad branches
            if estimate > best_solution[0]:
                dp_second[weight] = (new_value, estimate, value[2] + [iteration])
                if new_value > best_solution[0]:
                    best_solution = (new_value, value[2] + [iteration] + [0] * (num_values - index - 1))
    return best_solution


def go_over_dp(dp, dp_second, sorted_size, sorted_values, capacity, index, estimator, num_values, best_solution):
    for key, value in dp.items():
        for iteration in range(2):
            weight = key + iteration * sorted_size[index]
            best_solution = cut_invalid_capacity(weight, capacity, value, index, sorted_values, dp_second, estimator,
                                                 iteration, best_solution, num_values)
    return best_solution


def take_the_best(num_values, hash_table, best_solution):
    """
    Convert the indexes and return the result of the problem.
    """
    final_items = np.zeros(num_values, 'int')
    for index in range(num_values):
        final_items[hash_table[index]] = best_solution[1][index]
    return final_items


def calculate_knapsack(values, sizes, capacity):
    """
    Compute the optimum knapsack according to the given values, sizes and capacity.
    :param values: list of values
    :param sizes: list of sizes
    :param capacity: the capacity of the knapsack
    :return: tuple (max value, repartition of the items)
    """
    num_values = len(values)
    efficiency = build_efficiency_dict(values, sizes, num_values)
    sorted_values, sorted_size, hash_table = build_sorted_and_hash_table(num_values, efficiency)

    # initialize an estimator in order to avoid cloning the list
    estimator = Estimator(sorted_values, sorted_size)
    dp = {0: (0, estimator.get_best_relaxed_value(0, capacity), [])}
    best_solution = (0, [0] * num_values)
    for index in range(num_values):
        dp_second = {}
        best_solution = go_over_dp(dp, dp_second, sorted_size, sorted_values, capacity, index, estimator, num_values,
                                   best_solution)
        dp = dp_second

    # take the best
    final_items = take_the_best(num_values, hash_table, best_solution)

    # retrieve the data
    return best_solution[0], final_items


print(calculate_knapsack([12, 10, 7], [10, 7, 6], 12.5))
print(calculate_knapsack([12, 10, 7], [10, 7, 6], 14))

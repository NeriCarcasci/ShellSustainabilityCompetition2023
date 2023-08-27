import pandas as pd
import numpy as np

# Read data
bio_data = pd.read_csv("Biomass_History_with_ARIMA_Forecasts.csv")
dist_matrix = pd.read_csv("Distance_Matrix.csv").values[1:, 1:]
biomass_2019 = bio_data['2019'].values

# Parameters
num_harvesting_sites = dist_matrix.shape[0]
num_depots = dist_matrix.shape[1]
P = 25


def greedy_p_median(dist_matrix, P):
    dist_matrix = dist_matrix.copy()
    num_harvesting_sites, num_depots = dist_matrix.shape
    chosen_depots = []
    unchosen_depots = list(range(num_depots))

    for _ in range(P):
        best_depot = None
        best_total_distance = float('inf')

        for depot in unchosen_depots:
            # Calculate total distance if this depot was chosen
            total_distance = np.sum(np.min(dist_matrix[:, [depot] + chosen_depots], axis=1))
            if total_distance < best_total_distance:
                best_total_distance = total_distance
                best_depot = depot

        chosen_depots.append(best_depot)
        unchosen_depots.remove(best_depot)

        # Update distance matrix to set distances to chosen depots to a large value
        dist_matrix[:, best_depot] = 1e9
        # so they aren't selected again as closest depots
        #dist_matrix[:, best_depot] = float('inf')

    return chosen_depots


selected_depots = greedy_p_median(dist_matrix, P)
print("Selected depots indices:", selected_depots)


#print(dist_matrix[np.ix_(selected_depots, selected_depots)])
#print("Selected depots:", selected_depots)

def depot_to_depot_distance(original_dist_matrix, selected_depots):
    """Create a distance matrix between selected depots."""
    return original_dist_matrix[np.ix_(selected_depots, selected_depots)]


def greedy_p_median_for_subset(dist_matrix, P, subset):
    """Select facilities based on a given subset of locations."""
    dist_matrix = dist_matrix.copy()
    num_locations = len(subset)
    chosen_facilities = []
    unchosen_facilities = list(range(num_locations))

    for iteration in range(P):
        best_facility = None
        best_total_distance = float('inf')

        for facility in unchosen_facilities:
            # Calculate total distance if this facility was chosen
            total_distance = np.sum(np.min(dist_matrix[:, [facility] + chosen_facilities], axis=1))
            if total_distance < best_total_distance:
                best_total_distance = total_distance
                best_facility = facility

        if best_facility is not None:
            chosen_facilities.append(best_facility)
            unchosen_facilities.remove(best_facility)

            # Update distance matrix to set distances to chosen facilities to a large value
            dist_matrix[:, best_facility] = 1e9
            #dist_matrix[:, best_facility] = float('inf')
        else:
            print(f"Failed at iteration {iteration + 1}")
            print("Remaining unchosen facilities:", unchosen_facilities)
            print("Current chosen facilities:", chosen_facilities)
            print("Current distance matrix:\n", dist_matrix)
            raise ValueError("Could not find a best facility. Check the logic or the data.")

    # Convert the chosen facility indices to the actual indices in the original matrix
    chosen_facilities = [subset[index] for index in chosen_facilities]
    return chosen_facilities


# Get the depot-to-depot distance matrix
depot_dist_matrix = depot_to_depot_distance(dist_matrix, selected_depots)

# Use the greedy p-median approach to select 5 refineries from the 25 depots
selected_refineries = greedy_p_median_for_subset(depot_dist_matrix, 5, selected_depots)
print("Selected refineries indices:", selected_refineries)




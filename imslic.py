import numpy as np
import cv2
from multiprocessing import Pool
from functools import partial
import scipy.sparse as sparse
import itertools
import time
import gc

def recenter(k, labels, lab_image, region_size):
    height, width = labels.shape[0], labels.shape[1]

    dx = [1, 0]
    dy = [0, 1]

    indices = np.array([(np.ravel_multi_index((y, x), (height, width)), x, y) for y in range(height) for x in range(width) if labels[y, x] == k])
    _, xmax, ymax = indices.max(axis=0)
    _, xmin, ymin = indices.min(axis=0)
    delta_x = xmax - xmin + 1
    delta_y = ymax - ymin + 1

    assert indices.shape[0] > 0

    #sub_image = lab_image[ymin: ymax + 1, xmin: xmax + 1, :]
    #colors = np.sum(sub_image, axis=2)
    #min_color = colors.min()
    #max_color = colors.max()
    #color_factor = (max_color - min_color) / region_size
    

    subgraph = sparse.lil_matrix((delta_x * delta_y, delta_x * delta_y))
    to_save = []
    all_of_them = []

    for y in range(delta_y):
        for x in range(delta_x):
            linear_index = np.ravel_multi_index((y, x), (delta_y, delta_x))
            all_of_them.append(linear_index)
            if labels[ymin + y, xmin + x] == k:
                to_save.append(linear_index)
            for i in range(len(dx)):
                nx, ny = x + dx[i], y + dy[i]
                if 0 <= nx < delta_x and 0 <= ny < delta_y:
                    neighbor_linear_index = np.ravel_multi_index((ny, nx), (delta_y, delta_x))
                    subgraph[linear_index, neighbor_linear_index] = np.sqrt(np.sum(np.power(np.hstack((dx[i], dy[i], (lab_image[y + ymin, x + xmin] - lab_image[ny + ymin, nx + xmin]))), 2.0)))
                    subgraph[neighbor_linear_index, linear_index] = subgraph[linear_index, neighbor_linear_index]
    
    subgraph = subgraph.tocsr()
    distances = sparse.csgraph.dijkstra(subgraph, directed=False)

    assert distances.max() != np.inf

    to_delete = set(all_of_them) - set(to_save)
    distances = np.delete(distances, list(to_delete), axis=0)
    distances = np.delete(distances, list(to_delete), axis=1)
    marginal = np.sum(distances, axis=1)
    seed_index = marginal.argmin()
    seed_y, seed_x = np.unravel_index(to_save[seed_index], (delta_y, delta_x))
    seed_y += ymin
    seed_x += xmin
    return (k, seed_x, seed_y)

def compute_region_distances2(k, seeds_x, seeds_y, xi, region_size, areas, lab_image):
    seed_x, seed_y = seeds_x[k], seeds_y[k]
    height, width = lab_image.shape[0], lab_image.shape[1]
    lambda_factor = get_lambda(seed_x, seed_y, xi, region_size, areas, width, height)
    offset = region_size * lambda_factor
    xmin = np.max([0, seed_x - offset]).astype("int")
    xmax = np.min([width - 1, seed_x + offset]).astype("int")
    ymin = np.max([0, seed_y - offset]).astype("int")
    ymax = np.min([height - 1, seed_y + offset]).astype("int")
    delta_x = xmax - xmin + 1
    delta_y = ymax - ymin + 1
    region_graph = sparse.lil_matrix((delta_y * delta_x, delta_x * delta_y))
    dx = [1, 0]
    dy = [0, 1]
    sub_image = lab_image[ymin: ymax + 1, xmin: xmax + 1, :]
    seed_index = np.ravel_multi_index((seed_y - ymin, seed_x - xmin), (delta_y, delta_x))
    #colors = np.sum(sub_image, axis=2)
    #min_color = colors.min()
    #max_color = colors.max()
    #color_factor = (max_color - min_color) / region_size
    for y in range(delta_y):
        for x in range(delta_x):
            index = np.ravel_multi_index((y, x), (delta_y, delta_x))
            for i in range(len(dx)):
                if 0 <= x + dx[i] < delta_x and 0 <= y + dy[i] < delta_y:
                    neighbor_index = np.ravel_multi_index((y + dy[i], x + dx[i]), (delta_y, delta_x))
                    distance =  np.sqrt(
                                    dx[i] * dx[i] +
                                    dy[i] * dy[i] +
                                    np.sum(np.power((sub_image[y, x, :] - sub_image[y + dy[i], x + dx[i], :]), 2.0))
                                )
                    region_graph[index, neighbor_index] = distance
                    region_graph[neighbor_index, index] = distance
    distances = sparse.csgraph.shortest_path(region_graph.tocsr(), method='D', directed=False, indices=(seed_index))
    return (k, xmin, ymin, delta_x, delta_y, distances)

def get_lambda(seed_x, seed_y, xi, region_size, areas, width, height):
    xmin = np.maximum(0, seed_x - region_size).astype("int")
    xmax = np.minimum(width - 1, seed_x + region_size).astype("int")
    ymin = np.maximum(0, seed_y - region_size).astype("int")
    ymax = np.minimum(height - 1, seed_y + region_size).astype("int")
    sub_region_area = 0.0
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            sub_region_area += areas[y, x]
    return np.sqrt(xi / sub_region_area)

def performMSLIC2(lab_image, K, seeds, region_size, labels, xi, areas, max_iterations, pool):
    height, width = lab_image.shape[0], lab_image.shape[1]
    max_float = np.finfo("f").max
    dx = [0, 1, 0, -1]
    dy = [-1, 0, 1, 0]

    for iteration in range(max_iterations):
        start_time = time.time()
        print("Iteration {}.".format(iteration + 1))

        global_distances = np.ones((height, width)) * max_float
        labels = np.ones((height, width)) * -1.0
        
        seeds_x, seeds_y = seeds[0, :].astype("int"), seeds[1, :].astype("int")

        for k in range(K):
            x, y = seeds_x[k], seeds_y[k]
            labels[y, x] = k
            global_distances[y, x] = -1.0

        parallel_function = partial(compute_region_distances2, seeds_x=seeds_x, seeds_y=seeds_y, xi=xi, region_size=region_size, areas=areas, lab_image=lab_image)
        parallel_results = pool.map(parallel_function, range(K))

        for k, xmin, ymin, delta_x, delta_y, distances in parallel_results:
            assert distances.max() != np.inf
            for i in range(distances.shape[0]):
                y, x = np.unravel_index(i, (delta_y, delta_x))
                if distances[i] < global_distances[ymin + y, xmin + x]:
                    global_distances[ymin + y, xmin + x] = distances[i]
                    labels[ymin + y, xmin + x] = k
        
        for k in range(K):
            x, y = seeds_x[k], seeds_y[k]
            assert labels[y, x] == k
        
        parallel_results = None
        gc.collect()

        print("Computed shortest distances and assigned labels.")

        orphanes_indices = [(np.ravel_multi_index((y, x), (height, width)), x, y) for y in range(height) for x in range(width) if labels[y, x] == -1]
        orphanes_indices_to_pop = []

        print("There are {} orphanes.".format(len(orphanes_indices)))

        for o in range(len(orphanes_indices)):
            index, x, y = orphanes_indices[o]
            global_distance = max_float
            for i in range(len(dx)):
                nx, ny = x + dx[i], y + dy[i]
                if 0 <= nx < width and 0 <= ny < height and labels[ny, nx] != -1:
                    distance = global_distances[ny, nx] + np.sqrt(np.sum(np.power(np.hstack((dx[i], dy[i], lab_image[y, x, :] - lab_image[ny, nx, :])), 2.0)))
                    if distance < global_distance:
                        global_distance = distance
                        labels[y, x] = labels[ny, nx]
            orphanes_indices_to_pop.append(index)
        
        orphanes_indices = [i for i in orphanes_indices if i[0] not in orphanes_indices_to_pop]

        if len(orphanes_indices) > 0:
            print("There are {} unassiged orphanes.".format(len(orphanes_indices)))

        parallel_results = None
        gc.collect()

        parallel_function = partial(recenter, labels=labels, lab_image=lab_image, region_size=region_size)
        parallel_results = pool.map(parallel_function, range(K))

        for k, seed_x, seed_y in parallel_results:
            seeds[:, k] = np.hstack((seed_x, seed_y, lab_image[seed_y, seed_x, :]))

        parallel_results = None
        gc.collect()

        print("Seeds recentered.")

        print("End iteration {}, time {}.".format(iteration + 1, time.time() - start_time))

    return (labels, seeds)

def compute_seeds2(lab_image, width, height, K, cumulative_areas):
    coordinates = set()
    counter = 0
    seeds = np.zeros((5, K))
    for area in np.linspace(0, cumulative_areas[-1], K):
        for i in range(len(cumulative_areas)):
            if cumulative_areas[i] >= area and np.unravel_index(i, (height, width)) not in coordinates:
                y, x = np.unravel_index(i, (height, width))
                coordinates.add((y, x))
                seeds[:, counter] = np.hstack((x, y, lab_image[y, x, :]))
                counter += 1
                break
    while counter < K:
        x, y = np.random.rand() * (width - 1), np.random.rand() * (height - 1)
        if (y, x) not in coordinates:
            coordinates.add((y, x))
            seeds[:, counter] = np.hstack((x, y, lab_image[y, x, :]))
            counter += 1
    return seeds

def compute_cumulative_areas(areas, width, height):
    cumulative_areas = np.zeros(width * height)
    for y in range(height):
        for x in range(width):
            index = np.ravel_multi_index((y, x), (height, width))
            if index > 0:
                cumulative_areas[index] = cumulative_areas[index - 1] + areas[y, x]
            else:
                cumulative_areas[index] = areas[y, x]
    return cumulative_areas

def get_corner2(lab_image, width, height, x, y, dx, dy):
    counter = 1
    lab = lab_image[y, x, :]
    n = np.array([x, y])
    for i in range(len(dx)):
        if x + dx[i]>= 0 and x + dx[i] < width and y + dy[i] >= 0 and y + dy[i] < height:
            lab += lab_image[y + dy[i], x + dx[i], :]
            counter =+ 1
        n += np.array([dx[i], dy[i]])
    return np.concatenate((n / 4.0, lab / counter))

def get_corner(lab_image, width, height, x, y, dx, dy):
    counter = 1
    l = lab_image[y, x, 0] 
    a = lab_image[y, x, 1]
    b = lab_image[y, x, 2]
    nx = x
    ny = y
    for i in range(len(dx)):
        l += lab_image[y + dy[i], x + dx[i], 0]
        a += lab_image[y + dy[i], x + dx[i], 1]
        b += lab_image[y + dy[i], x + dx[i], 2]
        counter += 1
        nx += x + dx[i]
        ny += y + dy[i]
    return np.array([nx / 4.0, ny / 4.0, l / 4.0, a / 4.0, b / 4.0])

def compute_row_areas(index, lab_image, width, height):
    row_areas = np.zeros(width)
    for x in range(width):
        a1 = get_corner(lab_image, width, height, x, index, [-1, -1, 0], [0, -1, -1])
        a2 = get_corner(lab_image, width, height, x, index, [-1, -1, 0], [0, 1, 1])
        a3 = get_corner(lab_image, width, height, x, index, [0, 1, 1], [1, 1, 0])
        a4 = get_corner(lab_image, width, height, x, index, [0, 1, 1], [-1, -1, 0])
        a21 = a2 - a1
        a23 = a2 - a3
        a43 = a4 - a3
        a41 = a4 - a1
        angle123 = np.sqrt(1 - np.power(np.dot(a21, a23) / (np.linalg.norm(a21) * np.linalg.norm(a23)), 2.0))
        angle341 = np.sqrt(1 - np.power(np.dot(a43, a41) / (np.linalg.norm(a41) * np.linalg.norm(a43)), 2.0))
        delta123 = 0.5 * np.linalg.norm(a21) * np.linalg.norm(a23) * angle123
        delta341 = 0.5 * np.linalg.norm(a43) * np.linalg.norm(a41) * angle341
        row_areas[x] = delta123 + delta341
    return (index, row_areas)

def find_neighbors(x, y, width, height):
    neighbors = []
    if x + 1 < width:
        neighbors.append((x + 1, y))
    if x - 1 >= 0:
        neighbors.append((x - 1, y))
    if y + 1 < height:
        neighbors.append((x, y + 1))
    if y - 1 >= 0:
        neighbors.append((x, y - 1))
    return neighbors

def main():
    pool = Pool()

    region_size = 10.0
    max_iterations = 15

    bgr_image = cv2.imread("./6.jpg")
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab).astype("float32")

    height = lab_image.shape[0]
    width = lab_image.shape[1]

    areas = np.zeros((height, width))
    cumulative_areas = None

    labels = np.ones((height, width)) * -1.0
    K = np.int((width * height) / (np.power(region_size, 2.0)))
    xi = None
    seeds = None

    print("Loaded the image. Width: {} Height: {}".format(width, height))

    for index, row_areas in pool.map(partial(compute_row_areas, lab_image=lab_image, width=width, height=height), range(height)):
        areas[index, :] = row_areas

    print("Computed the area for each pixel.")

    cumulative_areas = np.cumsum(areas)#compute_cumulative_areas(areas, width, height)
    xi = cumulative_areas[-1] * 4.0 / K

    print("Computed the cumulative area for each pixel.")

    seeds = compute_seeds2(lab_image, width, height, K, cumulative_areas)

    print("Computed the seeds and perturbed.")

    labels, seeds = performMSLIC2(lab_image, K, seeds, region_size, labels, xi, areas, max_iterations, pool)

    final_image = lab_image.copy()

    for y in range(height):
        for x in range(width):
            seed = labels[y, x]
            neighbors = find_neighbors(x, y, width, height)
            for nx, ny in neighbors:
                neighbor_seed = labels[ny, nx]
                if neighbor_seed != seed and not np.array_equal(final_image[ny, nx, :], [0, 128, 128]):
                    final_image[y, x, :] = [0, 128, 128]
                    break
    cv2.imwrite("IMSLIC6.jpg",  cv2.cvtColor(final_image.astype("uint8"), cv2.COLOR_Lab2BGR))

    pool.close()

def get_corner_2(lab_image, width, height, x, y, dx, dy):
    counter = 1
    l = lab_image[y, x, 0] 
    a = lab_image[y, x, 1]
    b = lab_image[y, x, 2]
    for i in range(len(dx)):
        if x + dx[i]>= 0 and x + dx[i] < width and y + dy[i] >= 0 and y + dy[i] < height:
            l += lab_image[y + dy[i], x + dx[i], 0]
            a += lab_image[y + dy[i], x + dx[i], 1]
            b += lab_image[y + dy[i], x + dx[i], 2]
            counter += 1
    return np.array([l / counter, a / counter, b / counter])

def test_area():
    p_lab_image = np.load("./my_results/padded_lab_image.npy")
    width = p_lab_image.shape[1] - 2
    height = p_lab_image.shape[0] - 2
    area = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            a1 = get_corner(p_lab_image, width, height, x + 1, y + 1, [-1, -1, 0], [0, -1, -1])
            a2 = get_corner(p_lab_image, width, height, x + 1, y + 1, [-1, -1, 0], [0, 1, 1])
            a3 = get_corner(p_lab_image, width, height, x + 1, y + 1, [0, 1, 1], [1, 1, 0])
            a4 = get_corner(p_lab_image, width, height, x + 1, y + 1, [0, 1, 1], [-1, -1, 0])
            a21 = a2 - a1
            a23 = a2 - a3
            a43 = a4 - a3
            a41 = a4 - a1
            angle123 = np.sqrt(1 - np.power(np.dot(a21, a23) / (np.linalg.norm(a21) * np.linalg.norm(a23)), 2.0))
            angle341 = np.sqrt(1 - np.power(np.dot(a43, a41) / (np.linalg.norm(a41) * np.linalg.norm(a43)), 2.0))
            delta123 = 0.5 * np.linalg.norm(a21) * np.linalg.norm(a23) * angle123
            delta341 = 0.5 * np.linalg.norm(a43) * np.linalg.norm(a41) * angle341
            area[y, x] = delta123 + delta341
    np.save("./cv_results/area.npy", area)

def test_cumulative_area():
    area = np.load("./my_results/area.npy")
    c_area = np.cumsum(area).reshape(area.shape)
    np.save("./cv_results/cumulative_area.npy", c_area)

if __name__ == "__main__":
    #test_area()
    test_cumulative_area()
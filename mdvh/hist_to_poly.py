import matplotlib.pyplot as plt
import pandas as pd
from math import isclose
from numpy import *
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d


def dist2(p1, p2):
    return (p1[0]-p2[0])**2. + (p1[1]-p2[1])**2.

def find_closest(test, points):
    min_dist = float("inf")
    closest_ind = -1
    for (i, point) in enumerate(points):
        dist = dist2(point,test)
        if dist < min_dist:
            min_dist = dist
            closest_ind = i
    return closest_ind

def in_box(test, bounds):
    for coord, bmin in zip(test, bounds[0]):
        if coord < bmin:
            return False
    for coord, bmax in zip(test, bounds[1]):
        if coord > bmax:
            return False
    return True

def point_dir_to_bbox(point, direction, bounds):
    xbound = bounds[int(direction[0] > 0)][0]
    ybound = bounds[int(direction[1] > 0)][1]
    m = direction[0]/direction[1]
    
    xint = m*(ybound-point[1])+point[0]
    yint = (1./m)*(xbound-point[0])+point[1]
    xt = (xbound-point[0])/direction[0]
    yt = (ybound-point[1])/direction[1]
    
    return [xbound, yint] if xt < yt else [xint, ybound]
    

def calc_bbox_intersection(point_pair, vertex, points, bounds):
    dists = [dist2(vertex, point) for point in points]
    assert(isclose(*(dists[p] for p in point_pair)))
    radius = dists[point_pair[0]]
    other_point = -1
    for i, dist in enumerate(dists):
        if isclose(dist, radius) and i not in point_pair:
            other_point = i
    assert(other_point > 0)
    pair_vec = array(points[point_pair[0]])-array(points[point_pair[1]])
    ridge_dir = array([pair_vec[1], -pair_vec[0]])
    trial = point_dir_to_bbox(array(vertex), ridge_dir, bounds)
    if find_closest(trial, points) not in point_pair:
        return point_dir_to_bbox(array(vertex), -1*ridge_dir, bounds)
    return trial

def voronoi_polygons_bbox_2d(vor, bmin, bmax):
    if vor.points.shape[1] != 2:
        raise ValueError("Can only plot 2d regions")
    bounds = [bmin, bmax]
    vertices = vor.vertices.tolist()[:]
    ridge_vertices = vor.ridge_vertices[:]
    ridge_points = vor.ridge_points.tolist()[:]
    regions = vor.regions[:]
    
    # Trim vertices lying out of bounds. These will become new points at infinity, and will have
    # negative labels.
    vertex_labels = []
    trimmed_vertices = []
    curr_label = 0
    num_cut = 0
    for vertex in vertices:
        if in_box(vertex, bounds):
            vertex_labels.append(curr_label)
            trimmed_vertices.append(vertex)
            curr_label += 1
        else:
            num_cut += 1
            vertex_labels.append(-1-num_cut)
    vertices = trimmed_vertices[:]
    num_unaltered_vertices = len(vertices)
    ridge_vertices = [[vertex_labels[v] if v >= 0 else -1 for v in ridge] for ridge in ridge_vertices]
    regions = [[vertex_labels[v] if v >= 0 else -1 for v in region] for region in regions]
        
    # Delete all ridges between points at infinity
    is_inf_ridges = [all([v < 0 for v in ridge]) for ridge in ridge_vertices]
    trimmed_ridge_vertices = []
    trimmed_ridge_points = []
    for i, is_inf in enumerate(is_inf_ridges):
        if not is_inf:
            trimmed_ridge_vertices.append(ridge_vertices[i])
            trimmed_ridge_points.append(ridge_points[i])
    ridge_vertices = trimmed_ridge_vertices[:]
    ridge_points = trimmed_ridge_points[:]
    
    # For each escaping edge, calculate bbox intersection, append to vertices, and replace all
    # instances in regions of that edge with the new vertex
    for e, ridge in enumerate(ridge_vertices):
        if ridge[0] >= 0 and ridge[1] >= 0:
            continue
        escapee = min(ridge)
        partner = max(ridge)
        for r, region in enumerate(regions):
            new_region = region.copy()
            for i, v in enumerate(region):
                before = (i-1)%len(region)
                after = (i+1)%len(region)
                if region[i] == escapee:
                    if region[after]==partner:
                        if region[before]>=num_unaltered_vertices: # Has already escaped to infinity?
                            new_region[i] = len(vertices)
                        else:
                            new_region.insert(after, len(vertices))
                    elif region[before]==partner:
                        if region[after]>=num_unaltered_vertices:
                            new_region[i] = len(vertices)
                        else:
                            new_region.insert(i, len(vertices))
            regions[r] = new_region
        bbox_intersection = calc_bbox_intersection(ridge_points[e], vertices[partner], vor.points, bounds)
        vertices.append(bbox_intersection)
        
    # Delete any lingering ridges at infinity from each region
    for r, region in enumerate(regions):
        regions[r] = list(filter(lambda x: x>=0, region))

    # Insert corners into appropriate regions
    for i in range(4):
        corner = [bounds[i//2][0], bounds[i%2][1]]
        closest_region_ind = vor.point_region[find_closest(corner, vor.points)]
        regions[closest_region_ind].append(len(vertices))
        vertices.append(corner)
    
    # Sort regions so all points go counterclockwise (borrowed from https://gist.github.com/pv/8036995)
    for r, region in enumerate(regions):
        if len(region) == 0:
            continue
        vs = asarray([vertices[v] for v in region])
        c = vs.mean(axis=0)
        angles = arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        regions[r] = list(array(region)[argsort(angles)])
    
    # Finally, match up regions with the initial points
    regions = array(regions, dtype='object')
    regions = list(regions[vor.point_region])
    
    return regions, vertices

def voronoi_polygons_bbox_2d(points, bmin, bmax):
    vor = Voronoi(points, incremental=True)
    vor.close()

    return voronoi_polygons_bbox_2d(vor, bmin, bmax)

def calc_area(region, vertices):
    if(len(region) < 3):
        return 0
    reg = region.copy()
    reg.append(reg[0])
    reg.append(reg[1])
    area = 0
    xs = [vertices[v][0] for v in reg]
    ys = [vertices[v][1] for v in reg]
    for i in range(len(region)):
        area += 0.5*xs[i+1]*(ys[i+2]-ys[i])
    return area

def make_polygon(region, vertices):
    xy = array([array(vertices[v]) for v in region])
    return Polygon(xy)

def voronoi_hist_from_1D_hist(hist, points, bmin, bmax):
    vor = Voronoi(points, incremental=True)
    vor.close()

    regions, vertices = voronoi_polygons_bbox_2d(points, bmin, bmax)
    areas = [calc_area(reg, vertices) for reg in regions]
    scaled = array(hist)/array(areas)
    
    return list(scaled), regions, vertices

def get_polygons(points, bmin, bmax):
    vor = Voronoi(points, incremental=True)
    vor.close()
    
    regions, vertices = voronoi_polygons_bbox_2d(vor, bmin, bmax)
    areas = [calc_area(reg, vertices) for reg in regions]
    scaled = array(hist)/array(areas)
    smax = max(scaled)
    return [make_polygon(reg,vertices) for reg in regions]

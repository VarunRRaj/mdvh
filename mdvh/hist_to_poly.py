import matplotlib.pyplot as plt
import pandas as pd
from math import isclose
from numpy import *
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d


def dist2(p1, p2):
    # Calculate the squared distance between two points
    return (p1[0]-p2[0])**2. + (p1[1]-p2[1])**2.

def find_closest(test, sites):
    # Find the closest site to a test point
    min_dist = Inf
    closest_ind = -1
    for i, site in enumerate(sites):
        dist = dist2(site,test)
        if dist < min_dist:
            min_dist = dist
            closest_ind = i
    return closest_ind

def in_box(test, bmin, bmax):
    # Check if a test point lies inside the bounding box
    for coord, b in zip(test, bmin):
        if coord < b:
            return False
    for coord, b in zip(test, bmax):
        if coord > b:
            return False
    return True

def calc_area(region, vertices):
    # Use the shoelace theorem to compute the area of a region
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

def voronoi_hist_from_1D_hist(hist, sites, bmin, bmax):
    # Scale a histogram by the areas of the voronoi cells
    vor = Voronoi(sites, incremental=True)
    vor.close()

    regions, vertices = voronoi_polygons_bbox_2d(sites, bmin, bmax)
    areas = [calc_area(reg, vertices) for reg in regions]
    scaled = array(hist)/array(areas)
    
    return list(scaled), regions, vertices

def get_polygons(sites, bmin, bmax):
    vor = Voronoi(sites, incremental=True)
    vor.close()
    
    regions, vertices = voronoi_polygons_bbox_2d(sites, bmin, bmax)
    
    return [make_polygon(reg,vertices) for reg in regions]

def along_boundary(test, site_pair, sites):
    # Check if a given test point is on the boundary between two sites
    dists = [dist2(test, sites[s]) for s in site_pair] # test should be equidistant from the sites
    neighbor = find_closest(test, sites) # The closest site to test should be in site_pair
    return neighbor in site_pair and isclose(dists[0], dists[1])

def slice_bbox(vertex, site_pair, sites, bmin, bmax):
    # Calculates entry and exit points for ray that passes through vertex and is a perpendicular bisector
    # to the pair of sites
    pts = [array(sites[s]) for s in site_pair]
    direction = pts[0]+pts[1] - 2*array(vertex) # v2 = p2 + p1 - v1, v2 is the mirror of v1 across p1-p2
    
    itxs = []
    for bound in [bmin, bmax]:
        for c in [0,1]:
            if direction[c] == 0:
                itxs.append(array([Inf, Inf]))
                itxs[-1][c] = bound[c]
                itxs[-1][1-c] = vertex[1-c]
            else:
                t = (bound[c]-vertex[c])/direction[c] # |v1> + t*|d> = |bound>
                itxs.append(vertex + t * direction)
    
    # Add some slop to avoid floating-point errors
    for itx in itxs:
        for i, c in enumerate(itx):
            if isclose(c,0):
                itx[i] = 0.
    
    result = []
    for itx in itxs:
        if in_box(itx, bmin, bmax) and along_boundary(itx, site_pair, sites):
            result.append(itx)     
    
    return result
    
def escape_bbox(vertex, site_pair, sites, bmin, bmax):
    # Calculates exit point for ray passing through vertex, perpendicular to the pair of sites, that passes
    # between the sites
    assert(in_box(vertex, bmin, bmax))
    slices = slice_bbox(vertex, site_pair, sites, bmin, bmax)
    return slices[0]
    
def voronoi_polygons_bbox_2d(sites, bmin, bmax, verbose=False):
    
    vor = Voronoi(sites, incremental=True)
    vor.close()
    
    if vor.points.shape[1] != 2:
        raise ValueError("Can only plot 2d Voronoi Histograms")
    
    # Prepare data structures
    regions = [vor.regions[p] for p in vor.point_region]
    vertices = list(vor.vertices[:])
    points = list(vor.points[:])
    ridge_points = list(vor.ridge_points[:])
    ridge_vertices = list(vor.ridge_vertices[:])
    
    corners = []
    for bound1 in bmin, bmax:
        for bound2 in bmin, bmax:
            corners.append([bound1[0], bound2[1]])
    
    vertex_labels = list(range(len(vertices)))
    
    # Relabel all infinite vertices by looping through all ridges
    true_infinities = 0
    for e, ridge in enumerate(ridge_vertices):
        if -1 in ridge:
            # Insert the new infinities into the appropriate regions
            true_infinities += 1
            new_label = -1-true_infinities
            ridge[ridge.index(-1)] = new_label
            other_vertex = max(ridge)
            for reg in [regions[p] for p in ridge_points[e]]:
                reg.append(new_label)
            vertex_labels.append(new_label)
    # Count number of "true infinities"
    last_infinity_label = -1-true_infinities
    # Remove -1 from all regions
    for reg in regions:
        if -1 in reg:
            reg.remove(-1)
    
    # Relabel vertices outside bounding box
    # Edit all regions and ridges to use the new labels
    new_label = last_infinity_label-1
    for v, vtx in enumerate(vertices):
        if not in_box(vtx, bmin, bmax):
            new_label -= 1
            vertex_labels[v] = new_label
            # Edit all ridges to use the new label
            for rv in ridge_vertices:
                if v in rv:
                    rv[rv.index(v)] = new_label
            # Edit all regions to use the new label
            for reg in regions:
                if v in reg:
                    reg[reg.index(v)] = new_label            
    
    # Keep edges with vertices outside bounding box that intersect the bounding box
    new_ridge_points = []
    new_ridge_vertices = []
    for e, rv in enumerate(ridge_vertices):
        # Check for ridges where both points are outside the bounding box
        rp = ridge_points[e]
        if rv[0] < 0 and rv[1] < 0:
            assert(any(array(rv) < last_infinity_label)) # There shouldn't be any ridges between true infinities.
            # Check if ridge/ray intersects the bounding box.
            new_vtxs = slice_bbox(vertices[vertex_labels.index(min(rv))], ridge_points[e], points, bmin, bmax)
            # If the ridge/ray intersects the bbox:
            if len(new_vtxs) > 0:
                # Draw the ray from minimum label (which should always be a finite vertex)
                new_vtxs = slice_bbox(vertices[vertex_labels.index(min(rv))], ridge_points[e], points, bmin, bmax)
                
                for new_vtx in new_vtxs:
                    vertex_labels.append(len(vertices))
                    vertices.append(new_vtx)
                
                for reg in regions:
                    if rv[0] in reg and rv[1] in reg:
                        reg.append(vertex_labels[-1])
                        reg.append(vertex_labels[-2])  
                
                new_ridge_points.append(ridge_points[e])
                new_ridge_vertices.append([vertex_labels[-1], vertex_labels[-2]])
                
            # If the ridge/ray does not intersect the bbox we can safely ignroe it
            else:
                pass
        
        # If edge escapes bounding box from inside, calculate intersection of ray from inside point to bounding box
        elif rv[0] < 0 or rv[1] < 0:

            finite_vertex = vertices[max(rv)]
            new_vtx = escape_bbox(finite_vertex, ridge_points[e], points, bmin, bmax)
            
            vertex_labels.append(len(vertices))
            vertices.append(new_vtx)
            
            for reg in regions:
                if rv[0] in reg and rv[1] in reg:
                    reg.append(vertex_labels[-1])
                       
            new_ridge_points.append(ridge_points[e])
            new_ridge_vertices.append([max(rv), vertex_labels[-1]])
            
        # Keep ridges that are definitiely in the bounding box
        else:
            new_ridge_points.append(ridge_points[e])
            new_ridge_vertices.append(rv)
            
    ridge_points = new_ridge_points[:]
    ridge_vertices = new_ridge_vertices[:]
    
    # Remove negative vertices from regions
    new_regions = []
    for reg in regions:
        new_regions.append([])
        for v in reg:
            if v >= 0:
                new_regions[-1].append(v)
    regions = new_regions[:]
    
    # Clean up vertex labels
    # TODO, not strictly necessary
    # Without this step, the output list of vertices will include the original vertices
    # which lie outside the bounding box. However, none of the regions will use these
    # vertices, so skipping this step won't affect final polygons.
    
    # Insert bbox corners into appropriate regions
    for corner in corners:
        closest_site = find_closest(corner, points)
        vertex_labels.append(len(vertices))
        vertices.append(corner)
        regions[closest_site].append(vertex_labels[-1])
    
    # Sort regions so all points go counterclockwise (borrowed from https://gist.github.com/pv/8036995)
    for r, region in enumerate(regions):
        if len(region) == 0:
            continue
        vs = asarray([vertices[v] for v in region])
        c = vs.mean(axis=0)
        angles = arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        regions[r] = list(array(region)[argsort(angles)])
        
    return regions, asarray(vertices)

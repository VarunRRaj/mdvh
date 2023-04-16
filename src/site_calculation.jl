module CapacityConstrainedVoronoi

export calc_sites, calc_cell_polygons

using UnROOT, DataFrames, DataFramesMeta, Printf
using Plots, PlotThemes, Optim, Statistics, Random
using Distances, NearestNeighbors, StatsBase
#using VoronoiCells, GeometryBasics
using JSON, TravelingSalesmanHeuristics
using PyCall

function loadFile(filename, cut)::DataFrame
	rfile = ROOTFile(filename);
	df = DataFrame(LazyTree(rfile, cut))
	transform!(df, names(df) .=> ByRow(Float64), renamecols=false)
end

function norm_by_column_std!(df::DataFrame)
	stds = Dict()
	for n in names(df)
		if n != "weight"
			col_std = std(df[!, n])
			df[!, n]/= col_std
			stds[n] = col_std
		end
	end
	transform!(df, names(df) .=> ByRow(Float32), renamecols=false)
	return stds
end

function is_closer(point, old_site, new_site)::Bool
    sqeuclidean(point, new_site) < sqeuclidean(point, old_site)
end

function find_closer(point, sites, old_label::Integer, new_label::Integer)::Integer
    if old_label==new_label
        return old_label
    end
    new_site = sites[new_label]
    old_site = sites[old_label]
    if is_closer(point, old_site, new_site) 
        return new_label
    else
        return old_label
    end
end

function label_points(points, weights, sites::Vector)::DataFrame
    sites_tree::KDTree = KDTree(reduce(hcat, sites))
    labels = [nn(sites_tree, pt)[1] for pt in points]
    DataFrame(point=points, weight=weights, label=labels)
end

function relabel_points!(df::DataFrame, sites::Vector, 
			selected_label::Integer)::DataFrame
	sites_tree::KDTree = KDTree(reduce(hcat, sites))
	df = @eachrow! df begin
		if :label == selected_label
			:label = nn(sites_tree, :point)[1]
		else
			:label = find_closer(:point, sites, :label, selected_label)
		end
	end
end

function calc_capacities(df::DataFrame, num_sites::Int64)::Vector
	caps = zeros(num_sites)
	@eachrow df begin caps[:label] += :weight end
	caps
end

function optimization_func!(df::DataFrame, sites::Vector, new_site::Vector,
			selected_label::Integer, target_cap::AbstractFloat)::AbstractFloat
	sites[selected_label] = new_site
	df = relabel_points!(df, sites, selected_label)
	capacities = calc_capacities(df, length(sites))
	sum((capacities .- target_cap).^2)
end

function capacity_constrained_voronoi(points::Vector, weights::Vector, num_sites::Int64; n_iter::Int64=13, random_seed::Int64=20, target_fom::Float32=0.04, verbose=false)
	Random.seed!(random_seed)
	sites = wsample(points, weights, num_sites, replace=false)

	target_cap = sum(weights)/num_sites
	if verbose
		println("TARGET: ", target_cap)
	end

	stable = false
	iteration=0
	numevals=0

	point_df = label_points(points, weights, sites)

	capacities=Vector(1:num_sites)
	fom=0.0
	anim=Animation()
    theme(:dark)
	
	while !stable
		stable=true

		labels_to_move = Vector(1:num_sites)
		while length(labels_to_move) > 0
			capacities = calc_capacities(point_df, num_sites)
			
			smallest_cap = Inf
			selected_label = 0
			for label in labels_to_move
				cap = capacities[label]
				if cap < smallest_cap
					selected_label = label
					smallest_cap = cap
				end
			end
			idx_to_remove = argmin([capacities[l] for l in labels_to_move])
			selected_label = popat!(labels_to_move, idx_to_remove)

			old_site = sites[selected_label]
			
			to_optimize(site::Vector) = optimization_func!(point_df, sites, site, selected_label, target_cap)
			res = try optimize(to_optimize, old_site, method=NelderMead(), iterations=n_iter)
			catch e
				if isa(e,InterruptException)
					break
				end
			end
			
			new_site = Optim.minimizer(res)
			sites[selected_label] = new_site
			point_df = relabel_points!(point_df, sites, selected_label)
			fom = sqrt(Optim.minimum(res)/num_sites)/target_cap

			numevals += Optim.f_calls(res)
			iteration += 1

			if(iteration%10==0)
				bar(1:num_sites, capacities, label="", color="mediumaquamarine", linewidth=0)
				plot!(0:(num_sites+1), [target_cap for x in 0:(num_sites+1)], 
					ribbon=[target_cap/5 for x in 0:(num_sites+1)],
					label="", color="gold", xlim=(0.5,num_sites+0.5), 
					xticks=0:max(1,trunc(Int, (num_sites)/10)):(num_sites+1),
					linewidth=3)
				my_ylim = ylims()
				plot!(Shape([0,.35*num_sites,.35*num_sites,0],
					[0,0,.3*my_ylim[2],.3*my_ylim[2]]),
					color="moccasin", alpha=0.75, label="")
				annotate!(num_sites*.05, my_ylim[2]*.2,
					("ITERATION: $iteration", 12, :black, :left))
				annotate!(num_sites*.05, my_ylim[2]*.1,
					("FOM: $(@sprintf("%.3f",fom))", 12, :black, :left))
				frame(anim)
			end

			if verbose && iteration%10==0
				println("ITERATIONS: ", iteration)			
				println("FOM: ",fom)
				println("EVALUATIONS: ", numevals)
			end
			if fom > target_fom && !isapprox(sites[selected_label], old_site, atol=0.001)
				stable=false
			end
			if fom < .75*target_fom
				break
			end
		end
	end
	println("Final FOM: ", fom)
	sites, capacities, anim
end

function sort_sites(sites; quality_factor=100)
	l = length(sites)
	distmat = zeros(Float32, l, l)
	dists_to_origin = zeros(Float32, l)
	for i in 1:l
		for j in 1:l
			distmat[i,j] = euclidean(sites[i], sites[j])
		end
		dists_to_origin[i] = sqeuclidean(sites[i], zeros(Float32, length(sites[i])))
	end

	imin = argmin(dists_to_origin)
	imax = argmax(dists_to_origin)
	distmat[imin, imax] = 0
	distmat[imax, imin] = 0

	path, cost = solve_tsp(distmat, quality_factor=quality_factor)

	path = path[1:l]
	
	start_path_idx = findfirst(path .== imin)
	next_idx = (start_path_idx+1)%length(path)
	next_idx = next_idx == 0 ? l : next_idx
	bwd = path[next_idx] == imax
	if bwd
		reverse!(path)
	end
	start_path_idx = findfirst(path .== imin)
	return circshift(path, 1-start_path_idx), cost
end

function calc_sites(filename, cut, vars, num_sites; 
					n_iter=13, subset=typemax(Int64), random_seed=66, 
					target_fom=0.06f0, out_dir=".", verbose=false, save_anim=false)
	df = loadFile(filename, cut)
	stds = norm_by_column_std!(df)

	points = [[row[var] for var in vars] for row in eachrow(df)]
	weights = [row.weight for row in eachrow(df)]

	stride = max(1, trunc(Int, length(weights)/subset))
	points = points[1:stride:length(points)]
	weights = weights[1:stride:length(weights)]

	sites, caps, anim = capacity_constrained_voronoi(points, weights, num_sites,
							n_iter=n_iter, random_seed=random_seed, 
							target_fom=target_fom, verbose=verbose)

	curr = ""
    if occursin("fhc", filename)
        curr = "_fhc"
    elseif occursin("rhc", filename)
        curr = "_rhc"
    end
    
	out_name = "$out_dir/$(@sprintf("%s%s_%d.json", cut, curr, num_sites))"

    if verbose
        println("Sorting and saving sites...")
    end

	sorted_path, cost = sort_sites(sites)
	sorted_sites = [sites[p] for p in sorted_path]

	out_dict = Dict()
	out_dict["names"] = vars
	out_dict["scales"] = [stds[v] for v in vars]
	out_dict["sites"] = sorted_sites

	jout = JSON.json(out_dict)
	open(out_name, "w") do f
		write(f, jout)
	end

	if save_anim
        if verbose
            println("Saving Animation...")
        end
		gif(anim, "$out_dir/$(@sprintf("%s%s_%d.gif", cut, curr, num_sites))")
	end
	
end

function calc_cell_polygons(in_json, unscaled_min, unscaled_max; out_name="")
    jin = JSON.parsefile(in_json)
    sites = jin["sites"]
    scales = jin["scales"]

    bbox_min = unscaled_min ./ scales
    bbox_max = unscaled_max ./ scales

    pushfirst!(PyVector(pyimport("sys")."path"), "")
    poly = pyimport("cell_geometry")
    regions, vertices, areas = poly.voronoi_polygons_bbox_2d(sites, bbox_min, bbox_max)

    vertices = [vertices[i,:] for i in 1:length(vertices[:,1])]

    jin["regions"] = regions
    jin["vertices"] = vertices
    jin["areas"] = areas
    jin["bounds"] = [unscaled_min, unscaled_max]

    if(out_name=="")
        out_name=in_json
    end
    jout = JSON.json(jin)
    open(out_name, "w") do f
        write(f, jout)
    end
end

end

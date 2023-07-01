using LinearAlgebra

"""
plot_neurons(skeletons)
Plot the skeleton models of neurons.
Examples
========
julia> ids = fetch_neurons(NeuronType("APL"))
julia> skeletons = fetch_skeletons(ids)
julia> scene = plot_neurons(skeletons)
"""
function plot_neurons(skeletons; color=RGB(0f0, 0f0, 0f0))
	ids = unique(skeletons.bodyId)
	plot_neurons(skeletons, ids, color=color)
end
function plot_neurons(skeletons, ids; color=RGB(0f0, 0f0, 0f0))
	scene = Scene(show_axis=false)
	plot_neurons!(scene, skeletons, ids, color)
	return scene
end
function plot_neurons!(scene, skeletons, ids, color::Vector{<:Colorant})
	for (i, id) in enumerate(ids)
		linesegments!(scene, to_segments(get_neuron(skeletons, id)), color=color[i], linewidth=4, transparency=true, shininess=0f0)
	end
	return scene
end

function plot_neurons!(scene::Makie.Scene, skeletons; color=RGB(0f0, 0f0, 0f0))
	ids = unique(skeletons.bodyId)
	plot_neurons!(scene, skeletons, ids, color)
end
function plot_neurons!(scene, skeletons, ids, color::Colorant)

	for (i, id) in enumerate(ids)
		linesegments!(scene, to_segments(get_neuron(skeletons, id)), color=color, linewidth=1, transparency=true, shininess=0f0)
	end
	return scene
end


"""
plot_upstream(connections, skeletons)
Plots just the input neurons for a given set of connections.
"""
function plot_upstream(connection_df, skeletons; base_color=RGB(.2,.2,.2))

	scene = Scene(show_axis=false)
	plot_upstream!(scene, connection_df, skeletons, base_color=base_color)
	return scene
end
function plot_upstream!(scene, connection_df, skeletons; base_color=RGB(.2,.2,.2))

	upstream_ids = unique(connection_df.bodyId_pre)
	weights = pre_strength(connection_df)
	weights = normalise(weights)
	colors = Vector{HSLA}(undef, length(upstream_ids));

	for i in 1:length(upstream_ids)
	   colors[i] = RGBA(base_color, weights[i])
	end
	plot_neurons!(scene, skeletons, upstream_ids, colors)
	return scene
end


"""
plot_downstream(connections, skeletons)
Plots just the output neurons for a given set of connections.
"""
function plot_downstream(connection_df, skeletons; base_color=RGB(.2, .2, .2))
	scene = Scene(show_axis=false)
	plot_downstream!(scene, connection_df, skeletons, base_color=base_color)
	return scene
end
function plot_downstream!(scene, connection_df, skeletons; base_color=RGB(.2,.2,.2))
	
	downstream_ids = unique(connection_df.bodyId_post)
	weights = post_strength(connection_df)
	weights = normalise(weights)
	colors = Vector{HSLA}(undef, length(downstream_ids));

	for i in 1:length(downstream_ids)
	   colors[i] = RGBA(base_color, weights[i])
	end
	plot_neurons!(scene, skeletons, downstream_ids, colors)
	return scene
end



function plot_retinotopicity(sk, ids)

	upto= 60

	rand_points() = rand(Point{3,Float64}, upto)
	rand_retina = Vector{Int}(undef, upto)
	actual_retina = similar(rand_retina)

	for i = 1:upto
		rand_retina[i] = sum(calculate_retinotopicity(rand_points(), rand_points(), breadth=i))
		actual_retina[i] = sum(calculate_retinotopicity(sk, ids, breadth=i))
	end
	p = Plots.plot(rand_retina, legend=false)
	Plots.plot!(p, actual_retina)
	return p
end

function retinotopicity_neighboursizes(input_locations, output_locations; max_neighbours=60)

	rand_locations() = rand(Point{3,Float64}, max_neighbours)

	random_results = Vector{Int}(undef, max_neighbours)
	actual_results = Vector{Int}(undef, max_neighbours)
	perfect_results = Vector{Int}(undef, max_neighbours)

	for i = 1:max_neighbours
		random_results[i] = sum(calculate_retinotopicity(rand_locations(), rand_locations(), breadth=i))
		actual_results[i] = sum(calculate_retinotopicity(input_locations, output_locations, breadth=i))
		perfect_results[i] = sum(calculate_retinotopicity(input_locations, input_locations, breadth=i))
	end

	return (random_results, actual_results, perfect_results)
end


function plot_arrows!()

	arrows!([0],[0],[0],[10_000],[0],[0], linewidth=400, arrowsize=1000, arrowcolor=:blue, linecolor=:blue)
	arrows!([0],[0],[0],[0],[10_000],[0], linewidth=400, arrowsize=1000, arrowcolor=:red, linecolor=:red)
	arrows!([0],[0],[0],[0],[0],[-10_000], linewidth=400, arrowsize=1000, arrowcolor=:green, linecolor=:green)
	#text!("Anterior",textsize = 1000.0, position = (1000, 5000, 0), color=:red)
	#text!("Ventral",textsize = 1000.0, position = (1000, 0, -5000), color=:green)
	#text!("Medial",textsize = 1000.0, position = (1000, 1000, 0), color=:blue)
end

function plot_arrows!(s)

	arrows!(s, [0],[0],[0],[10_000],[0],[0], linewidth=400, arrowsize=1000, arrowcolor=:blue, linecolor=:blue)
	arrows!(s, [0],[0],[0],[0],[10_000],[0], linewidth=400, arrowsize=1000, arrowcolor=:red, linecolor=:red)
	arrows!(s, [0],[0],[0],[0],[0],[-10_000], linewidth=400, arrowsize=1000, arrowcolor=:green, linecolor=:green)
	#text!("Anterior",textsize = 1000.0, position = (1000, 5000, 0), color=:red)
	#text!("Ventral",textsize = 1000.0, position = (1000, 0, -5000), color=:green)
	#text!("Medial",textsize = 1000.0, position = (1000, 1000, 0), color=:blue)
end

"""
spin the camera for a scene for an amount of seconds.
"""
function spinning_camera(s, time_seconds)
	s.center = false
	spinning_time = [1:time_seconds;]
	rotate_cam!(s, (0.0,0.0,0.0))	
	for _ in spinning_time		
		rotate_cam!(s, ((2*pi)/time_seconds, 0.0, 0.0))
		sleep(0.01)
	end
end


function save_spinning_camera(s, time_seconds, fname)
	s.center = false
	spinning_time = [1:time_seconds;]

	rotate_cam!(s, (0.0,0.0,0.0))
	record(s,fname, 1:time_seconds; framerate=30) do i
       rotate_cam!(s, (2π)/time_seconds, 0.,  0.)
       #update_cam!(s) # disabling stops the camera reset.
   end
end



"""
Neuron Analysis
"""
# TODO: explain how this works because i've forgotten.
function to_segments(x,y,z,links)
    
    segments = Vector{Pair{Point{3,Float64},Point{3,Float64}}}(undef,length(links));
    
    for i = 1:length(links)
		
		j = links[i]    	
    	if j == -1
    		segments[i] = Point(0,0,0) => Point(0,0,0)
    	else

	        segments[i] = Point(x[i], y[i], z[i]) => Point(x[j], y[j], z[j])
	    end
    end
    return segments
end


xyz_links(df) = return(df.x, df.y, df.z, df.link)


to_segments(df::DataFrame) = to_segments(xyz_links(df)...)


function find_terminals(row_ids, links)
	terminal_ids = Int[]
	for i = 1:length(row_ids)
		row_ids[i] in links || push!(terminal_ids, row_ids[i])
	end
	return terminal_ids
end
find_terminals(df::DataFrame) = find_terminals(df.rowId, df.link)


function terminal_locations(df::DataFrame)

	terminal_row_ids = find_terminals(df)
	terminal_rows = df[terminal_row_ids,:]
	return terminal_rows.x, terminal_rows.y, terminal_rows.z
end


function furthest_terminal(x,y,z, furthest_x, furthest_y, furthest_z)

	distances = distance_3d(x,y,z, furthest_x, furthest_y, furthest_z)
	return argmax(distances)
end


function distance_3d(x1,y1,z1,x2,y2,z2)
	
	δx = x1 .- x2
	δy = y1 .- y2
	δz = z1 .- z2

	distances = sqrt.(δx .^ 2 + δy .^2 + δz .^2)
end


distance(p1::Point{2,<:Real}, p2::Point{2,<:Real}) = sqrt.( (p1[1] - p2[1])^2 + (p1[2] - p1[2])^2 )
distance(p1::Point{3,<:Real}, p2::Point{3,<:Real}) = sqrt.( (p1[1] - p2[1])^2 + (p1[2] - p1[2])^2 + (p1[3] - p2[3])^2 )
distance(p1::NTuple{3,<:Real}, p2::NTuple{3,<:Real}) = sqrt.( (p1[1] - p2[1])^2 + (p1[2] - p1[2])^2 + (p1[3] - p2[3])^2 )
distance(p1, p2) = LinearAlgebra.norm(p1.-p2)


function median_furthest_points(df::DataFrame, ids)

	furthest_points = Vector{Point{3,Float64}}(undef, length(ids))
	median_points = similar(furthest_points)

	for (i,id) in enumerate(ids)
	
		x, y, z = terminal_locations(get_neuron(df, id))

		furthest_id = furthest_terminal(x,y,z, median(x), median(y), median(z))
		furthest_points[i] = Point(x[furthest_id], y[furthest_id], z[furthest_id])
		median_points[i] = Point(median(x), median(y), median(z))
	end

	return furthest_points, median_points
end

function furthest_points(skeletons::DataFrame, ids, furthest_from)

	out = fill((0.,0.,0.,), length(ids))

	for (i, id) in enumerate(ids)
		
		from = furthest_from[i]
		neuron_skeleton = get_neuron(skeletons, id)
		x,y,z,_ = xyz_links(neuron_skeleton)
		x_from = from[1]; 
		y_from = from[2]; 
		z_from = from[3] 
		row = furthest_terminal(x,y,z, x_from, y_from, z_from)
		out[i] = (x[row], y[row], z[row])
	end
	return out

end



get_dimensions_geo(point_array, dims) = [GeometricalPredicates.Point(p[dims]...) for p in point_array]
get_dimensions(p_array,n) = [pnt[n] for pnt in p_array]


function nearest_neighbours(points, n)

	neighbours = Array{Int}(undef, length(points), n)
	distances = Array{Float64}(undef, length(points), n)

	for i in 1:length(points)
	
		neighbours[i,:], distances[i,:] = nearest_neighbour(points[i], points[1:end .!= i], n)

	end	
	return neighbours, distances
end
function nearest_neighbour(point, other_points, n)

	nearest_ids = zeros(Int, n)
	distances = Inf .* ones(n)
	isempty(point) && return (nearest_ids, distances)
	
	for (i,p) in enumerate(other_points)
		
		max_d, max_i = findmax(distances)
		isempty(p) && continue
		d = distance(point, p)
		
		if d < max_d
			nearest_ids[max_i] = i
			distances[max_i] = d
		end

	end
	dist_order = sortperm(distances)
	return nearest_ids[dist_order], distances[dist_order]
end


function common_neighbours(neighbours1, neighbours2)

	common_count = Vector{Int}(undef, size(neighbours1,1))

	for i in 1:size(neighbours1,1)
	
		common_count[i] = sum(common_elements(neighbours1[i,:], neighbours2[i,:]))

	end
	return common_count
end


function common_elements(a, b)

	commons = Vector{Bool}(undef, length(a))
	for (i, elem) in enumerate(a)
		commons[i] = elem in b
	end
	return commons
end 


function calculate_retinotopicity(df::DataFrame, ids; breadth=5)
	
	furthest_points, median_points = median_furthest_points(df, ids)
	return calculate_retinotopicity(furthest_points, median_points, breadth=breadth)
end
function calculate_retinotopicity(furthest_points, median_points; breadth=5)
	
	furthest_neighbours, _ = nearest_neighbours(furthest_points, breadth)
	median_neighours, _ = nearest_neighbours(median_points, breadth)

	return common_neighbours(furthest_neighbours, median_neighours)
end

function slice_skeletons(sk, ax, step)

end

function find_intersection(plane, line)
	
end

"""
k_means: some stats stuff that isn't that useful tbh.
"""
function k_means(x,y, num_classes)
	
	num_points = length(x)
	centres = random_starts(x,y,num_classes)
	data_points = Point.(x,y)

	distances = Array{Float64}(undef, num_points, num_classes)
	classes = Vector{Int}(undef, num_points)
	centre_vectors = zeros(Point{2,Float64},num_classes)
	repeats = 200
	j = 1

	while j < repeats

		for i in 1:num_classes
			distances[:,i] .= distance_between(data_points, centres[i])
		end
		argmin_rows!(classes, distances)
		find_centre_vectors!(centres, centre_vectors, num_classes, classes, data_points)
		centres .= centres .+ centre_vectors
		j += 1
	end

	return centres
end


"""
Helper Functions
"""
function find_centre_vectors!(centres, centre_vectors, num_classes, classes, data_points)

	for i = 1:num_classes
		centre_vectors[i] = mean(find_difference(centres[i], data_points[classes .== i]))
	end
end

find_difference(p1::Point, p2::Vector{Point{2,Float64}}) = [p - p1 for p in p2]

mean(points) = length(points) == 0 ? zero(eltype(points)) : sum(points) / length(points)

median(a) = sort(a)[Int(round(length(a)/2))]

function argmin_rows!(classes,a)

	for i in 1:size(a)[1]

		classes[i] = argmin(a[i,:])

	end
	return classes
end

function argmin_rows(a)

	argmins = Vector{Int}(undef, size(a)[1])

	for i in 1:size(a)[1]

		argmins[i] = argmin(a[i,:])

	end
	return argmins
end

function random_starts(x,y,num_classes)

	xs = uniform_between(minimum(x), maximum(x), num_classes)
	ys = uniform_between(minimum(y), maximum(y), num_classes)

	return Point.(xs,ys)
end

uniform_between(mini, maxi, num_classes) = mini .+ (maxi-mini) .* rand(num_classes)

function distance_between(points1::Vector{<:Point}, point2)
	
	distances = Vector{Float64}(undef, length(points1))
	for (i,p) in enumerate(points1)
		distances[i] = dist_between(p, point2)
	end
	return distances
end

@inline function distance_between(point1::Point, point2) 

	√( (point1[1] - point2[1])^ 2 + (point1[2] - point2[2])^2 )
end

normalise(a) = (a .- minimum(a)) ./ (maximum(a) - minimum(a))
normalise(a, mini, maxi) = (a .- mini) ./ (maxi - mini)

function normalise(a::Vector{Vector{Float64}}, add)
	
	mx = maximum(reduce(max,a))
	mi = minimum(reduce(min,a))
	
	out = copy(a) #this is a bit dumb but some type problem otherwise...

	for (i,v) in enumerate(a)

		out[i] .= add .+ normalise(v, mi, mx)

	end
	return out
end


function get_groups(csv_name)
	CSV.read(csv_name)
end

function id_in_list(ids, idlist)

	out = BitArray{1}(undef, length(ids))
	for (i, id) in enumerate(ids)
		out[i] = id ∈ idlist
	end
	return out
end


function get_group(ids, list)
	id_group = Vector{Int}(undef, length(ids))
	for (i, id) in enumerate(ids)
		location_in_list = list["bodyIds"] .== id
		id_group[i] = list["group"][location_in_list][1]
	end
	return id_group
end


function total_connection_strength(from_neurons, to_neurons, connection_df)

	# find the rows that contain connections from -> to
	valid_connections = map(x-> x ∈ to_neurons, connection_df.bodyId_post)

	weights = zeros(Int, length(from_neurons))
	i = 1

	for pre_neuron in from_neurons

		pre_neuron_location = connection_df.bodyId_pre .== pre_neuron

		weights[i] = sum(connection_df[pre_neuron_location .* valid_connections, :].weight)
		i += 1

	end
	return weights
end

function pre_strength(connection_df)

	pre_ids = unique(connection_df.bodyId_pre)
	post_ids = unique(connection_df.bodyId_post)

	weights = zeros(Int, length(pre_ids))
	for (i, pre_neuron) in enumerate(pre_ids)
		weights[i] = sum(connection_df[connection_df.bodyId_pre .== pre_neuron,:].weight)
	end
	return weights
end

function post_strength(connection_df)

	pre_ids = unique(connection_df.bodyId_pre)
	post_ids = unique(connection_df.bodyId_post)

	weights = zeros(Int, length(post_ids))
	for (i, post_neuron) in enumerate(post_ids)
		weights[i] = sum(connection_df[connection_df.bodyId_post .== post_neuron,:].weight)
	end
	return weights	
end


function synapse_location_statistics(df, ids)

	stats = DataFrame(bodyId=Int[], mean_x=Float64[], mean_y=Float64[], mean_z=Float64[], std_x=Float64[], std_y=Float64[], std_z=Float64[])


	for (i, id) in enumerate(ids)

		id_df = get_neuron(df, id)
		x = mean(id_df.x)
		y = mean(id_df.y)
		z = mean(id_df.z)

		sx = std(id_df.x)
		sy = std(id_df.y)
		sz = std(id_df.z)

		push!(stats,[id,x,y,z,sx,sy,sz])

	end

	return stats
end

function find_common_connections(pre_ids, post_ids, df)

	out = DataFrame(bodyId_pre=Int[], num_connections=Int[], type=String[])

	for (i,id) in enumerate(pre_ids)

		common_connections = sum(df.bodyId_pre .== id)
		push!(out, [id, common_connections, df.type_pre[i]])

	end

	return out
end

get_neuron(df, id) = df[df.bodyId .== id, :];


function weight_transparency(pre_ids, all_ids, df)
	upstream_weights = total_connection_strength(all_ids, pre_ids, df)
	colors = Vector{HSLA}(undef, length(all_ids));
	normalised_weights = normalise(upstream_weights)
	for i in 1:length(unique_up)
	   colors[i] = HSLA(236, .08, .23, normalised_weights[i])
	end
end

function centroid(synapse_locations, id; average=median)
    
	id_synapses = filter(:bodyId => isequal(id), synapse_locations)
	x,y,z = id_synapses.x, id_synapses.y, id_synapses.z 
    if isempty(x)
    	return (0., 0., 0.)
    elseif length(x) == 1
    	return (float(x[]),float(y[]),float(z[]))
    else
	    return (float(average(x)), float(average(y)), float(average(z)))
	end
end


function centroids(ids, synapse_locations; average=mean)

	out = fill((0.,0.,0.,), length(ids))
	for (i,id) in enumerate(ids)
		out[i] = centroid(id, synapse_locations, average=average)
	end
	return out
end

function plot_neuron_and_partner(skeletons, connections, id)

	scene = plot_neurons(skeletons, id, color=RGBA(.1,0.1,0.9,0.2))
	plot_neurons!(scene, skeletons, connections.bodyId_pre[connections.bodyId_post .== id], RGBA(.8,.1,.1,.3))
	return scene
end

function plot_neuron_and_partner!(scene, skeletons, connections, id)

	plot_neurons!(scene, skeletons, id, RGBA(.1,0.1,0.9,0.2))
	plot_neurons!(scene, skeletons, connections.bodyId_pre[connections.bodyId_post .== id], RGBA(.8,.1,.1,.3))
	return scene
end

"""
Plots of various kinds.
"""
function plot_furthest_terminals(df::DataFrame, ids)

	xs = Vector{Float64}(undef, length(ids))
	ys = similar(xs)
	zs = similar(xs)


	for (i,id) in enumerate(ids)
		x, y, z = terminal_locations(get_neuron(df, id))		
		id = furthest_terminal(x,y,z, median(x), median(y), median(z))
		xs[i], ys[i], zs[i] = x[id], y[id], z[id]
	end
	AbstractPlotting.scatter!(xs, ys, zs, markersize=300_000, color=HSLA(269, .96, .56, 1), shininess=0f0, strokewidth=0, transparency=true)
end


function furthest_terminals(df::DataFrame, ids)

	
	xs = Vector{Float64}(undef, length(ids))
	ys = similar(xs)
	zs = similar(xs)


	for (i,id) in enumerate(ids)
		x, y, z = terminal_locations(get_neuron(df, id))		
		id = furthest_terminal(x,y,z, median(x), median(y), median(z))
		xs[i], ys[i], zs[i] = x[id], y[id], z[id]
	end
	return xs, ys, zs
end


function plot_terminals(df::DataFrame, ids)
	s = Scene()
	for id in ids
		x, y, z = terminal_locations(get_neuron(df, id))
		AbstractPlotting.scatter!(x, y, z, markersize=50)
	end
	return s
end

function plot_median_terminals(df::DataFrame, ids)
	
	xs = Vector{Float64}(undef, length(ids))
	ys = similar(xs)
	zs = similar(xs)

	for (i,id) in enumerate(ids)
		x,y,z = terminal_locations(get_neuron(df, id))
		xs[i] = median(x)
		ys[i] = median(y)
		zs[i] = median(z)
	end

	scatter!(xs, ys, zs, markersize=200)
end


function roi_labels(points, roi_volume; scale=5) # assumes points in x,y,z format.	
	"""	
    1. get borders of volume (basically size).
    2. downsample points, (points ./ 2^scale), convert to int32
    3. remove points outside of volume.
    4. labels = volume[x,y,z]
	"""
	lx, ly, lz = size(roi_volume)
	down_points = round.(Int, points ./ 2^scale)
	x, y, z = view(down_points, :, 1), view(down_points, :, 2), view(down_points, :, 3)
	in_limit_idx = (0 .<= x .< lx) .& (0 .<= y .< ly) .& (0 .<= z .< lz)
	vx, vy, vz = x[in_limit_idx], y[in_limit_idx], z[in_limit_idx]
	labels = map(i-> roi_volume[i...], zip(vx,vy,vz))
	return labels
end
function roi_labels(points, roi_volume, label_key; scale=5)
	labels = roi_labels(points, roi_volume, scale=scale)
	return _roi_label_index(label_key, labels)
end
function roi_labels(df::AbstractDataFrame, args...; scale=5) 
	roi_labels(hcat(df.x,df.y,df.z), args..., scale=scale)
end

_roi_label_index(label_key, labels) = label_key[_py2ju_ind.(labels)]
_roi_label_index(vol_file::HDF5.File, labels) = read(vol_file["primary-roi-names"])[_py2ju_ind.(labels)]

_py2ju_ind(i) = i+1
_ju2py_ind(i) = i-1

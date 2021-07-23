function collapse_duplicates(connection_df)

	unique_connections = unique(connection_df,[1,2])[:,[1,2]]
	num_connections = size(unique_connections)[1]
	weights = zeros(Int, num_connections)
	celltype = fill("", num_connections)

	for connection in 1:num_connections


		pre_matches = connection_df.bodyId_pre .== unique_connections[connection,1]
		post_matches =  connection_df.bodyId_post .== unique_connections[connection,2]
		matching_rows = pre_matches .& post_matches
		weights[connection] = sum(connection_df[matching_rows,:].weight)
		celltype[connection] = choose_type(connection_df[matching_rows,:].type_pre)
	end
	return DataFrame(bodyId_pre=unique_connections[:,1], bodyId_post=unique_connections[:,2], weights=weights, type_pre=celltype)
end


function choose_type(type_pres)
	
	non_missing_types = .!ismissing.(type_pres)
	# if all are missing, return ""
	any(non_missing_types) ? type_pres[findfirst(.!ismissing.(type_pres))] : ""
end


function count_celltypes_weighted(up_df)

	presynaptic_types = Vector{String}(up_df["type_pre"])
	presynaptic_uniquetypes = unique(presynaptic_types)	
	filter!(!isempty, presynaptic_uniquetypes)

	counts = Vector{Int}(undef, length(presynaptic_uniquetypes))

	i = 1
	for unique_type in presynaptic_uniquetypes
	    counts[i] = sum(up_df[presynaptic_types .== unique_type,:].weights)
	    i += 1
	end
	return DataFrame(celltype=presynaptic_uniquetypes, count=counts)
end
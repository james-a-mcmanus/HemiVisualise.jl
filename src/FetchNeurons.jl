import HemiVisualise: neuprint

abstract type AbstractCriteria end
abstract type AbstractROI <: AbstractCriteria end
abstract type AbstractID <: AbstractCriteria end


struct CellType <: AbstractCriteria
	name::String
end

struct InputROI <: AbstractROI
	name::String
end

struct OutputROI <: AbstractROI
	name::String
end

struct BodyROI <: AbstractROI
	name::String
end

struct StringId <: AbstractID
	name::String
end
struct IntId <: AbstractID
	name::Int
end
struct ArrayId <: AbstractID
	name::Vector{Int}
end

"""
Fetch a dataframe containing the neuron IDs, + the ROI for some given criteria.
"""
function neuron_ids(neuprint::PyCall.PyObject, crit)
	_, neurons = neuprint.fetch_neurons(criterion(neuprint, crit))
	return pd_to_df(neurons)
end
neuron_ids(crit) = neuron_ids(neuprint(), crit)
function neuron_ids(crits::AbstractArray)

	criteria = []
	for crit in crits
		push!(criteria, criterion(crit))
	end
	neuron_ids(criteria...)
end

"""
handles conversion from python pandas to julia Dataframe
"""
function pd_to_df(df_pd)
    df= DataFrame()
	for col in df_pd.columns
		df[:, col] = convert_column(getproperty(df_pd,col))
	end
    df
end


"""
Conversion of column types, inferring the type of the data
"""
function convert_column(column)
	coltype = julia_equivalent(column.dtype)
	out = Vector{coltype}(undef, length(column)) # bad practise, but assume that the type is the same for the whole column.
	for i = 1:length(column)
		out[i] = isnothing(column.iloc[i]) ? none_equivalent(coltype) : convert(coltype, column.iloc[i])
	end
	return out
end


"""
Guess the julia equivalent of a python type
"""
function julia_equivalent(pytype)
	if pytype == pybuiltin(:int)
		return Int
	elseif contains(pytype.name, "int")
		return Int
	elseif contains(pytype.name, "float")
		return Float64
	elseif pytype == pybuiltin(:object)
		return String
	elseif pytype == pybuiltin(:float)
		return Float64
	end
end


"""
returns the zero() equivalent for a type, including strings. not actually equivalent, but blank
"""
none_equivalent(s::Type{String}) = ""
none_equivalent(i::Type{Int}) = zero(i)
none_equivalent(f::Type{Float64}) = zero(f)


"""
Takes in a criteria and returns a criterion that can be used by neuprint.
"""
criterion(neuprint::PyCall.PyObject, c::CellType) = neuprint.NeuronCriteria(type=c.name)
criterion(neuprint::PyCall.PyObject, c::IntId) = neuprint.NeuronCriteria(bodyId=c.name)
criterion(neuprint::PyCall.PyObject, c::String) = neuprint.NeuronCriteria(type=c)
criterion(neuprint::PyCall.PyObject, c::Int) = neuprint.NeuronCriteria(bodyId=c)
function criterion(c)
	criterion(neuprint(), c)
end


"""
Fetches skeleton data for given criteria.
"""
get_skeletons(ids) = get_skeletons(neuprint(), ids)
get_skeletons(neu::PyCall.PyObject, ids::DataFrame) = get_skeletons(neu, ids.bodyId)
function get_skeletons(neu, ids::AbstractArray)

	ids = unique(ids)
	out = DataFrame(bodyId=Int[], x=Float64[], y=Float64[], z=Float64[], radius=Float64[], link=Int[])

	for id in ids
		print('\r')
		print("Fetching Skeleton:   ", id)
		try
			out = vcat(out, get_skeletons(neu, id))
		catch
		end

	end
	return out
end
function get_skeletons(neu, id)
	was_saved = check_saved_skeleton(id) && return get_saved_skeleton(id)
	df = pd_to_df(neu.default_client().fetch_skeleton(id, format="pandas"))
	df = df[:, 2:end]
	df.bodyId = id
	save_skeleton(id, df)
	return df
end


"""
Get the neurons upstream of (has presynapse with) a given neuron(s)
"""
function upstream_neurons(neuprint::PyCall.PyObject, id_list)
	neuron_df, conn_df = neuprint.fetch_adjacencies(nothing, id_list)
	conn_df = neuprint.merge_neuron_properties(neuron_df, conn_df, ["type", "instance"])
	conn_df.replace(to_replace=[nothing], value="Unknown", inplace=true)
	return pd_to_df(conn_df)
end


"""
Get the neurons downstream of (has postsynapse with) a given neuron(s)
"""
function downstream_neurons(neuprint::PyCall.PyObject, id_list)
	conn_df = neuprint.get_downstream(id_list)
	return pd_to_df(conn_df)
end


"""
Returns true if skeleton/neuron for id is saved
"""
function check_saved_skeleton(id)
	return any(x-> splitext(basename(x))[1] == string(id), readdir(skeleton_folder(),join=true))
end

function check_saved_neuron(id)
end

"""
fetches saved skeletons/neurons from local storage.
"""
function get_saved_skeleton(id)
	skelfile = filter(x-> splitext(basename(x))[1] == string(id), readdir(skeleton_folder(),join=true))[]
	return CSV.read(skelfile, DataFrame)
end


"""
Saves skeletons/neurons to local storage
"""
save_skeleton(id, df) = CSV.write(joinpath(skeleton_folder(), string(id) * ".csv"), df)


"""
returns folders for local storage
"""
skeleton_folder() = joinpath(dirname(pathof(HemiVisualise)),"SavedData","Skeletons")
neuron_folder() = joinpath(dirname(pathof(HemiVisualise)),"SavedData","Neurons")
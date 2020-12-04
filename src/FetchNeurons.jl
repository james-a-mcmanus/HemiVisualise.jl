abstract type AbstractCriteria end
abstract type AbstractROI <: AbstractCriteria end
abstract type AbstractID <: AbstractCriteria end


struct NeuronType <: AbstractCriteria
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

function get_neurons(crit::AbstractCriteria)
	NEU.fetch_neurons(criterion(crit))
end

function get_neurons(crit::NeuronType)
	_, out_pandas = NEU.fetch_neurons(NEU.NeuronCriteria(type=crit.name))
	pd_to_df(out_pandas)
end

function get_neurons(crits::Vector{AbstractCriteria})

	criteria = []
	for crit in crits
		push!(criteria, criterion(crit))
	end
	NEU.fetch_neurons(criteria...)
end

criterion(c::NeuronType) = NEU.NeuronCriteria(type=c)
criterion(c::IntId) = criterion(c.name)
criterion(c::ArrayId) = criterion(c.name)
criterion(c::StringId) = criterion(parse(Int, c.name))
criterion(c::Int) = NEU.NeuronCriteria(bodyId=c)

function pd_to_df(df_pd)
    df= DataFrame()
	for col in df_pd.columns
		df[!, col] = convert_column(getproperty(df_pd,col))
	end
    df
end

function convert_column(column)

	coltype = julia_equivalent(column.dtype)
	out = Vector{coltype}(undef, length(column)) # bad practise, but assume that the type is the same for the whole column.
	for i = 1:length(column)
		out[i] = convert(coltype, column[i])
	end
	return out
end

function fetch_skeletons(ids)

	out = DataFrame(bodyId=Int[], x=Float64[], y=Float64[], z=Float64[], radius=Float64[], link=Int[])

	for id in ids
		print('\r')
		print("Fetching Skeleton:   ", id)
		df = pd_to_df(NEU.CLIENT.fetch_skeleton(id, format="pandas"))
		df = df[:, 2:end]
		df.bodyId = id
		out = vcat(out, df)
	end

	return out
end

function julia_equivalent(pytype)

	if pytype == pybuiltin(:int)
		return Int
	elseif pytype == pybuiltin(:object)
		return String
	elseif pytype == pybuiltin(:float)
		return Float64
	end
end

function get_ids(df)
	unique(df.bodyId)
end
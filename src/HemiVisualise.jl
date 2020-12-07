__precompile__() # this module is safe to precompile
module HemiVisualise

using CSV, DataFrames, PyCall, AbstractPlotting, Colors, Makie

const NEU = PyNULL()

function __init__()
	pushfirst!(PyVector(pyimport("sys")."path"), "")
	copy!(NEU, pyimport("NeuprintInterface"))
end

include("FetchNeurons.jl")
include("PlotNeurons.jl")

end

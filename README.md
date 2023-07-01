# HemiVisualise.jl

## Install
Navigate into your .julia/dev directory
```
git clone https://github.com/james-a-mcmanus/HemiVisualise.jl.git
cd HemiVisualise.jl/src
julia
julia> dev HemiVisualise
```

## QuickStart
```
julia> using HemiVisualise
julia> cell_df = neuron_ids("KCg-d")
julia> skeletons = get_skeletons(unique(cell_df.bodyId))
julia> scene = plot_neurons(skeletons, color=RGBA(0,0,0,.2))
```
<video src='https://raw.githubusercontent.com/james-a-mcmanus/HemiVisualise.jl/master/Demo.mp4' />

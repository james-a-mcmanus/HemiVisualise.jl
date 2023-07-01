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
julia> using HemiVisualise # This may take some time
julia> neurons = get_ids(get_neurons(NeuronType("KCg-d")))
julia> skeletons = fetch_skeletons(neurons)
julia> scene = plot_neurons(skeletons, color=RGBA(0,0,0,.2))
```



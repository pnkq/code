using DelimitedFiles
using Statistics
using Plots
plotly()

path = "/home/phuonglh/code/con/dat/depx-scores-uas-sun.tsv"

A = readdlm(path, ';')

language = "eng"
ts = ["t", "t+p", "f", "x"]
# ws = [32, 64, 128, 200]
# hs = [64, 128, 200, 300]

function filter(modelType::String, w::Int, h::Int)
  j = (A[:, 1] .== language) .& (A[:, 2] .== modelType)
  E = Float64.(A[j, 3:end])

  i = (E[:, 1] .== w) .& (E[:, 2] .== h)
  F = E[i, 5:end]
  avg = mean(F, dims=1)
  return (w, h, avg)
end

for t in ts
  print(t, "\t")
  local WHR = filter(t, 200, 300)
  println(WHR)
end
# for triple in WHRs
#   plot!(triple[1], triple[2], triple[3], st=:scatter, xlabel="token emb. size", ylabel="recurrent size", zlabel="accuracy", legend=false)
# end


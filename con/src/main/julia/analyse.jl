using DelimitedFiles
using Statistics
using Plots
plotly()

path = "/home/phuonglh/code/con/dat/depx-scores-uas-sun.tsv"

A = readdlm(path, ';')

language = "eng"
ts = ["t", "t+p", "f"]
ws = [32, 64, 128, 200]
hs = [64, 128, 200, 300]

function filter(modelType::String)
  j = (A[:, 1] .== language) .& (A[:, 2] .== modelType)
  E = Float64.(A[j, 3:end])

  W = []
  H = []
  R = []
  for w in ws
    for h in hs
      i = (E[:, 1] .== w) .& (E[:, 2] .== h)
      F = E[i, 5:end]
      avg = mean(F, dims=1)
      append!(W, w)
      append!(H, h)
      append!(R, avg[2])
    end
  end
  return (W, H, R)
end

WHRs = map(t -> filter(t), ts)
print(WHRs[1])
# for triple in WHRs
#   plot!(triple[1], triple[2], triple[3], st=:scatter, xlabel="token emb. size", ylabel="recurrent size", zlabel="accuracy", legend=false)
# end


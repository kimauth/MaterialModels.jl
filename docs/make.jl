using Documenter, MaterialModels

makedocs(sitename="MaterialModels.jl",  format = Documenter.HTML(prettyurls = false))

deploydocs(
    repo = "github.com/kimauth/MaterialModels.jl.git",
    devbranch = "main",
)
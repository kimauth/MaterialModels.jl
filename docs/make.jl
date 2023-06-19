using Documenter, MaterialModels, Tensors

makedocs(
    sitename="MaterialModels.jl",  
    format = Documenter.HTML(prettyurls = false),
    pages = Any[
        "Home" => "index.md",
        "Interface" => "interface.md",
        "Materials" => [
            "materials/LinearElastic.md",
            "materials/TransverselyIsotropic.md",
            "materials/Plastic.md",
            "materials/XuNeedleman.md",
        ]
    ]
)

deploydocs(
    repo = "github.com/kimauth/MaterialModels.jl.git",
    devbranch = "main",
    push_preview=true,
)

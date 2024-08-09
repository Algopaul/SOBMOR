using SOBMOR
using Documenter

DocMeta.setdocmeta!(SOBMOR, :DocTestSetup, :(using SOBMOR); recursive=true)

makedocs(;
  modules=[SOBMOR],
  authors="Algopaul <72927083+Algopaul@users.noreply.github.com> and contributors",
  sitename="SOBMOR.jl",
  format=Documenter.HTML(; edit_link="main", assets=String[]),
  pages=["Home" => "index.md"],
)

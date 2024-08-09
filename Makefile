include runner_definition.mk

data:
	mkdir -p data

install:
	$(RUN) julia -e "using Pkg; Pkg.activate(\".\"); Pkg.add(url=\"https://github.com/Algopaul/PortHamiltonianBenchmarkSystems.jl/\")"
	$(RUN) julia -e "using Pkg; Pkg.activate(\".\"); Pkg.instantiate()"

.PHONY: test
test:
	$(RUN) julia -e "using Pkg; Pkg.activate(\".\"); Pkg.test()"


demo_portHamiltonian20:
	$(RUN) julia ./scripts/driver.jl

demo_parametricMOR: data
	$(RUN) julia ./scripts/driver_parametric.jl --reduced_order=10

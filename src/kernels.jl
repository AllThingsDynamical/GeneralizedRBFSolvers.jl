abstract type PositiveDefiniteKernel end

struct SquaredExponentialKernel <: PositiveDefiniteKernel

end

struct CauchyKernel <: PositiveDefiniteKernel

end

struct Matern72Kernel <: PositiveDefiniteKernel

end

struct Matern52Kernel <: PositiveDefiniteKernel

end
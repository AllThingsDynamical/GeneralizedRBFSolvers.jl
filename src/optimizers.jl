using ForwardDiff
using LinearAlgebra
using Plots
TEST = false

struct GaussNewton{T}
    R::Function
    maxiters::Int
    tol::T
    λ::T
end

function gauss_newton(model::GaussNewton, x0)
    x = x0
    history = []
    for i in 1:model.maxiters
        r = model.R(x)
        J = ForwardDiff.jacobian(model.R, x)'
        V = -(J'*J+ model.λ *I) \ (J'*r')
        x += V
        rtres = norm(rosenbrock_residual(x))^2
        @show (i,rtres)
        push!(history, rtres)
        if rtres < model.tol
            break
        end
    end
    return x, history
end


function rosenbrock_residual(x)
    n = length(x)
    @assert iseven(n) "Length of x must be even"
    r = zeros(eltype(x), n)
    for i in 1:2:n
        r[i]   = 10 * (x[i+1] - x[i]^2)
        r[i+1] = 1 - x[i]
    end
    return r'
end
Dres = x -> ForwardDiff.jacobian(rosenbrock_residual, x)'

if TEST
    n = 20
    x0 = rand(n) # Example for 4D Rosenbrock
    res = rosenbrock_residual(x0)
    optimizer = GaussNewton(rosenbrock_residual, 2000, 1e-6, 1e2)
    x_opt, history = gauss_newton(optimizer, x0)
    display(plot(history, yaxis=:log))
    norm(x_opt - ones(n)).^2
end


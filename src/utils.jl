include("samplers.jl")
include("kernels.jl")
using QuasiMonteCarlo
using LinearAlgebra
using NonlinearSolve
using Statistics



function bounds(domain, ndims)
    lb = []
    ub = []
    for i=1:ndims
        push!(lb, domain[i][1])
        push!(ub, domain[i][2])
    end
    if ndims == 3
        return float.(lb), float.(ub)
    else 
        return tuple(float.(vcat(lb, 0.0))...), tuple(float.(vcat(ub, 0.0))...)
    end
end

function resolve(resolution, ndims)
    Ns = []
    for i=1:ndims
        push!(Ns, resolution[i])
    end
    return Ns
end

function sample_domain(domain, resolution)
    ndims = length(domain)
    lb, ub = bounds(domain, ndims)
    Nx, Ny = resolve(resolution, ndims)
    grid = RectangularGrid(lb, ub, Ny, Nx)
    sampler = QuasiMonteCarlo.HaltonSample
    interior_points, boundary_points = sample(grid, sampler)
    interior_points, boundary_points
end

function eval_kernels(X, Y, kernel)
    m = size(X,2)
    n = size(Y,2)
    K = zeros(m,n)
    for i=1:m
        for j=1:n
            x = X[:,i]
            y = Y[:,j]
            K[i,j] = kernel(x[1], x[2], 0.0,  y[1], y[2], 0.0)
        end
    end
    return K
end

function evaluated_kernels(Kernels, interior_points, boundary_points)
    r = Int(sqrt(length(Kernels)))
    K11 = eval_kernels(boundary_points, interior_points, Kernels[1])
    K12 = eval_kernels(boundary_points, boundary_points, Kernels[1])
    K13 = eval_kernels(boundary_points, interior_points, Kernels[2])
    K14 = eval_kernels(boundary_points, interior_points, Kernels[3])
    K1 = hcat(K11, K12, K13, K14)

    K21 = eval_kernels(interior_points, interior_points, Kernels[1])
    K22 = eval_kernels(interior_points, boundary_points, Kernels[1])
    K23 = eval_kernels(interior_points, interior_points, Kernels[2])
    K24 = eval_kernels(interior_points, interior_points, Kernels[3])
    K2 = hcat(K21, K22, K23, K24)

    K31 = eval_kernels(interior_points, interior_points, Kernels[4])
    K32 = eval_kernels(interior_points, boundary_points, Kernels[4])
    K33 = eval_kernels(interior_points, interior_points, Kernels[5])
    K34 = eval_kernels(interior_points, interior_points, Kernels[6])
    K3 = hcat(K31, K32, K33, K34)

    K41 = eval_kernels(interior_points, interior_points, Kernels[7])
    K42 = eval_kernels(interior_points, boundary_points, Kernels[7])
    K43 = eval_kernels(interior_points, interior_points, Kernels[8])
    K44 = eval_kernels(interior_points, interior_points, Kernels[9])
    K4 = hcat(K41, K42, K43, K44)

    Θ = vcat(K2, K1, K3, K4)'

    return [K11, K12, K13, K14, K21, K22, K23, K24, K31, K32, K33, K34, K41, K42, K43, K44], Θ
end

using Plots
function spectra_decay(Theta, nugget)
    Theta = Theta + nugget*I
    @assert isposdef(Theta)
    L = cholesky(Theta).L
    inverse = inv(L)'* inv(L)

    pca = svd(Theta)
    pca_i = svd(inverse)
    pca_s = svd(inv(L)')

    
    figure1 = plot(pca.S, yaxis=:log, label="Canonical", xlabel="M", ylabel="Singular values")
    plot!(pca_i.S, yaxis=:log, label="Gamblets")
    plot!(pca_s.S, yaxis=:log, label="Square root")
    return figure1
end

function evaluate_gram_matrix(interior_kernels, boundary_kernels, interior_points, boundary_points, kernel_collection)
    all_kernels = Bool.(vec(interior_kernels .| boundary_kernels))
    m_all_kernels = all_kernels * all_kernels' 
    Kernels = kernel_collection[m_all_kernels]
    E_kernels, Theta = evaluated_kernels(Kernels, interior_points, boundary_points)
    E_kernels, Theta
end


function munge_trace_data(obj::NonlinearSolveBase.NonlinearSolveTrace)
    hist = obj.history
    M = length(hist)
    iterations = []
    condition = []
    residuals = []

    for i=1:M
        f1 = hist[i].fnorm
        f2 = hist[i].condJ
        push!(iterations, i)
        push!(condition, f2)
        push!(residuals, f1)
    end
    return iterations, residuals, condition
end


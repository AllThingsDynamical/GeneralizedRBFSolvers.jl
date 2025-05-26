include("utils.jl")
using ForwardDiff

begin
    domain = ([-pi, pi], [-pi, pi])
    Nx = 100
    NDx = 300
    resolution = (Nx, NDx)
    interior_points, boundary_points = sample_domain(domain, resolution)
    NI = size(interior_points, 2)
    NB = size(boundary_points, 2)
    Σ = [1.2, 1.2, 5e-1]
    kernel_object = SquaredExponentialKernel(Σ[1], Σ[2], Σ[3])
    kernel_collection = get_kernels(kernel_object)
    interior_kernels = [1 0 0 0 1 1 0]
    boundary_kernels = [1 0 0 0 0 0 0]
    E_kernels, Theta = evaluate_gram_matrix(interior_kernels, boundary_kernels, interior_points, boundary_points, kernel_collection)
    nugget = 1e-7
    display(spectra_decay(Theta, nugget))

    all_kernels = Bool.(vec(interior_kernels .| boundary_kernels))
    m_all_kernels = all_kernels * all_kernels' 
    Kernels = kernel_collection[m_all_kernels]

    k = 2
    function forcing_function(x, y)
        -2*k^2*sin(k*x)*sin(k*y) + (sin(k*x)*sin(k*y))^3
    end

    function solution_function(x,y)
        sin(k*x)*sin(k*y)
    end

    X = collocation_points = hcat(interior_points, boundary_points)[1:2,:]
    UX = [solution_function(X[1,i], X[2,i]) for i=1:size(X,2)]
    FX = [forcing_function(X[1,i], X[2,i]) for i=1:size(X,2)]

    x = LinRange(-pi, pi, 100)
    y = LinRange(-pi, pi, 100)
    XE = reduce(hcat,[[xi, yi] for xi in x, yi in y])
end

begin
    M = size(X,2)
    idx = i -> (i-1)*M + 1 : i*M
    idxa = i -> (i-1)*size(interior_points,2) + 1: i*size(interior_points,2)
    idxb = i -> size(interior_points, 2)+1:size(interior_points, 2)+ Int(M - size(interior_points, 2)) 

    Theta
    idx1 = idx(1)[end] .+ idxa(1)
    idx2 = idx(1)[end] .+ idxa(2)
    idx3 = idx(1)
    idx4 = idxb(1)

    T0 = Theta[1:size(interior_points,2), :]
    T1 = Theta[idx1, :]
    T2 = Theta[idx2, :]
    T3 = Theta[idx4, :]

    T4 = Theta
    issymmetric(T4)
    isposdef(T4)
    nugget = 1e-6
    T4 = T4 + nugget*I
    isposdef(T4)

    L = cholesky(T4).L
    temp = inv(L)
    P = temp'*temp

    FXT = FX[1:size(interior_points,2)]
    FXB = FX[size(interior_points,2)+1:M]

    T1G = T1*P
    T2G = T2*P
    T3G = T3*P
    T0G = T0*P
# end

function scimls_residual(A, p)
    v1 = T1G*A + T2G*A + (T0G*A).^3 .- FXT
    v2 = T3G*A
    v3 = 1e-4*temp*A
    return vcat(v1, v2, v3)
end

begin
    A = zeros(size(Theta, 2))
    nonlinear_problem = NonlinearProblem(scimls_residual, A, [])
    solution = solve(nonlinear_problem, TrustRegion(); trace_level = TraceAll(), store_trace = Val(true))
    solution.stats
    solution.retcode
    solution.resid

    A_opt = solution.u
    y1 = T0G*A_opt
    y2 = UX[1:size(interior_points,2)]
    plot(y2, y2)
    scatter!(y2, y1, ms=1.0)
end

K1 = eval_kernels(XE, X, Kernels[1])
K2 = eval_kernels(XE, X, Kernels[2])
K3 = eval_kernels(XE, X, Kernels[3])
K4 = eval_kernels(XE, interior_points, Kernels[4])
K5 = eval_kernels(XE, interior_points, Kernels[5])
K6 = eval_kernels(XE, interior_points, Kernels[6])
K7 = eval_kernels(XE, interior_points, Kernels[7])
K8 = eval_kernels(XE, interior_points, Kernels[8])
K9 = eval_kernels(XE, interior_points, Kernels[9])

K = hcat(vcat(K1, K2, K3), vcat(K4,K5,K6), vcat(K7,K8,K9))
FXE = [forcing_function(xi, yi) for xi in x, yi in y]
UXE = [solution_function(xi, yi) for xi in x, yi in y]

c = solution.u
L = size(K1, 1)
Fpred = reshape((K[L+1:2*L,:] + K[2*L+1:3*L,:])*P*c + (K[1:L,:]*P*c).^3 , 100,100)
residual_evaluation = Fpred .- FXE

Upred = reshape(K[1:L,:]*P*c, 100, 100)
error_evalution = Upred .- UXE

figure1 = heatmap(x,y,Fpred, clim=(-10,10), title="Approximated f(x,y)")
# scatter!(XE[1,:], XE[2,:], ms=0.01, label=false)
figure2 = heatmap(x,y,FXE, clim=(-10, 10), title="Actual f(x,y)")
scatter!(X[1,:], X[2,:], ms=0.3, label=false)
figure5 = plot(residual_evaluation[:], label="Residual")
figure3 = heatmap(x,y,Upred,clim=(-1,1), title="Approximated u(x,y)")
figure4 = heatmap(x,y,UXE,clim=(-1,1), title="Actual u(x,y)")
figure6 = plot(error_evalution[:], label="Error")
figure7 = heatmap(x, y, residual_evaluation, title="Approximation error - f(x,y)")
figure8 = heatmap(x,y, error_evalution, title="Approximation error - u(x,y)")

iterations, residuals, condition_numbers = munge_trace_data(solution.trace)
figure9 = plot(iterations, residuals, yaxis=:log, label=false, title="Residual convergence", xlabel="Number of iterations", ylabel="Residual")


figure = plot(figure1, figure3, figure2, figure4, figure7, figure8, 
figure5, figure6, figure9, layout=(5,2), size=(1000,1200))
display(figure)
savefig(figure, "GeneralizedRBFSolvers.jl/with_representers_non_linear_poisson.png")

@info "GP bases approximation -- N = $(Nx), $(NB)"
@show maximum(error_evalution), mean(error_evalution)
end
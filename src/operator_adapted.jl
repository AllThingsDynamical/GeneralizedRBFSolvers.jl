include("utils.jl")
begin
    Nx = 1200
    NDx = 100
    domain = ([-pi, pi], [-pi, pi])
    resolution = (Nx, NDx)
    interior_points, boundary_points = sample_domain(domain, resolution)
    NI = size(interior_points, 2)
    NB = size(boundary_points, 2)
    Σ = [1.5, 1.5, 5e-1]
    kernel_object = SquaredExponentialKernel(Σ[1], Σ[2], Σ[3])
    kernel_collection = get_kernels(kernel_object)
    interior_kernels = [1 0 0 0 1 1 0]
    boundary_kernels = [1 0 0 0 0 0 0]
    E_kernels, Theta = evaluate_gram_matrix(interior_kernels, boundary_kernels, interior_points, boundary_points, kernel_collection)
    nugget = 1e-7
    # display(spectra_decay(Theta, nugget))

    all_kernels = Bool.(vec(interior_kernels .| boundary_kernels))
    m_all_kernels = all_kernels * all_kernels' 
    Kernels = kernel_collection[m_all_kernels]

    k = 2
    function forcing_function(x, y)
        -2*k^2*sin(k*x)*sin(k*y)
    end

    function solution_function(x,y)
        sin(k*x)*sin(k*y)
    end

    X = collocation_points = hcat(interior_points, boundary_points)[1:2,:]
    UX = [solution_function(X[1,i], X[2,i]) for i=1:size(X,2)]
    FX = [forcing_function(X[1,i], X[2,i]) for i=1:size(X,2)]
    M = size(X,2)

    x = LinRange(-pi, pi, 100)
    y = LinRange(-pi, pi, 100)
    XE = reduce(hcat,[[xi, yi] for xi in x, yi in y])


    idx = i -> (i-1)*M + 1 : i*M
    idxa = i -> (i-1)*size(interior_points,2) + 1: i*size(interior_points,2)
    idxb = i -> size(interior_points, 2)+1:size(interior_points, 2)+ Int(M - size(interior_points, 2)) 


    idx1 = idx(1)[end] .+ idxa(1)
    idx2 = idx(1)[end] .+ idxa(2)
    idx3 = idx(1)
    idx4 = idxb(1)


    Theta
    T1 = Theta[idx1, :]
    T2 = Theta[idx2, :]
    T3 = Theta[idx4, :]

    T = vcat(T1+T2, T3)
    c =  pinv(T, atol=1e-8)*FX
    β = T
    residual = T*c .- FX
    plot(residual)


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

    L = size(K1, 1)
    Fpred = reshape((K[L+1:2*L,:] + K[2*L+1:3*L,:])*c, 100,100)
    residual_evaluation = Fpred .- FXE

    Upred = reshape(K[1:L,:]*c, 100, 100)
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

    figure = plot(figure1, figure3, figure2, figure4, figure7, figure8, 
    figure5, figure6, layout=(4,2), size=(1000,1000))
    display(figure)
    savefig(figure, "GeneralizedRBFSolvers.jl/with_representers_poisson.png")

    @info "GP bases approximation -- N = $(Nx), $(NB)"
    @show maximum(error_evalution), mean(error_evalution)
end

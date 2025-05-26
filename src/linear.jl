include("utils.jl")

begin 
    begin
        domain = ([-pi, pi], [-pi, pi])
        Nx = 1200
        NDx = 100
        resolution = (Nx, NDx)
        interior_points, boundary_points = sample_domain(domain, resolution)
        NI = size(interior_points, 2)
        NB = size(boundary_points, 2)
        Σ = [1.0, 1.0, 5e-1]
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
    end

# ## How to approximate the forcing function with radial bases.
begin
    M = size(X,2)
    idx = i -> (i-1)*M + 1 : i*M
    idxa = i -> (i-1)*size(interior_points,2) + 1: i*size(interior_points,2)
    idxb = i -> size(interior_points, 2)+1:size(interior_points, 2)+ Int(M - size(interior_points, 2)) 

    T0 = Theta[idx(1), idx(1)]
    heatmap(T0)
    X
    FX
    c = T0 \ FX
    T0*c

    M = size(X,2)
    kernel = kernel_collection[1]
    x = LinRange(-pi, pi, 100)
    y = LinRange(-pi, pi, 100)
    XE = reduce(hcat,[[xi, yi] for xi in x, yi in y])
    K = eval_kernels(XE, X, kernel)
    Fpred = reshape(K*c, 100, 100)
    figure1 = contour(x, y, Fpred)
    scatter!(X[1,:], X[2,:], ms=1.0, label=false)

    Factual = reduce(hcat, forcing_function(xi, yi) for xi in x, yi in y)
    Ferror = reshape(Factual, 100, 100) .- Fpred
    # @info "Radial bases approximation"
    # @show maximum(Ferror)
    
    # figure2 = contour(x,y,Ferror)
    # plot(figure1, figure2, size=(1000, 800))
end

## Can you solve this linear PDE with radial bases approximations
begin
    idx = i -> (i-1)*M + 1 : i*M
    idxa = i -> (i-1)*size(interior_points,2) + 1: i*size(interior_points,2)
    idxb = i -> size(interior_points, 2)+1:size(interior_points, 2)+ Int(M - size(interior_points, 2)) 


    Theta
    idx1 = idx(1)[end] .+ idxa(1)
    idx2 = idx(1)[end] .+ idxa(2)
    idx3 = idx(1)
    idx4 = idxb(1)

    T1 = Theta[idx1, idx3]
    T2 = Theta[idx2, idx3]
    T3 = Theta[idx4, idx3]
    T = vcat(T1 + T2, T3)

    begin
        c =  pinv(T, atol=1e-8)*FX
        β = T
        residual = T*c .- FX
        plot(residual)
        k1 = Kernels[1]
        k2 = Kernels[2]
        k3 = Kernels[3]

        K1 = eval_kernels(XE, X, k1)
        K2 = eval_kernels(XE, X, k2)
        K3 = eval_kernels(XE, X, k3)
        K = vcat(K1, K2, K3)

        # Residual over evaluation points
        FXE = [forcing_function(xi, yi) for xi in x, yi in y]
        UXE = [solution_function(xi, yi) for xi in x, yi in y]
        Fpred = reshape((K2 + K3)*c, 100,100)
        residual_evaluation = Fpred .- FXE

        Upred = reshape(K1*c, 100, 100)
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
        savefig(figure, "GeneralizedRBFSolvers.jl/without_representers_poisson.png")
    end
    end
    @info "Radial bases approximation -- N = $(Nx), $(NB)"
    @show maximum(error_evalution), mean(error_evalution)
end

## How to approximate PDE solultions in the Gamblets bases
begin
    M = size(X,2)
    idx = i -> (i-1)*M + 1 : i*M
    idxa = i -> (i-1)*size(interior_points,2) + 1: i*size(interior_points,2)
    idxb = i -> size(interior_points, 2)+1:size(interior_points, 2)+ Int(M - size(interior_points, 2)) 

    N = size(interior_points, 2)


    Theta
    idx1 = idx(1)[end] .+ idxa(1)
    idx2 = idx(1)[end] .+ idxa(2)
    idx3 = idx(1)
    idx4 = idxb(1)

    T1 = Theta[idx1, idx3]
    T2 = Theta[idx2, idx3]
    T3 = Theta[idx4, idx3]
    T = T1 + T2

    nugget = 1e-10
    T4 = Theta[idx3, idx3] + nugget*I
    issymmetric(T4)
    isposdef(T4)
    L = cholesky(T4).L
    L_inv = inv(L)
    Φ = T*L_inv'*L_inv[:, 1:N]
    FXT = FX[1:N]
    c = pinv(Φ, atol=1e-8)*FXT

    residual = Φ*c .- FXT
    plot(residual)

    η = L_inv'*L_inv[:, 1:N]
    Zpred = K*η*c
    p = size(XE,2)
    Upred = reshape(Zpred[1:p], 100, 100)
    Y1pred = Zpred[p+1:2*p]
    Y2pred = Zpred[2*p+1:end]
    Fpred = reshape(Y1pred + Y2pred, 100, 100)

    residual_evaluation = Fpred .- FXE

    error_evalution = Upred .- UXE

    figure9 = heatmap(x,y,Fpred, clim=(-10,10), title="Approximated f(x,y)")
    # scatter!(XE[1,:], XE[2,:], ms=0.01, label=false)
    figure10 = heatmap(x,y,FXE, clim=(-10, 10), title="Actual f(x,y)")
    scatter!(X[1,:], X[2,:], ms=0.3, label=false)
    figure11 = plot(residual_evaluation[:], label="Residual")
    figure12 = heatmap(x,y,Upred,clim=(-1,1), title="Approximated u(x,y)")
    figure13 = heatmap(x,y,UXE,clim=(-1,1), title="Actual u(x,y)")
    figure14 = plot(error_evalution[:], label="Error")
    figure15 = heatmap(x, y, residual_evaluation, title="Approximation error - f(x,y)")
    figure16 = heatmap(x,y, error_evalution, title="Approximation error - u(x,y)")

    figure = plot(figure9, figure12, figure10, figure13, figure15, figure16, 
    figure11, figure14, layout=(4,2), size=(1000,1000))
    display(figure)
    savefig(figure, "GeneralizedRBFSolvers.jl/without_representers_poisson_Gamblets.png")
end

begin
    S1 = svd(β).S
    S2 = svd(Φ).S
    figure17 = plot(1:length(S1), S1, yaxis=:log, label="Canonical bases", xlabel="Number of eigenvalues", ylabel="Eigenvalues")
    plot!(1:length(S2), S2, yaxis=:log, label="Gamblets bases")
    savefig(figure17, "GeneralizedRBFSolvers.jl/without_representers_conditioning.png")
end


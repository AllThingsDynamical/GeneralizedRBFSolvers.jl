abstract type Grid end 

struct RectangularGrid{T} <: Grid
    lower_bounds::Tuple{T, T, T} # xmin, ymin, tmin
    upper_bounds::Tuple{T, T, T} # xmax, ymax, tmax
    N_boundary_points::Int
    N_interior_points::Int

    function RectangularGrid(lb::Tuple, ub::Tuple, Nboundaries::Int, Ninteriors::Int)
        new{eltype(lb)}(lb, ub, Nboundaries, Ninteriors)
    end
end

function sample(grid::RectangularGrid, sampler)
    ni = grid.N_interior_points
    nb = grid.N_boundary_points
    nbp = floor(Int, nb/6)
    
    lower_bounds = collect(grid.lower_bounds)
    upper_bounds = collect(grid.upper_bounds)
    interior_points = QuasiMonteCarlo.sample(ni, lower_bounds, upper_bounds, sampler())
    lb = [lower_bounds[1], lower_bounds[2], lower_bounds[3]]
    ub = [lower_bounds[1], upper_bounds[2], upper_bounds[3]]
    boundary_points_1 = QuasiMonteCarlo.sample(nbp, lb, ub, sampler())
    lb = [lower_bounds[1], lower_bounds[2], lower_bounds[3]]
    ub = [upper_bounds[1], lower_bounds[2], upper_bounds[3]]
    boundary_points_2 = QuasiMonteCarlo.sample(nbp, lb, ub, sampler())
    lb = [lower_bounds[1], lower_bounds[2], lower_bounds[3]]
    ub = [upper_bounds[1], upper_bounds[2], lower_bounds[3]]
    boundary_points_3 = QuasiMonteCarlo.sample(nbp, lb, ub, sampler())
    lb = [upper_bounds[1], lower_bounds[2], lower_bounds[3]]
    ub = [upper_bounds[1], upper_bounds[2], upper_bounds[3]]
    boundary_points_4 = QuasiMonteCarlo.sample(nbp, lb, ub, sampler())
    lb = [lower_bounds[1], upper_bounds[2], lower_bounds[3]]
    ub = [upper_bounds[1], upper_bounds[2], upper_bounds[3]]
    boundary_points_5 = QuasiMonteCarlo.sample(nbp, lb, ub, sampler())
    lb = [lower_bounds[1], lower_bounds[2], upper_bounds[3]]
    ub = [upper_bounds[1], upper_bounds[2], upper_bounds[3]]
    boundary_points_6 = QuasiMonteCarlo.sample(nbp, lb, ub, sampler())

    boundary_points = [boundary_points_1, boundary_points_2, boundary_points_3, boundary_points_4, boundary_points_5, boundary_points_6]

    if lower_bounds[end] ==0 && upper_bounds[end] == 0
        boundary_points = [boundary_points_1, boundary_points_2, boundary_points_5, boundary_points_4]
    end

    interior_points, hcat(boundary_points...)
end


abstract type NonlinearOptimizers end
abstract type LinearOptimizers end

struct OptimziationFunction

end

struct OptimizationProblem

end

struct OptimizationSolution

end

function __solve(prob::OptimizationProblem, method::Union{NonlinearOptimizers, LinearOptimizers})

end


struct GradientDescent <: NonlinearOptimizers

end

struct Newton <: NonlinearOptimizers

end

struct Nesterov <: NonlinearOptimizers

end

struct LinearSolve <: LinearOptimizers

end
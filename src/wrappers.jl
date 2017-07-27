
function fit_model(::Type{SimpleLTVModel}, fitmethod, args...; kwargs...)
    fun = if fitmethod == :gd
        fit_statespace_gd
    elseif fitmethod == :scs
        fit_statespace
    elseif fitmethod == :constrained
        fit_statespace_constrained
    elseif fitmethod == :dp
        fit_statespace_dp
    end
    fun(args...; kwargs...)
end


function fit_model(::Type{KalmanModel}, x,u,args...; kwargs...)::KalmanModel
    n,T = size(x)
    @assert T > n "The calling convention for x and u is that time is the second dimention (n,T = size(x))"
    m = size(u,1)
    N = n*(n+m)
    model = KalmanModel(zeros(n,n,T),zeros(n,m,T),zeros(N,N,T))
    fit_model!(model, x,u,args...; kwargs...)
    model
end

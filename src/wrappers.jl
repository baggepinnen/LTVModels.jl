
function SimpleLTVModel(fitmethod::Symbol, args...; kwargs...)
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


function KalmanModel(x,u,R1,R2,args...; extend=false, kwargs...)::KalmanModel
    n,T = size(x)
    T  -=1 # To split x in x and xnew
    @assert T > n "The calling convention for x and u is time in the second dimention (n,T = size(x))"
    m     = size(u,1)
    N     = n*(n+m)
    model = KalmanModel(zeros(n,n,T),zeros(n,m,T),zeros(N,N,T),extend)
    LTVModels.KalmanModel(model, x,u,R1,R2,args...; extend=extend, kwargs...)
    model
end

function KalmanModel(x,R1,R2,args...; extend=false, kwargs...)::KalmanModel
    n,T = size(x)
    T  -=1 # To split x in x and xnew
    @assert T > n "The calling convention for x and u is time in the second dimention (n,T = size(x))"
    N     = n^2
    model = KalmanModel(zeros(n,n,T),zeros(n,0,T),zeros(N,N,T),extend)
    LTVModels.KalmanModel(model, x,R1,R2,args...; extend=extend, kwargs...)
    model
end


function KalmanAR(y::AbstractVector,R1,args...; extend=false, kwargs...)
    T = length(y)
    extend || (T -= 1)
    n = size(R1,1)
    model = KalmanAR(zeros(n,T),zeros(n,n,T),extend,0.0)
    LTVModels.KalmanAR(model,y,R1,args...; extend=extend, kwargs...)
end

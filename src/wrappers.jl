# Constructors to create empty models

function SimpleLTVModel(d;extend=false)
    T = length(d)
    n = nstates(d)
    m = ninputs(d)
    N = n*(n+m)
    T -= 1
    model = SimpleLTVModel(zeros(n,n,T),zeros(n,m,T),extend)
end


function KalmanModel(d::AbstractIdData; extend=false, kwargs...)
    T = length(d)
    n = nstates(d)
    m = ninputs(d)
    extend || (T -= 1)
    @assert T > n "The calling convention for x and u is time in the second dimention (n,T = size(x))"
    N     = n*(n+m)
    model = KalmanModel(zeros(n,n,T),zeros(n,m,T),zeros(N,N,T),extend)
end
function LTVAutoRegressive(d, na; extend=false)
    T = length(d)
    extend || (T -= 1)
    model = LTVAutoRegressive(na,zeros(na,T),zeros(na,na,T),extend,0.0)
end

# Constructors that also fit model to data

# TODO: these should be thought through

function KalmanModel(d::AbstractIdData,R1::AbstractMatrix,R2,args...; extend=false, D=1)::KalmanModel
    model = KalmanModel(d,extend=extend)
    KalmanModel(model,d,R1,R2,args...; extend=extend, D=D)
end

function LTVAutoRegressive(d::AnyInput,R1::AbstractMatrix, args...; extend=false, kwargs...)
    na = size(R1,1)
    model = LTVAutoRegressive(d, na, extend=extend)
    LTVAutoRegressive(model,d,R1, args...; extend=extend, kwargs...)
end

# function SimpleLTVModel(fitmethod::Symbol, args...; kwargs...)
#     fun = if fitmethod == :gd
#         fit_statespace_gd
#     elseif fitmethod == :scs
#         fit_statespace
#     elseif fitmethod == :constrained
#         fit_statespace_constrained
#     elseif fitmethod == :dp
#         fit_statespace_dp
#     end
#     fun(args...; kwargs...)
# end

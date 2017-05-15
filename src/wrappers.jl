# fit_model!(model::KalmanModel, x,u,xnew,R1,R2, extend=false)::KalmanModel
export fit_model

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


function fit_model(::Type{KalmanModel}, args...; kwargs...)::KalmanModel
    model = KalmanModel(zeros(1,1,1),zeros(1,1,1),zeros(1,1,1))
    fit_model!(model, args...; kwargs...)
    model
end

export getARXregressor, find_na, arx, bodeconfidence

"""
    getARXregressor(y::AbstractVecOrMat,u::AbstractVecOrMat, na, nb)
Returns a shortened output signal `y` and a regressor matrix `A` such that the least-squares ARX model estimate of order `na,nb` is `y\\A`
Return a regressor matrix used to fit an ARX model on, e.g., the form
`A(z)y = B(z)f(u)`
with output `y` and input `u` where the order of autoregression is `na` and
the order of input moving average is `nb`
# Example
Here we test the model with the Function `f(u) = √(|u|)`
```julia
A     = [1,2*0.7*1,1] # A(z) coeffs
B     = [10,5] # B(z) coeffs
u     = randn(100) # Simulate 100 time steps with Gaussian input
y     = filt(B,A,u)
yr,A  = getARXregressor(y,u,3,2) # We assume that we know the system order 3,2
x     = A\\yr # Estimate model polynomials
plot([yr A*x], lab=["Signal" "Prediction"])
```
For nonlinear ARX-models, see [BasisFunctionExpansions.jl](https://github.com/baggepinnen/BasisFunctionExpansions.jl/)
"""
function getARXregressor(y::AbstractVector,u::AbstractVecOrMat, na, nb)
    assert(length(nb) == size(u,2))
    m    = max(na+1,maximum(nb))
    n    = length(y) - m+1
    offs = m-na-1
    A    = toeplitz(y[offs+na+1:n+na+offs],y[offs+na+1:-1:1])
    y    = copy(A[:,1])
    A    = A[:,2:end]
    for i = 1:length(nb)
        offs = m-nb[i]
        A = [A toeplitz(u[nb[i]+offs:n+nb[i]+offs-1,i],u[nb[i]+offs:-1:1+offs,i])]
    end
    return y,A
end

function getARXregressor(y::AbstractMatrix,u::AbstractVecOrMat, na::Number, nb)
    @assert length(nb) == size(u,1) "Length of nb must equal size(u,1)"
    # assert(length(na) == size(y,1), "Length of na must equal size(y,1)")
    m    = max(na+1,maximum(nb))
    n    = size(y,2) - m+1
    offs = m-na-1
    A    = Matrix{eltype(y)}(n,0)
    yo   = y[:,offs+na+1:n+na+offs]
    for i = 1:size(y,1)
        offs = m-na
        A = [A toeplitz(y[i,na+offs:n+na+offs-1],y[i,na+offs:-1:1+offs])]
    end
    for i = 1:length(nb)
        offs = m-nb[i]
        A = [A toeplitz(u[i,nb[i]+offs:n+nb[i]+offs-1],u[i,nb[i]+offs:-1:1+offs])]
    end
    return yo,A
end

"""
    find_na(y::AbstractVector,n::Int)
Plots the RMSE and AIC For model orders up to `n`. Useful for model selection
"""
function find_na(y::AbstractVector,n::Int)
    error = zeros(n,2)
    for i = 1:n
        w,e = ar(y,i)
        error[i,1] = rms(e)
        error[i,2] = aic(e,i)
        print(i,", ")
    end
    println("Done")
    scatter(error, show=true)
end

# ControlSystems.TransferFunction(h,y::AbstractVector{Float64}, u::AbstractVector{Float64}, na, nb; kwargs...) = arx(h,y,u,na,nb; kwargs...)

"""
    Gtf, Σ = arx(h,y, u, na, nb; λ = 0)
Fit a transfer Function to data using an ARX model.
`nb` and `na` are the orders of the numerator and denominator polynomials.
"""
function arx(h,y::AbstractVector{Float64}, u::AbstractVector{Float64}, na, nb; λ = 0)
    na -= 1
    y_train, A = getARXregressor(y,u, na, nb)

    if λ == 0
        w = A\y_train
    else
        w = (A'A + λ*eye(size(A,2)))\A'y_train
    end
    a,b = params2poly(w,na,nb)
    model = tf(b,a,h)
    Σ = parameter_covariance(y_train, A, w, λ)
    return model, Σ
end

"""
    a,b = params2poly(params,na,nb)
"""
function params2poly(w,na,nb)
    a = [1; -w[1:na]]
    b = w[na+1:end]
    a,b
end

"""
    Σ = parameter_covariance(y_train, A, w, λ=0)
"""
function parameter_covariance(y_train, A, w, λ=0)
    σ² = var(y_train .- A*w)
    iATA = if λ == 0
        inv(A'A)
    else
        ATA = A'A
        ATAλ = factorize(ATA + λ*I)
        ATAλ\ATA/ATAλ
    end
    iATA = (iATA+iATA')/2
    Σ = σ²*iATA + sqrt(eps())*eye(iATA)
end

"""
    bodeconfidence(arxtf::TransferFunction, Σ::Matrix; ω = logspace(0,3,200))
Plot a bode diagram of a transfer function estimated with [`arx`](@ref) with confidence bounds on magnitude and phase.
"""
bodeconfidence

@userplot BodeConfidence

@recipe function BodeConfidence(p::BodeConfidence; ω = logspace(-2,3,200))
    arxtfm = p.args[1]
    Σ      = p.args[2]
    L      = chol(Hermitian(Σ))
    am, bm = -denpoly(arxtfm)[1].a[2:end], arxtfm.matrix[1].num.a
    wm     = [am; bm]
    na,nb  = length(am), length(bm)
    mc     = 100
    res = map(1:mc) do _
        w             = L'randn(size(L,1)) .+ wm
        a,b           = params2poly(w,na,nb)
        arxtf         = tf(b,a,arxtfm.Ts)
        mag, phase, _ = bode(arxtf, ω)
        mag[:], phase[:]
    end
    magmc      = hcat(getindex.(res,1)...)
    phasemc    = hcat(getindex.(res,2)...)
    mag        = mean(magmc,2)[:]
    phase      = mean(phasemc,2)[:]
    uppermag   = getpercentile(magmc,0.95)
    lowermag   = getpercentile(magmc,0.05)
    upperphase = getpercentile(phasemc,0.95)
    lowerphase = getpercentile(phasemc,0.05)

    layout := (2,1)

    @series begin
        subplot := 1
        title --> "ARX estimate"
        ylabel --> "Magnitude"
        ribbon := (lowermag, uppermag)
        yscale --> :log10
        xscale --> :log10
        alpha --> 0.3
        ω, mag
    end
    @series begin
        subplot := 2
        ribbon := (lowerphase, upperphase)
        ylabel --> "Phase [deg]"
        xlabel --> "Frequency [rad/s]"
        xscale --> :log10
        alpha --> 0.3
        ω, phase
    end
    nothing
end

function getpercentile(mag,p)
    uppermag = mapslices(mag, 2) do magω
        sort(magω)[round(Int,endof(magω)*p)]
    end
end

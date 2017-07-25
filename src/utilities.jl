export toeplitz, toOrthoNormal, flatten, segment, segmentplot, rms, modelfit

rms(x::AbstractVector) = sqrt(mean(x.^2))
sse(x::AbstractVector) = x⋅x

rms(x::AbstractMatrix) = sqrt.(mean(x.^2,2))[:]
sse(x::AbstractMatrix) = sum(x.^2,2)[:]
modelfit(y,yh) = 100 * (1-rms(y.-yh)./rms(y.-mean(y)))
aic(x::AbstractVector,d) = log(sse(x)) + 2d/size(x,2)

function toeplitz{T}(c::Array{T},r::Array{T})
    nc = length(c)
    nr = length(r)
    A = zeros(T, nc, nr)
    A[:,1] = c
    A[1,:] = r
    for i in 2:nr
        A[2:end,i] = A[1:end-1,i-1]
    end
    A
end

function toOrthoNormal(Ti)
    local T = deepcopy(Ti)
    U_,S_,V_ = svd(T[1:3,1:3])
    local R = U_*diagm([1,1,sign(det(U_*V_'))])*V_'
    T[1:3,1:3] = R
    return T
end

function getD(D,T)
    if D == 3
        return sparse(toeplitz([-1; zeros(T-4)],[-1 3 -3 1 zeros(1,T-4)]))
    elseif D == 2
        return sparse(toeplitz([1; zeros(T-3)],[1 -2 1 zeros(1,T-3)]))
    elseif D == 1
        return sparse(toeplitz([-1; zeros(T-2)],[-1 1 zeros(1,T-2)]))
    end
    error("Can not handle your choice of D: $D")
end

function matrices(x,u)
    n,T = size(x)
    T -= 1
    m = size(u,1)
    A = spzeros(T*n, n^2+n*m)
    y = zeros(T*n)
    I = speye(n)
    for i = 1:T
        ii = (i-1)*n+1
        ii2 = ii+n-1
        A[ii:ii2,1:n^2] = kron(I,x[:,i]')
        A[ii:ii2,n^2+1:end] = kron(I,u[:,i]')
        y[ii:ii2] = (x[:,i+1])
    end
    y,A
end
flatten(A) = reshape(A,prod(size(A,1,2)),size(A,3))'
decayfun(iters, reduction) = reduction^(1/iters)

function ABfromk(k,n,m,T)
    At = reshape(k[:,1:n^2]',n,n,T)
    At = permutedims(At, [2,1,3])
    Bt = reshape(k[:,n^2+1:end]',m,n,T)
    Bt = permutedims(Bt, [2,1,3])
    At,Bt
end

"""
    At,Bt = segments2full(parameters,breakpoints,n,m,T)
"""
function segments2full(parameters,breakpoints,n,m,T)
    At,Bt = zeros(n,n,T), zeros(n,m,T)
    i = 1
    for t = 1:T
        i ∈ breakpoints && (i+=1)
        At[:,:,t] = reshape(parameters[i][1:n^2],n,n)'
        Bt[:,:,t] = reshape(parameters[i][n^2+1:end],n,m)'
    end
    At,Bt
end




segment(res) = segment(res...)
segment(model::LTVStateSpaceModel) = segment(model.At,model.Bt)
function segment(At,Bt, args...)
    diffparams = (diff([flatten(At) flatten(Bt)],1)).^2
    # diffparams .-= minimum(diffparams,1)
    # diffparams ./= maximum(diffparams,1)
    activation = sqrt.(sum(diffparams,2)[:])
    activation
end

function segmentplot(activation, state; filterlength=10, doplot=false, kwargs...)
    plot(activation, lab="Activation")
    ds = diff(state)
    ma = mean(activation)
    ds[ds.==0] = NaN
    segments = findpeaks(activation; filterlength=filterlength, doplot=doplot, kwargs...)
    doplot || scatter!(segments,[3ma], lab="Automatic segments", m=(10,:xcross))
    scatter!(2ma*ds, lab="Manual segments", xlabel="Time index", m=(10,:cross))
end

# filterlength=2; minh=5; threshold=-Inf; minw=18; maxw=Inf; doplot=true
# plot(activationf)
# scatter!(peaks,activationf[peaks])

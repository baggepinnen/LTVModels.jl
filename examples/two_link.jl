# This file reproduces figures 2-3 of
# Identification of LTV Dynamical Models with Smooth or Discontinuous Time Evolution by means of Convex Optimization
# Fredrik Bagge Carlson, Anders Robertsson, Rolf Johansson
# https://arxiv.org/abs/1802.09794
# The script requires DynamicMovementPrimitives.jl to be installed

using OrdinaryDiffEq
using LTVModels

using ValueHistories, IterTools, MLDataUtils, OrdinaryDiffEq, Parameters, InteractNext

# Import robot simulator from DynamicMovementPrimitives
include(Pkg.dir("DynamicMovementPrimitives","src","two_link.jl"))
using TwoLink

@with_kw struct TwoLinkSys
    N  = 1000
    n  = 2
    ns = 2n
    h = 0.02
    σ0 = 0
    sind = 1:ns
    uind = ns+1:(ns+n)
    s1ind = (ns+n+1):(ns+n+ns)
end

# Problem parameters and trajectory
n,m,N,h,σ0 = 4,3,200,0.02,0.0
np         = n*(n+m)
sys        = TwoLinkSys(N=N, h=h, σ0 = σ0)#; warn("Added noise")
seed       = 1

ni         = N+2
traj       = connect_points([0 -0.5; 0.9 -0.1],ni)
q          = inverse_kin(traj[1],:up)
p          = forward_kin(q)
qd         = centraldiff(q')'/sys.h
qdd        = centraldiff(qd')'/sys.h
u0         = torque(q,qd,qdd) # Computed torque
u0       .+= 0.1randn(size(u0)) # Random noise on input

# Define callback to simulate the robot hitting a wall
const wall = -0.2
function affect!(i)
    q = i.u[1:2]
    qd = i.u[3:4]
    p1,p2 = forward_kin(q)[2]
    J = TwoLink.jacobian(q)
    v1,v2 = J*qd
    d = [0, wall-p2]
    δq = J\d
    i.u[3:4] = J\[v1, 0]
    i.u[1:2] .+= δq
end
condition(x,t,i) = forward_kin(x)[2][2] > wall
cb = DiscreteCallback(condition,affect!)

# Simulate robot using ODE-solver
t      = 0:h:N*h
q0     = [q[:,1]; qd[:,1]]
prob   = OrdinaryDiffEq.ODEProblem((x,p,t)->time_derivative(x, u0[:,floor(Int,t/h)+1]),q0,(t[[1,end]]...))
sol    = solve(prob,Tsit5(),reltol=1e-8,abstol=1e-8, callback=cb)
x      = hcat(sol(t)...)
u      = u0[:,1:N]
u      = [u; ones(1,size(u,2))]
x,y    = x[:,1:N], x[:,2:N+1]

plot(forward_kin(x[1:2,:])', layout=2, subplot=1, title="End-effector positions")
plot!([1,N],[wall, wall], l=(:dash, :black), grid=false, subplot=1, lab="Wall")
plot!(u', subplot=2, title="Control torques")

# Normalize input data to mean 0 and unit variance
sx = std(x,2)
su = std(u[1:2,:])
mx, mu = mean(x,2), mean(u[1:2,:],2)
xt = (x.-mx) ./ sx
ut = copy(u)
ut[1:2,:] = (ut[1:2,:].-mu) ./ su

# Fit model using ADMM
model = LTVModels.fit_statespace_admm(xt,ut,10^-0.5, extend=false, tol=1e-8, iters=20000, D=1, zeroinit=false)

signch = findfirst(u[2,:] .< 0) # Detect when control signal changes sign
contact = findfirst(forward_kin(x[1:2,:])[2,:] .> wall) # Detect first contact with wall

fig=plot(xt', label="States", show=false, layout=grid(3,2), subplot=1:4, size=(1800,1000), title="State trajectories", title=["\$q_1\$" "\$q_2\$" "\$\\dot{q}_1\$" "\$\\dot{q}_2\$"], c=:blue, linewidth=2)
vline!([signch].*ones(1,4), show=false, subplot=1:4, lab="Velocity sign change", l=(:black, :dash))
vline!([contact].*ones(1,4), show=false, subplot=1:4, lab="Stiff contact", l=(:black, :dash))
plot!(forward_kin(x[1:2,:])', subplot=5, title="End-effector positions", lab=["x" "y"], c=[:blue :red])
hline!([wall], l=(:dash, :black), grid=false, subplot=5, lab="Constraint")
vline!([contact], subplot=5, lab="Stiff contact", l=(:black, :dash))
plot!(ut[1:2,:]', subplot=6, title="Control torques", lab="", c=[:blue :red])
vline!([signch], subplot=6, lab="Velocity sign change", l=(:black, :dash))
plot!(LTVModels.simulate(model, xt[:,1], ut)', label="Simulation", show=false, subplot=1:4, l=(:red,), legend=[false false :left false])
gui()


# Validation data. Simulate system again with different input trajectory
uv   = torque(q,qd,qdd)
uv .+= 0.1randn(size(uv))
prob   = OrdinaryDiffEq.ODEProblem((x,p,t)->time_derivative(x, uv[:,floor(Int,t/h)+1]),q0,(t[[1,end]]...))
sol   = solve(prob,Tsit5(),reltol = 1e-8,abstol = 1e-8, callback = cb)
xv    = hcat(sol(t)...)
uv    = uv[:,1:N]
uv    = [uv; ones(1,size(uv,2))] # Extend to estimate friction
xv,yv = xv[:,1:N], xv[:,2:N+1]

# Normalize data with same parameters as during training
xt2        = (xv.-mx) ./ sx
ut2        = copy(uv)
ut2[1:2,:] = (uv[1:2,:].-mu) ./ su

plot(xt2', label="States", show=false, layout=grid(2,2), subplot=1:4, size=(1800,1000), title="State trajectories", title=["\$q_1\$" "\$q_2\$" "\$\\dot{q}_1\$" "\$\\dot{q}_2\$"], c=:blue, linewidth=2, reuse=false, grid=false)
vline!([signch].*ones(1,4), show=false, subplot=1:4, lab="Original Velocity sign change", l=(:black, :dash))
vline!([contact].*ones(1,4), show=false, subplot=1:4, lab="Original Stiff contact", l=(:black, :dash))

plot!(LTVModels.simulate(model, xt2[:,1], ut2)', label="Simulation", show=false, subplot=1:4, l=(:red,), legend=[false false :left false])
gui()

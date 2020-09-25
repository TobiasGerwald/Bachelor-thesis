# Parameterschätzung und Sensitivitätsanalyse für Ausbreitungsmodelle infektiöser Krankheiten am Beispiel von Covid-19

using DifferentialEquations

## RKI Modell
"
S = Susceptibles
E = Exposed
I = Infectious without symptomes
Q = Infectious with symptomes
H = Hospitalized
Z = Intensive (medical) Care
D = Dead
R = Recovered
C = Comulative number of infections


β₁ transmission rate Infectious
β₂ transmission rate Symptomous
β₃ transmission rate Hospitalized
1/α mean value latent period (in weeks)
1/γ₁ mean value weeks between Infectious - Symptomous
1/γ₂ mean value weeks until Infected recover
1/δ₁ mean value weeks between Symptomous - Hospitalized
1/δ₂ mean value weeks until Symptomous recover
1/ϵ₁ mean value weeks between Hospitalized - Intensive care
1/ϵ₂ mean value weeks until Hospitalized recover
1/ϕ mean value weeks in Intensive care
p₁ percentage of infected who show symptomes
p₂ percentage of people showing symtomes who need to go to hospital
p₃ percentage of people in hospitales who need intensive medical care
p₄ percentage of people recieving intensive medical care who die
"



function RKI_ode(du, u, p, t) #prozentual

    S,E,I,Q,H,Z,D,R,C = u
    β₁,β₂,β₃, α, γ₁,γ₂, δ₁,δ₂, ϵ₁,ϵ₂, ϕ, p₁,p₂,p₃,p₄ = p

    λ = (β₁*I + β₂*Q + β₃*H)/(S+E+I+R)

    du[1] =  - λ*S
    du[2] =    λ*S - α*E
    du[3] =                 α*E - p₁*γ₁*I - (1-p₁)*γ₂*I
    du[4] =                       p₁*γ₁*I               - p₂*δ₁*Q - (1-p₂)*δ₂*Q
    du[5] =                                               p₂*δ₁*Q               - p₃*ϵ₁*H - (1-p₃)*ϵ₂*H
    du[6] =                                                                       p₃*ϵ₁*H               - ϕ*Z
    du[7] =                                                                                               p₄*ϕ*Z
    du[8] =                                 (1-p₁)*γ₂*I           + (1-p₂)*δ₂*Q           + (1-p₃)*ϵ₂*H + (1-p₄)*ϕ*Z

    du[9] =          α*E
end

RKI_parms = [1.4,0.35,0.07, 7/2, 7/3,7/10, 7/4,7/9, 7,7/14, 7/10, 0.6,0.045,0.25,0.5]
RKI_init = [0.99999, 0, 0.00001, 0, 0, 0, 0, 0, 0.00001]
RKI_tspan = (0.0, 100)

RKI_prob = ODEProblem(RKI_ode, RKI_init, RKI_tspan, RKI_parms)
RKI_sol = solve(RKI_prob, RadauIIA5(), dt=0.01)
# RadauIIA5 - An A-B-L stable fully implicit Runge-Kutta method with internal tableau complex basis transform for efficiency

using StatsPlots

plot(RKI_sol,
      title  = "RKI Modellverlauf COVID-19",
      xlabel = "Wochen",
      ylabel = "in %",
      #vars   = [2,3,4,5,6] #um nur einen Teil der Kurven zu plotten
      #yaxis  = :log, #Alle Startwerte müssen größer null gesetzt werden
      color  = [:blue :yellow :red :pink :purple :orange :darkgray :green :black ],
      label  = ["Susceptibles" "Exposed" "Infectious" "Symptomous" "Hospitalized" "Intensive care" "Dead" "Recovered" "Cumulative"] )


using LinearAlgebra

"
Abweichungen jeweils um p
"

function parametersample(parms, p)

    unif = 2 * rand(length(parms) ) .- 1  #rand() erzeugt unif(0,1) -> erhalte 15 unif(-1,1) verteilte Proben
    scaling = Diagonal(p * parms)

    samplePara = scaling * unif + parms

    return samplePara
end


function initsample(init, p)

    unif = 2 * rand() - 1
    scaling = p * init[3]

    sample = unif * scaling + init[3]

    sampleInit = copy(init)

    sampleInit[1]   = 1 - sample  #Summe über alle Compartments muss immer noch 1 ergeben
    sampleInit[3]   = sample
    sampleInit[end] = sample      #Startwert C₀ = I₀

    return sampleInit
end

## Test
function test(testlength,parms,init)

    pts = zeros(testlength, length(parms))
    its = zeros(testlength)

    for i in 1:testlength
        pts[i,:] = parametersample(parms, 0.05)
        its[i]   = initsample(init, 0.05)[3]
    end

    Pl1=violin(pts,
                xlabel = "Parameternummer",
                ylabel = "Paramterintervalle",
                legend = false,
                color  = "blue",
                side   = :right,
                xticks = 1:length(parms),
                yaxis  = :log10)

    Pl2=violin(its,
                ylabel = "Intervall für Anfangsbedingung von I",
                legend = false,
                color  = "blue",
                xticks = false,
                side   = :right )

    boxplot!(Pl2, its)

    plot(Pl1, Pl2,
          layout = grid(1, 2, widths = [0.7, 0.3] ) )

end


test(100000,RKI_parms,RKI_init)

#Teste rechnerisch, ob Intervalle richtig abgebildet werden
Diagonal(0.05 * RKI_parms) * (ones(15) * [-1 0 1] ) + [RKI_parms RKI_parms RKI_parms]
"
Die Abweichung liegt außerhalb der Maschienengenauigkeit und ergibt sich eigentlich ebenfalls zu 0.
"
## Test end

"
parameter/initial condition sensitivity:
tsteps = time steps at which the solution is evaluated
p -> parameters are uniformly drawn from their respective intervals [(1-p)*parms[i],(1+p)*parms[i]];
     I also varies in its respective interval (this also changes S so that the sum still adds up to one)
n = sample size
"

function odesample(ode, init, parms, tspan, tsteps, p, n)

    lt = length(tsteps)

    #Bereite Output vor
    S = zeros(n, lt)  #Speichere einen Verlauf von S pro Zeile
    E = zeros(n, lt)
    I = zeros(n, lt)
    Q = zeros(n, lt)
    H = zeros(n, lt)
    Z = zeros(n, lt)
    D = zeros(n, lt)
    R = zeros(n, lt)
    C = zeros(n, lt)

    for i in 1:n

        parameter = parametersample(parms, p)
        start     = initsample(init, p)

        sol       = solve(ODEProblem(ode, start, tspan, parameter), RadauIIA5(), dt=0.05)

        #Abspeichern
        S[i,:] = sol(tsteps)[1,:] #S Komponenten der i-ten Lösung
        E[i,:] = sol(tsteps)[2,:]
        I[i,:] = sol(tsteps)[3,:]
        Q[i,:] = sol(tsteps)[4,:]
        H[i,:] = sol(tsteps)[5,:]
        Z[i,:] = sol(tsteps)[6,:]
        D[i,:] = sol(tsteps)[7,:]
        R[i,:] = sol(tsteps)[8,:]
        C[i,:] = sol(tsteps)[9,:]
    end

    return S, E, I, Q, H, Z, D, R, C
end

RKI_tsteps = 0:100
data       = odesample(RKI_ode, RKI_init, RKI_parms, RKI_tspan, RKI_tsteps, 0.05, 10000)

using Statistics

"
Quantile in jedem Schritt
"

function DataQuantile(data, q)

    a = size(data)[2]
    values = zeros(5, a)

    for i in 1:a
        values[:,i] = quantile(data[:,i], [0, q, 0.5, 1-q, 1] )
    end

    return values
end

"Maximaler I-Wert: Wie hoch und wann erreicht?"

Imaxvalues = findmax(data[3], dims=2)
Imaxtime   = getindex.(Imaxvalues[2], [1 2] )[:,2]

histogram(Imaxvalues[1],
           legend = false,
           title  = "Maximale I-Werte",
           xlabel = "maximaler I-Wert in %",
           ylabel = "Anzahl" )

histogram(Imaxtime,
           legend = false,
           title  = "",
           xlabel = "Woche, in der der maximale I-Wert erreicht wurde",
           ylabel = "Anzahl" )

"
ploting sensitivity
"

function sensitivityplot(data, tsteps, name, q)

    values = DataQuantile(data, q)

    plot(tsteps, values[5,:],
            color  = "red",
            label  = "Maximalwert",
            title  = "Parameterabhängige Schwankungen in $name",
            xlabel = "Wochen",
            ylabel = "in %" )

    plot!(tsteps, values[3,:],
            color  = "blue",
            label  = "Mittelwert",
            ribbon = (values[3,:] - values[2,:], values[4,:] - values[3,:] ),
            fillalpha = 0.3 )

    plot!(tsteps, values[1,:],
            color = "green",
            label = "Minimalwert" )
end

sensitivityplot(data[3], RKI_tsteps, "I", 0.05)

sensitivityplot(data[9], RKI_tsteps, "C", 0.05)


## SEIR Modell


using DifferentialEquations
using StatsPlots
using LinearAlgebra
using Statistics
using CSV


"
S = Susceptibles
E = Exposed
I = Infectious without symptomes
R = Recovered
C = Comulative


β transmission rate
1/α mean value latent period (in weeks)
1/γ mean value weeks infectious period
"


function seir_ode(du,u,p,t)

    S,E,I,R,C = u
    β,α,γ = p

    du[1] = - β*S*I
    du[2] =   β*S*I - α*E
    du[3] =           α*E - γ*I
    du[4] =                 γ*I

    du[5] =           α*E
end

seir_parms = [1.4, 7/2, 7/6]
seir_init  = [0.99999, 0, 0.00001, 0, 0.00001]
seir_tspan = (0.0, 100)

seir_prob  = ODEProblem(seir_ode, seir_init, seir_tspan, seir_parms)
seir_sol   = solve(seir_prob, RadauIIA5(), dt=0.1)


plot(seir_sol,
      xlabel = "Wochen",
      ylabel = "in %",
      color  = [:blue :orange :red :green :black],
      label  = ["Susceptibles" "Exposed" "Infectious" "Recovered" "Cumulative"] )

test(100000, seir_parms, seir_init)

function odesample2(ode, init, parms, tspan, tsteps, p, n)

    lt = length(tsteps)

    #Bereite Output vor
    S = zeros(n, lt)  #Speichere einen Verlauf von S pro Zeile
    E = zeros(n, lt)
    I = zeros(n, lt)
    R = zeros(n, lt)
    C = zeros(n, lt)

    for i in 1:n

        parameter = parametersample(parms, p)
        start     = initsample(init, p)

        sol       = solve(ODEProblem(ode, start, tspan, parameter), RadauIIA5(), dt=0.05)

        #Abspeichern
        S[i,:] = sol(tsteps)[1,:] #S Komponenten der i-ten Lösung
        E[i,:] = sol(tsteps)[2,:]
        I[i,:] = sol(tsteps)[3,:]
        R[i,:] = sol(tsteps)[4,:]
        C[i,:] = sol(tsteps)[5,:]
    end

    return S, E, I, R, C
end

seir_tsteps = 0:100
data2 = odesample2(seir_ode, seir_init, seir_parms, seir_tspan, seir_tsteps, 0.05, 10000)

sensitivityplot(data2[3], seir_tsteps, "I", 0.05)

sensitivityplot(data2[5], seir_tsteps, "C", 0.05)


df = CSV.File("C:\\Users\\Tobias\\Documents\\RKI_COVID19.csv") #Stand 19.07.2020 ; !!! Speicherort der Datei hier statt mit \ mit doppel \\ angeben !!!
truedata = df.Kummuliert/83200000 #relative Anzahl
model_pred = seir_sol(1.0:25.0)[5,:]

plot(model_pred,
     title  = "Vergleich: Modell und Daten",
     label  = "Modellprediction",
     ylabel = "Erkrankte in %",
     xlabel = "Wochen")
plot!(truedata, label = "Messdaten")


## RWM
using Distributions
using StatsBase


function myRWM(samplesize, θ₀, πp, Σ)

    #initialising and initial values
    α      = zeros(samplesize)
    θ      = zeros(samplesize, length(θ₀))
    θ[1,:] = θ₀

    for i in 2:samplesize

        ω = rand(MvNormal(Σ))
        Φ = θ[i-1,:] + ω #random walk step
        α[i] = min(1, πp(Φ) / πp(θ[i-1,:]))

        if rand() < α[i] #acceptance check
            θ[i,:] = Φ
        else
            θ[i,:] = θ[i-1,:]
        end
    end
    return (θ = θ, α = α)
end

## Test sample μ bei bekannter Covarianz-Matrix
m = [6, 2]

S = [sqrt(2) 0.5;
     0.5 sqrt(0.5)]
Y = rand(MvNormal(m, S), 1000)'
n = size(Y)[1]
Ybar = vec(sum(Y, dims=1)/n)

πptest(θ) = pdf(MvNormal(Ybar, S/n), θ)

#Σtest = 1*Diagonal(ones(2))
mytestsample = myRWM(100000, [3, 3], πptest, S)

plot(mytestsample.θ,
     title = "Traceplot",
     label = false)

density(mytestsample.θ,
        title = "Test-Sample Kernel",
        label = ["μ₁" "μ₂"])

plot(mytestsample.α)
mean(mytestsample.α)

plot(autocor(mytestsample.θ),
     title = "Autocorrelation",
     label = ["μ₁" "μ₂"])
## test end


function θtransform(θ; A = Diagonal([3.5, 3.5, 3.5, 5e-5]), shift = [3.5, 3.5, 3.5, 5e-5] ) #θ in [-1, 1]⁴; β in [0, 7], α in [0, 7], γ in [0, 7], I₀ in [0, 0.0001]

    t =  A * θ .+ shift

end

## test
"teste ob Parameter richtig abgebildet werden"
density( θtransform( rand(Float64, (4, 100000) ) * 2 .- 1)'[:,1:3],
         label = ["β" "α" "γ" "I₀"])
θtransform([-1 0 1; -1 0 1; -1 0 1; -1 0 1]) #Die Ränder und der Mittelpunkt der normierten Intervalle werden genau auf die Ränder und die Mitte der orginalen Parameterintervalle abgebildet
## test end


"posterior"
sqrtΓinv = inv(Diagonal(sqrt.(0.05*truedata) ) ) # Γ^(-1/2)
function πp(θnormiert)

    if (θnormiert .>= -1) == (θnormiert .<= 1) #Überprüfe ob die Parameter in ihrem Intervall liegen

        θ = θtransform(θnormiert)
        d = exp(-0.5 * norm(sqrtΓinv * (solve(ODEProblem(seir_ode, [1 - θ[4], 0, θ[4], 0, θ[4]], (1.0,173.0), θ[1:3]), maxiters = 1e10 )(1:7:173)[5,:] - truedata) )^2 )

    else
        d = 0
    end

    return d

end


"Step matrix für normalisierte Parameter"
Σ = 0.1 * Diagonal( ones(4) ) # mittlere Akzeptanzrate ≈ 30%


"Initial value for RWM"
θ₀ = [-0.9, 0, 0, 0] #normalized initial parameters


mysample = myRWM(100000, θ₀, πp, Σ)
mytransformedsample = θtransform(mysample.θ')'


mean(mysample.α)

plot(autocor(mysample.θ, 0:150),
     title = "Autokorrelationen",
     label = ["β" "α" "γ" "I₀"],
     color = [:blue :red :green :purple])


pt1 = plot(mytransformedsample[:,1],
            title  = "Traceplot β",
            color  = :blue)

pt2 = plot(mytransformedsample[:,2],
            title  = "Traceplot α",
            color  = :red)

pt3 = plot(mytransformedsample[:,3],
            title  = "Traceplot γ",
            color  = :green)

pt4 = plot(mytransformedsample[:,4],
            title  = "Traceplot I₀",
            color  = :purple)

plot(pt1,pt2,pt3,pt4,
     #xticks = 0:2e4:1e5,
     xlims  = (5000,6000),
     legend = false)


d1 = histogram(mytransformedsample[10000:150:end, 1],
             color = :blue,
             bins  = 16)

d2 = histogram(mytransformedsample[10000:150:end, 2],
             color = :red,
             bins  = 16)

d3 = histogram(mytransformedsample[10000:150:end, 3],
             color = :green,
             bins  = 16)

d4 = histogram(mytransformedsample[10000:150:end, 4],
             color = :purple,
             bins  = 16)

plot(d1,d2,d3,d4,
     title  = ["Stichprobenverteilung β" "Stichprobenverteilung α" "Stichprobenverteilung γ" "Stichprobenverteilung I₀"],
     legend = false)


#corrplot(mytransformedsample[10000:150:end, :])

c1 = scatter(mytransformedsample[10000:150:end, 1],mytransformedsample[10000:150:end, 2],
             legend = false,
             title  = "Korrelation β-α",
             xlabel = "β",
             ylabel = "α",
             markersize = 2)

c2 = scatter(mytransformedsample[10000:150:end, 1],mytransformedsample[10000:150:end, 3],
             legend = false,
             title  = "Korrelation β-γ",
             xlabel = "β",
             ylabel = "γ",
             markersize = 2)

c3 = scatter(mytransformedsample[10000:150:end, 1],mytransformedsample[10000:150:end, 4],
             legend = false,
             title  = "Korrelation β-I₀",
             xlabel = "β",
             ylabel = "I₀",
             markersize = 2)

c4 = scatter(mytransformedsample[10000:150:end, 2],mytransformedsample[10000:150:end, 3],
             legend = false,
             title  = "Korrelation α-γ",
             xlabel = "α",
             ylabel = "γ",
             markersize = 2)

c5 = scatter(mytransformedsample[10000:150:end, 2],mytransformedsample[10000:150:end, 4],
             legend = false,
             title  = "Korrelation α-I₀",
             xlabel = "α",
             ylabel = "I₀",
             markersize = 2)

c6 = scatter(mytransformedsample[10000:150:end, 3],mytransformedsample[10000:150:end, 4],
             legend = false,
             title  = "Korrelation γ-I₀",
             xlabel = "γ",
             ylabel = "I₀",
             markersize = 2)

plot(c1,c2,c3,c4,c5,c6)

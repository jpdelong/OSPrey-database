# this script processes the link between abundances of mammal prey and their prevalence in barn owl diets globally

using CSV
using DataFrames
using GLMakie
using Statistics
using GLM
using MixedModels
using StatsBase
using FreqTables
using ColorSchemes
using CategoricalArrays
using DifferentialEquations

pwd() # check and confirm you are in the directory
dir = "C:/Users/93652672/OneDrive - University of Nebraska/Documents/GitHub/OSPrey-database/Diversity stability relationship in barn owls/"

cd(dir)

include("BAOW_model.jl")

# to create the environment, in the pkg repl (]), type generate BarnOwlGlobalDiet
# to activate the environment, in the pkg repl (]), type activate BarnOwlGlobalDiet

colorgrad = cgrad(:roma, 10, categorical = true, scale = :exp)

# =============================================================================
############# step 1, determine the body mass - density scaling relationship
# =============================================================================
df_density = CSV.read("Dataset_density2.csv",DataFrame) # read in density data
df_density = df_density[findall(df_density.Order .!= "NA"),:] # drop everything that is rows with just NA

# open up a vector to hold the body masses
number_rows = size(df_density,1)
masses = zeros(Float64,number_rows)

# read in the body masses to pair with the densities
df_bodymass = CSV.read("Dataset_predictions.csv",DataFrame) # read in density data

# cycle through the density dataset and pass in the mammal body mass
for i = 1:number_rows
    row_species = df_density.AcceptedName_COL[i]
    mass_to_fill = df_bodymass.BM[findall(df_bodymass.Species .== row_species)]
    if length(mass_to_fill) > 1
        mass_to_fill = mean(mass_to_fill)
    end
    if !isempty(mass_to_fill)
        masses[i] = mass_to_fill[]
    end
end

df_density[!, :BodyMass] .= masses
df_density = df_density[findall(df_density.BodyMass .!= 0),:] # drop rows with 0 body mass
df_density[!, :LogDensity] = log10.(df_density.Density_km) # log10 tranform the density
df_density[!, :LogMass] = log10.(df_density.BodyMass) # log10 transform the body mass

# model 1 - allometry with orders
density_model_2 = lm(@formula(LogDensity ~ LogMass * Order), df_density)
#lm_2 = lm(@formula(LogDensity ~ LogMass + Order), df_density)
mainr2 = r2(density_model_2)

# =============================================================================
###### step 2, read in the barn owl diets
# =============================================================================

df_baow = CSV.read("BarnOwlDiets_20.csv",DataFrame) # read in diet data
    deleteat!(df_baow, df_baow.DataSet .== 2255) # there's one dataset (#2255) that was done without individuals

    unique(df_baow.DataSet)

    # reduce to only rows with prey identified to species or genus
        df_baow = filter(row -> row.IDLevel == "Species" || row.IDLevel == "Genus", df_baow)

    # drop everything but mammals
        df_baow = df_baow[occursin.("Mammalia", df_baow.PreyClass), :] # reduce the data set to only mammal prey

    # drop rows with 0 body mass ======= NEED TO COME BACK AND FIND THESE MASSES
        df_baow = df_baow[findall(df_baow.PreyMass .!= 0),:] # drop rows with 0 body mass

    # drop rows with Afrosoricida - no density estimates for this order
        df_baow = df_baow[findall(df_baow.PreyOrder .!= "Afrosoricida"),:] # drop rows with 0 body mass

df_meta = CSV.read("BarnOwlMeta_20.csv",DataFrame) # read in diet data
    deleteat!(df_meta, df_meta.DataSet .== 2255) # there's one dataset (#2255) that was done without individuals
    unique(df_meta.DataSet)

# going to work with only the datasets that have prey per pellet data
# we can directly eliminate rows withoout ppp in df_meta    
df_meta_ppp = dropmissing(df_meta,:PreyPerPellet)

    howmanystudies = length(unique(df_meta_ppp.DataSet))

# in df_baow, we have to identify the datasets by id to drop them
idx_to_drop = ismissing.(df_meta.PreyPerPellet) # which dataset ids have no ppp
datasets_to_keep = df_meta.DataSet[findall(idx_to_drop .== 0)]

df_baow_ppp = DataFrame[]
for i = 1:length(datasets_to_keep)
    push!(df_baow_ppp,df_baow[df_baow.DataSet .== datasets_to_keep[i],:]) # keep the ones that aren't missing
end

# add the log10 of prey mass
    df_baow[!, :LogMass] = log10.(df_baow.PreyMass) 

# Need to convert the "orders" column to categorical type
    df_baow.group = categorical(df_baow.PreyOrder)
    # Then create a new column of the underlying integer codes
    df_baow.group_idx = levelcode.(df_baow.group)

# open an empty vector to hold prey number
numb_prey = zeros(Float64, howmanystudies)
# open an empty vector to hold prey richness
numb_prey_spp = zeros(Float64, howmanystudies)

# open an empty vector to hold some other information
baow_a_store = zeros(Float64, howmanystudies)
NE_rate = Array{Union{Float64,Missing}}(missing, howmanystudies)
sum_R = zeros(Float64, howmanystudies)

# ==========================================================================================
###### step 4, # cycle through the datasets to generate 
# ==========================================================================================

baow_h = 0.02 # choose this if you want to run the main analysis

diet_datasets = DataFrame[]
baow_a_all = []

for i = 1:howmanystudies # in this loop, run through all the different diets
    println(i)
    j = df_meta_ppp.DataSet[i]
    rare_richness = df_meta_ppp.DietRichness_rare[i]
    df_diet = df_baow[findall(df_baow.DataSet .== [j]),:]

    ################## check for the presence of prey identified only to genus
    # run through to find out if these should be allocated to prey identified to species
    # first check what rows have prey identified to genus
    test_genus = df_diet[findall(df_diet.IDLevel .== "Genus"),:] # drop rows with 0 body mass
    if size(test_genus,1) > 0 # only do this if there are prey id'd to genus
        genus_to_check = test_genus.PreyScientificName # pull out the genus sp names
        genus_split = split.(genus_to_check, " ") # split out the genus name from sp
        for zzz = 1:size(genus_split,1) # run a loop to check each genus
            temp_diet = test_genus # re-assign
            number_to_distribute = temp_diet.CountInt[zzz,:][] # how many prey to re-distribute
            subset_to_genus = df_diet[occursin.(genus_split[zzz][1], df_diet.PreyScientificName),:] # pull prey rows for that genus
            subset_to_genus = subset_to_genus[findall(subset_to_genus.PreyScientificName .!= genus_to_check[zzz]),:] # drop the genus id'd row
            to_add = round.(Int64, number_to_distribute .* subset_to_genus.CountInt ./ sum(subset_to_genus.CountInt)) # create vector to distribute
            if ~isempty(to_add)
                subset_to_genus.CountInt = subset_to_genus.CountInt .+ to_add # add the 
                to_drop = findall(occursin.(genus_split[zzz][1], df_diet.PreyScientificName)) # find original rows for genus
                deleteat!(df_diet, to_drop) # pull those original rows
                df_diet = vcat(subset_to_genus,df_diet) # re-assemble df_diet
            end
        end
    end
    ################## end of re-distribution block

    if sum(df_diet.CountInt) > 0
        numb_diet = df_diet.CountInt
        mass_diet = df_diet.LogMass
        order_diet = df_diet.PreyOrder
        new_x = DataFrame()
        new_x[!, :LogMass] = df_diet.LogMass
        new_x[!, :Order] = order_diet

        # here we need to predict the density from the mass and the order
        df_predict_y = predict(density_model_2,new_x)

        # add the mean predicted density to the data frame
        df_diet[!, :LogDensity] = df_predict_y
        df_diet[!, :Density] = 10 .^ (df_diet.LogDensity) # convert to non-log

        # calculate frequencies of diet items
        df_diet[!, :Frequencies] = df_diet.CountInt./ sum(df_diet.CountInt)

        # add the current indexed diet to the vector of diets
        push!(diet_datasets, df_diet)

        # tally number of prey species and individuals
        numb_prey[i] = sum(df_diet.CountInt)
        numb_prey_spp[i] = size(df_diet,1)

        # calculate the a for the diet
        mean_ppp = mean(skipmissing(df_meta_ppp.PreyPerPellet))
        sum_R[i] = sum(df_diet.Density)
        p3 = df_meta_ppp.PreyPerPellet[i]
        if ismissing(p3)
            baow_a = -mean_ppp / (mean_ppp * sum_R[i] * (baow_h - 1))
        else
            baow_a = -p3 / (p3 * sum_R[i] * (baow_h - 1))
            baow_a_store[i] = baow_a
        end

        # get the a for each prey type in the database
        baow_a_by_prey = -(p3 .* df_diet.Frequencies) ./ ((p3 .* df_diet.Frequencies) .* df_diet.Density .* (baow_h - 1))
        push!(baow_a_all,baow_a_by_prey) # push it to the array

    end
end

# ===================================================
# put all the processed data into the meta DataFrame
# ===================================================
df_meta_ppp[!, :numb_prey] = numb_prey
df_meta_ppp[!, :numb_prey_spp] = numb_prey_spp
df_meta_ppp[!, :baow_a] = baow_a_store
df_meta_ppp[!, :sum_R] = sum_R

#df_meta_no_ppp = df_meta_no_ppp[findall(df_meta_no_ppp.Num_obs .> 20),:] # drop rows with 0 body mass

# ==========================================================================================
###### step 4, # cycle through the datasets to generate ranks, slopes, and diversity metrics
# ==========================================================================================

all_solutions = Matrix{Float64}[]
t_solution = []

include("BAOW_model.jl")

for i = 1:howmanystudies # in this loop, run through all the different diets
    println(i)

    df_diet = diet_datasets[i]
    rare_richness = Int(round(df_meta_ppp.DietRichness_rare[i], digits = 0))
    if rare_richness > size(df_diet,1)
        rare_richness = size(df_diet,1)
    end

    df_diet = df_diet[sample(1:nrow(df_diet), rare_richness, replace=false), :]

    if sum(df_diet.CountInt) > 20

        # tally number of prey species and individuals
        numb_prey[i] = sum(df_diet.CountInt)
        numb_prey_spp[i] = size(df_diet,1)

            # ------------------------------------------
            # now set up and call the ODE
            # ------------------------------------------
            # data drawn from the dataset
            # only do datasets that had prey per pellet data

            u0 = zeros(Float64,13)
            u0[1:size(df_diet,1)] = df_diet.Density
            u0[12] = 0.56

            masses = zeros(Float64,12)
            masses[1:size(df_diet,1)] = df_diet.PreyMass 

            densities = zeros(Float64,12)
            densities[1:size(df_diet,1)] = df_diet.Density

            r_max = 0.025 .* masses .^ -0.26 # allometry from Blueweiss 1978 Oecologia
            r_max[isinf.(r_max)] .= 0

            K = 10 .* densities
            K[K .== 0] .= mean(K)
            # basic model setup
            tspan = [0 3650]

            # make owl mortality allometric
            owl_mass = 550
            fraction_dry_mass = 0.3
            d_owl = exp(-2.37)/365 .* (fraction_dry_mass * owl_mass) .^ -0.2 # allometric function from McCoy and Gillooly 2008 ECO LETT

            # using another approach
            # average lifespan of barn owl is about four years
            lifespan = 4*365
            d_owl = 1/lifespan # this value seems a little more realistic than the allometric

            # other owl-specific parameters
            GGE = 0.03
            h_owl = 0.02

                struct to_pass
                    d_owl::Float64
                    GGE::Float64
                    a_owl::Float64
                    h_owl::Float64
                    masses::Vector{Float64}
                    r_max::Vector{Float64}
                    K::Vector{Float64}
                end

            p = to_pass(d_owl, GGE, baow_a, h_owl, masses, r_max, K)

            prob = ODEProblem(BAOW_model, u0, tspan, p)
            sol_p = solve(prob, Tsit5(); saveat=0.1)

            push!(all_solutions, Array(sol_p))

    end
    t_solution = sol_p.t
end


WhichDataSet = 313

fig = Figure(size = (1000,700))
    ax1 = Axis(fig[1,1], title = "Prey")
    ax2 = Axis(fig[1,2],title = "Barn Owls")

for i in 1:11
    lines!(ax1, t_solution, all_solutions[WhichDataSet][i,:])
end

lines!(ax2, sol_p.t, sol_p[12,:], label="Owls", color=:black)
fig


axislegend(ax1)







# drop every row below the 20 prey cut-off
df_meta2 = df_meta[findall(df_meta.Num_obs .> 20),:] # drop rows with 0 body mass

# drop every row where the prey count was 0 (occur when there are few mammals in the diet)
df_meta2 = df_meta2[findall(df_meta2.numb_prey .> 0),:] # drop rows with 0 body mass

# ====================================================================
# save the data set as a csv
CSV.write("BarnOwlProcessed_20.csv", df_meta2)

# reload it if you are just doing StatsBase
df_meta2 = CSV.read("BarnOwlProcessed_20.csv",DataFrame) # read in density data

# ====================================================================
# ====================================================================



# ====================================================================
# make the diversity-scr figure
# ====================================================================

# open up a figure, set up diet-diversity panel
f_diet_div = Figure()
    ax_diet_div = Axis(f_diet_div[1, 1], xlabel = "Diet richness", ylabel = "Diet entropy")
    scatter!(df_meta2.DietRichness_rare, df_meta2.ShannEnt_rare, color = :black, alpha=0.1)

    df_meta2_no_ppp = dropmissing(df_meta2,:PreyPerPellet)

    scatter!(df_meta2_no_ppp.DietRichness_rare, df_meta2_no_ppp.ShannEnt_rare;
    color = df_meta2_no_ppp.baow_a,
    colormap = :viridis)

    Colorbar(f_diet_div[1,2], limits=extrema(df_meta2_no_ppp.baow_a), colormap=:viridis)

    # plot the maximum entropy onto the palette
    ext = round.(extrema(df_meta2_no_ppp.DietRichness_rare), digits = 0)
    richness_list = collect(ext[1]:ext[2])
    maxentropy = zeros(Float64,length(richness_list))

    for i = 1:length(richness_list)
        freqs = fill(20/richness_list[i],i)./20
        maxentropy[i] = -sum(freqs .* log.(freqs))
    end

    lines!(ax_diet_div, richness_list, maxentropy, color = :black)

# set up a panel for richness-scr
    ax_diet_scr = Axis(f_diet_div[1, 3], xlabel = "Diet richness", ylabel = "Space clearance rate")
    
    df_meta_scr = df_meta2[findall(df_meta2.baow_a .!= 0),:] # drop rows with 0 body mass

    
    scatter!(df_meta_scr.DietRichness_rare, df_meta_scr.baow_a, color = :black, alpha=0.3)

    save("SCR_diversity_relationship.pdf",f_diet_div)

# ====================================================================
# test for effects on scr
# ====================================================================

# do single effects
density_model1 = lm(@formula(baow_a ~ numb_prey_spp), df_meta_scr)
density_model1 = lm(@formula(baow_a ~ RERichness), df_meta_scr)
density_model1 = lm(@formula(baow_a ~ RaptorRichness), df_meta_scr)
density_model1 = lm(@formula(baow_a ~ ecozone), df_meta_scr)
density_model1 = lm(@formula(baow_a ~ ShannEnt_rare), df_meta_scr)
density_model1 = lm(@formula(baow_a ~ obs_largest_prey_freq), df_meta_scr)
density_model1 = lm(@formula(baow_a ~ FreqBirds), df_meta_scr)


density_model1 = lm(@formula(baow_a ~ numb_prey_spp * RaptorRichness * obs_largest_prey_freq), df_meta2)
r2(density_model1)

density_model1 = lm(@formula(baow_a ~ ShannEnt_rare + RERichness + obs_largest_prey_freq), df_meta2)
r2(density_model1)

density_model1 = lm(@formula(ShannEnt_rare ~ numb_prey_spp), df_meta2)

# ====================================================================
# ====================================================================


u0 = []
tspan = [0 10]

solve_BAOW_ODE(u0,tspan,masses,a_owl)





# ====================================================================
# ====================================================================





# save the candidate h and proportion data from the iterative run
dataout = DataFrame(cand_h = candidate_h,
    prop_meeting_rule = prop_prey_meeting_rule)
varied_h_results = CSV.write("Vary_h_results.csv", dataout)

# if you want to read back in the varied h results
varied_h_results = CSV.read("Vary_h_results.csv",DataFrame) # read in density data
# ====================================================================

# ===============================================================
# estimate the lower bound of the handling time
# ===============================================================
h_lower_bound = 1/mean(skipmissing(df_meta.PreyPerPellet))/3

# ===============================================================
# coming back to the net energy gain figure - main OFT figure
# ===============================================================
f_net_energy
    text!(ax_net,(770,11.8), text = "A")
    prop_prey_meeting_rule = num_NE_diffs_over_zero / num_NE_diffs_tested
save("Net_energy_return_of_prey_log.png",f_net_energy)
using CairoMakie
save("Net_energy_return_of_prey_log.pdf",f_net_energy)

# ===================================================

# ====================================================================
# Plot h against the proportion of diet matching the rule - Figure 2
# only works if you have run the whole vector of candidate h's
# ====================================================================

#f_h_by_OF = Figure()
    ax_h = Axis(f_net_energy[1, 2], xlabel = "Handling time (days)", ylabel = "Proportion meeting rule")
    xlims!(ax_h,[0,0.2])
    scatter!(ax_h,varied_h_results.cand_h, varied_h_results.prop_meeting_rule, color = :black)
    lines!(ax_h,varied_h_results.cand_h, varied_h_results.prop_meeting_rule, color = :black, alpha=0.4)
    lines!(ax_h,[0.02,0.02],[0,1], color = colorgrad[8], linewidth = 4)
        text!(ax_h,(0.02+0.001,0.25), text = "Estimated handling time")
        text!(ax_h,(0.185,0.98), text = "B")
    save("VariationInH.png",f_h_by_OF)

# ====================================================================
# Plot the body mass distribution of prey at the species level
# ====================================================================
df_mass_by_species = combine(groupby(df_baow, :PreyScientificName), :PreyMass .=> mean)

f_miss_dist = Figure()
    ax_mass_dist = Axis(f_miss_dist[1, 1], xlabel = "Body mass (g)", ylabel = "Frequency")
    hist!(log10.(df_mass_by_species.PreyMass_mean), color = :white, strokewidth = 1, strokecolor = colorgrad[8])
# ====================================================================

# ===================================================================
# coming back to the rank difference against prey body mass figure # not using this figure
# ===================================================================
# plot a horizontal line at 0
lines!(ax_mass_rank,2:4e3,0:0,color = :black, linewidth=2)
    save("Body_size_relative_freqs.png",f_mass_rank)
# ===================================================

# ===============================================================
# coming back to the density and mass rank correlations figure
# ===============================================================

f_ranks
    # some regressions don't have enough points to work
    df_tests = dropmissing(df_meta2,:dens_slopes)

    length(df_tests.dens_slopes[findall(df_tests.dens_slopes .> 0.0)]) / size(df_tests,1)

    # draw a 1:1 line overtop
    lines!(ax_dens,1:20,1:20,color = :black, linewidth=2)
        text!(ax_dens,(20,19), text = "1:1")

    hist!(ax_hist_dens,df_tests.dens_slopes, color = colorgrad[8], strokewidth = 1, strokecolor = :white)
        lines!(ax_hist_dens,0:0,0:100,color = :black, linewidth=2, linestyle = :dash)
        text!(ax_hist_dens,(-1,100), text = "B")
        
        text!(ax_dens,(2,19.5), text = "A")

        #using CairoMakie

        save("Density_count_correlations.pdf",f_ranks)

# ===================================================
# see if species diversity influences diet diversity
# ===================================================

df_meta2[!, :l_re] = log10.(df_meta2.RERichness) # add logged variable to dataframe
df_meta2[!, :l_ns] = log10.(df_meta2.numb_prey_spp) # add logged variable to dataframe

# there are some zeros, so drop these out
df_meta_subset = df_meta2

# run the model
model_richness = fit(MixedModel, @formula(l_ns ~ l_re + (1|ecozone)), df_meta_subset)

# get predictions
xgrid = range(minimum(df_meta_subset.l_re), maximum(df_meta_subset.l_re), length=10) # new range of x's
new_df = DataFrame(l_ns = 0.0,l_re = xgrid,ecozone = first(df_meta_subset.ecozone)) # construct dataframe of predictors

# do the prediction
yhat = predict(model_richness, new_df; new_re_levels=:population)

# have to manually get the confidence band on the regression
V = vcov(model_richness)
X = modelmatrix(model_richness.formula, new_df)
se = sqrt.(sum((X[1] * V) .* X[1], dims=2))

lower = yhat .- 1.96 .* se
upper = yhat .+ 1.96 .* se

# plot local diversity against diet diversity with a linear regression
diversity_test = Figure()
    ax_div = Axis(diversity_test[1, 1], xlabel = "log10 Richness of rodents and true insectivores", ylabel = "log10 Species richness in diet")
    a1 = scatter!(ax_div, df_meta_subset.l_re, df_meta_subset.l_ns, color = colorgrad[8], alpha=0.4)

    band!(ax_div,new_x.l_re,lower[:],upper[:], color = (:gray, 0.2))
    rod = lines!(ax_div,new_x.l_re, yhat, linewidth = 2, color = :black)

    save("Prey_richness.png",diversity_test)

# ===================================================
# try to explain variation in density slopes
# ===================================================

# first just checking if different major areas around the world differ
model_ecozone = lm(@formula(dens_slopes ~ ecozone), df_meta_subset)
# some diffs by ecozone, so we'll use ecozone as a random effect

# start with all terms, backwards select
model_1 = fit(MixedModel, @formula(dens_slopes ~ numb_prey_spp + RERichness + RaptorRichness + FreqBirds + (1|ecozone)), df_meta_subset)
# just confirm that freq birds doesn't have an effect on its own
model_FreqBirds = fit(MixedModel, @formula(dens_slopes ~ FreqBirds + (1|ecozone)), df_meta_subset)

# do the model without freq_birds
model_2 = fit(MixedModel, @formula(dens_slopes ~ numb_prey_spp + RERichness + RaptorRichness + (1|ecozone)), df_meta_subset)
# all terms significant

# check for significant interactions - nothing to see here
model_3 = fit(MixedModel, @formula(dens_slopes ~ numb_prey_spp * RERichness * RaptorRichness + (1|ecozone)), df_meta_subset)

# test NE intake separately because this reduces the sample size to 330
model_NE_intake = fit(MixedModel, @formula(dens_slopes ~ NE_intake + (1|ecozone)), df_meta_subset)

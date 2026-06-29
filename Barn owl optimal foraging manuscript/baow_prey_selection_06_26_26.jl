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

pwd() # check and confirm you are in the directory
#dir = "C:/Users/jdelong2/OneDrive - University of Nebraska/Projects in progress/Barn_owl_global_diet/"
dir = "C:/Users/johnp/Documents/GitHub/OSPrey-database/Barn owl optimal foraging manuscript"
cd(dir)

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

# model 1 - straight all species allometry
lm_1 = lm(@formula(LogDensity ~ LogMass), df_density)
# model 2 - allometry with orders
density_model_2 = lm(@formula(LogDensity ~ LogMass * Order), df_density)
#lm_2 = lm(@formula(LogDensity ~ LogMass + Order), df_density)
mainr2 = r2(density_model_2)

# =============================================================================
###### step 2, read in the barn owl diets
# =============================================================================

df_baow = CSV.read("BarnOwlDiets.csv",DataFrame) # read in diet data
df_meta = CSV.read("BarnOwlMeta.csv",DataFrame) # read in diet data
    deleteat!(df_meta, 529) # there's one dataset (#2255) that was done without individuals
    deleteat!(df_baow, df_baow.DataSet .== 2255) # there's one dataset (#2255) that was done without individuals

minimum(df_meta.Num_obs)
maximum(df_meta.Num_obs)

# grab some descriptive information
    datasets = unique(df_baow.DataSet) # need this for the loop below
    preycats = unique(df_baow.PreyClass)
    howmanystudies = length(unique(df_baow.DataSet))
    total_prey = sum(skipmissing(df_baow.CountInt))
    datasets_used = sum(df_meta.Num_obs .> 20)

# summarize - are mammals the key thing?
    sums_by_category = combine(groupby(df_baow, :PreyClass), :CountInt => sum => :TotalSum)
    prop_mammal = sums_by_category.TotalSum[1] / sum(skipmissing(sums_by_category.TotalSum[:]))
    prop_bird = sums_by_category.TotalSum[2] / sum(skipmissing(sums_by_category.TotalSum[:]))

# before dropping the birds, cycle through datasets to determine their rarefied frequency
    FreqBirds = zeros(Float64, howmanystudies) # open empty vector of zeros
    for i = 1:howmanystudies # run through the diets
        j = df_meta.DataSet[i] # pick out the barn owl ones
        df_diet = df_baow[findall(df_baow.DataSet .== [j]),:] # pull just the current diet
        BirdRows = df_diet[findall(skipmissing(df_diet.PreyClass .== "Aves")),:] # subset the rows with class = Aves
        if !isempty(BirdRows) # check if there are any rows with birds
            FreqBirds[i] = sum(BirdRows.CountInt) / sum(df_diet.CountInt) # if so, calculate the fraction birds
        end
    end

# reduce to only rows with prey identified to species or genus
    df_baow = filter(row -> row.IDLevel == "Species" || row.IDLevel == "Genus", df_baow)
    howmanyspecies = length(unique(df_baow.PreyScientificName))

# drop everything but mammals
    df_baow = df_baow[occursin.("Mammalia", df_baow.PreyClass), :] # reduce the data set to only mammal prey
    howmanyspecies = length(unique(df_baow.PreyScientificName))

# summarize by mammal order
    sums_by_order = combine(groupby(df_baow, :PreyOrder), :CountInt => sum => :TotalSum)
    prop_rodent = sums_by_order.TotalSum[2] / sum(sums_by_order.TotalSum[:])
    which_orders = sums_by_order.PreyOrder

# drop rows with 0 body mass ======= NEED TO COME BACK AND FIND THESE MASSES
    df_baow = df_baow[findall(df_baow.PreyMass .!= 0),:] # drop rows with 0 body mass

# drop rows with Afrosoricida - no density estimates for this order
    df_baow = df_baow[findall(df_baow.PreyOrder .!= "Afrosoricida"),:] # drop rows with 0 body mass

# add the log10 of prey mass
    df_baow[!, :LogMass] = log10.(df_baow.PreyMass) 

# summarize by mammal prey family
    sums_by_family = combine(groupby(df_baow, :PreyFamily), :CountInt => sum => :TotalSum)

# Need to convert the "orders" column to categorical type
    df_baow.group = categorical(df_baow.PreyOrder)
    # Then create a new column of the underlying integer codes
    df_baow.group_idx = levelcode.(df_baow.group)

# =============================================================================
# open up a new figure to plot corellations among ranks
# will come back to this after the loop
# =============================================================================
f_ranks = Figure()
    ax_dens = Axis(f_ranks[1, 1], xlabel = "Rank of density", ylabel = "Rank in diet")
                #xlims!(ax_dens,0,25) ylims!(ax_dens,0,25)
    ax_hist_dens = Axis(f_ranks[1, 2], xlabel = "Slope of line", ylabel = "Frequency")
# =============================================================================

# ===================================================
# plot average net energy gain against the prey types
# will come back to this after the loop
# ===================================================
# this shows whether prey should be included in the diets
f_net_energy = Figure()
    ax_net = Axis(f_net_energy[1, 1], xlabel = "Average daily mass intake rate", ylabel = "Mass/h for prey types")#, yscale = log10)
# ===================================================

# ===================================================
# plot rank difference against prey body mass
# ===================================================
# this shows whether prey should be included in the diets
f_mass_rank = Figure()
    ax_mass_rank = Axis(f_mass_rank[1, 1], xlabel = "Prey body mass (g)", ylabel = "Difference in frequency", xscale = log10)
# ===================================================

# =============================================================================
###### step 3, open up a bunch of storage containers
# =============================================================================
# open empty vectors to store slopes and pvals of regressions
dens_slopes = Array{Union{Float64,Missing}}(missing, howmanystudies)
pval_dens = Array{Union{Float64,Missing}}(missing, howmanystudies)

# open an empty vector to hold prey number
numb_prey = zeros(Float64, howmanystudies)
# open an empty vector to hold prey richness
numb_prey_spp = zeros(Float64, howmanystudies)

# open an empty vector to hold some other information
baow_a_store = zeros(Float64, howmanystudies)
number_prey_meeting_rule = zeros(Float64, howmanystudies)
number_prey_testing_rule = zeros(Float64, howmanystudies)
observed_largest_prey_freq = zeros(Float64, howmanystudies)
NE_rate = Array{Union{Float64,Missing}}(missing, howmanystudies)
sum_R = zeros(Float64, howmanystudies)
NumPositiveInStudy = zeros(Float64, howmanystudies)
family_matrix = zeros(Float64, howmanystudies,size(sums_by_family,1))

# ==========================================================================================
###### step 4, # cycle through the datasets to generate ranks, slopes, and diversity metrics
# ==========================================================================================

# read in the posterior sample for the mammal density allometry
# will eventually delete this
#df_chains = CSV.read("Density_allometry_chain.csv", DataFrame)

# initiate some reference states
num_NE_diffs_over_zero = 0
num_NE_diffs_tested = 0

# initiate an empty vector to store differences
OptForTests = zeros(Float64, 1)

candidate_h = [0.01, 0.02, 0.04, 0.08, 0.16]
candidate_h = [0.02]

prop_prey_meeting_rule = zeros(Float64, length(candidate_h))

for gg = 1:length(candidate_h) # in this loop, run through different guess for h
    baow_h = candidate_h[gg] # pick an h from the candidate vector
    num_NE_diffs_over_zero = 0
    num_NE_diffs_tested = 0

    for i = 1:howmanystudies # in this loop, run through all the different diets
        j = df_meta.DataSet[i]
        df_diet = DataFrame()
        new_x = DataFrame()
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

        if sum(df_diet.CountInt) > 20
        #if !isempty(df_diet)
            numb_diet = df_diet.CountInt
            mass_diet = df_diet.LogMass
            order_diet = df_diet.PreyOrder
            new_x[!, :LogMass] = df_diet.LogMass
            new_x[!, :Order] = order_diet

            # here we need to predict the density from the mass and the order
            df_predict_y = predict(density_model_2,new_x)

            # add the mean predicted density to the data frame
            df_diet[!, :LogDensity] = df_predict_y
            df_diet[!, :Density] = exp.(df_diet.LogDensity) # convert to non-log

            #####################################################################################
            # calculate frequencies of diet items
            df_diet[!, :Frequencies] = df_diet.CountInt./ sum(df_diet.CountInt)
            df_diet[!, :Rel_density] = df_diet.Density./ sum(df_diet.Density)

            # here find the biggest prey and predict its frequency
            row_biggest = findall(df_diet.PreyMass .== maximum(df_diet.PreyMass))
            observed_largest_prey_freq[i] = df_diet.Frequencies[row_biggest][]

            # calculate the rank of prey by density, highest to lowest
            df_rank_density = transform!(sort(df_diet, :Density), :Density, :Density => eachindex => :DensRank)
            # then calculate the rank of prey by count, highest to lowest
            df_ranked = transform!(sort(df_rank_density, :CountInt), :CountInt, :CountInt => eachindex => :CountRank)
            # then calculate the rank of prey by body mass, highest to lowest
            df_ranked = transform!(sort(df_ranked, :PreyMass), :PreyMass, :PreyMass => eachindex => :MassRank)

            # APPROACH 2 -- do regressions between ranks
                # make and plot a least-squares line for the count-density rank relationship
                lsline_count = lm(@formula(CountRank ~ DensRank), df_ranked) # regression
                new_x_for_lsline = DataFrame(DensRank = 1:1:maximum(df_ranked.DensRank)) # make new x
                df_predict_y = predict(lsline_count,new_x_for_lsline) # make new y
                lines!(ax_dens,new_x_for_lsline.DensRank,df_predict_y) # plot line

                # find the difference in observed versus expected ranks
                rank_diffs = df_ranked.Frequencies .- df_ranked.Rel_density            
                scatter!(ax_mass_rank,df_ranked.PreyMass,rank_diffs, color = colorgrad[8], alpha=0.4)

                dens_slopes[i] = coef(lsline_count)[2] # grab the slope of the model

                numb_prey[i] = sum(df_ranked.CountInt)
                numb_prey_spp[i] = size(df_ranked,1)

                pval_dens[i] = coeftable(lsline_count).cols[4][2]

            # APPROACH 3 -- check each prey for optimality
            mean_ppp = mean(skipmissing(df_meta.PreyPerPellet))
            sum_R[i] = sum(df_diet.Density)
            p3 = df_meta.PreyPerPellet[i]
            if ismissing(p3)
                baow_a = -mean_ppp / (mean_ppp * sum_R[i] * (baow_h - 1))
            else
                baow_a = -p3 / (p3 * sum_R[i] * (baow_h - 1))
                baow_a_store[i] = baow_a
            end
            # gotta use the dens sorted prey base in the simulation to make a match with the simulated diet ranks
            df_ranked2 = sort(df_ranked, :DensRank) 

            println(i)
            if !ismissing(p3)
                NE_rate[i] = mean(df_ranked.PreyMass, weights(df_ranked.CountInt .+ 0)) * p3 # whole diet
                # expected prey mass gain rate per foraging period (assume 8 hours)
                # new way is to subtract off each item in turn
                if length(df_ranked.PreyMass) > 1
                    DietTests = zeros(Float64,length(df_ranked.PreyMass))
                    for zz = 1:length(df_ranked.PreyMass)
                        NE_rate_discounted = (mean(df_ranked.PreyMass, weights(df_ranked.Frequencies .+ 0)) -
                                        df_ranked.Frequencies[zz] .* df_ranked.PreyMass[zz]) * p3
                        NE_prey = df_ranked.PreyMass[zz] ./ baow_h 

                        # log the number of contrasts being done
                        num_NE_diffs_tested = num_NE_diffs_tested + 1 # every prey tested is another test
                        if NE_prey > NE_rate_discounted
                            num_NE_diffs_over_zero = num_NE_diffs_over_zero + 1 # tally up the contrasts in favor of OFT
                        end

                        scatter!(ax_net,NE_rate_discounted, log.(NE_prey), color = colorgrad[8], alpha=0.4)
                        scatter!(ax_net,NE_rate_discounted, log.(NE_rate_discounted), marker = :diamond, color = :black)
                        
                        DietTests[zz] = NE_prey - NE_rate_discounted
                                   
                    end
                    NumPositiveInStudy[i] = sum(DietTests .< 0)
                    OptForTests = hcat(OptForTests, DietTests')
                    
                end

                #number_prey_meeting_rule[i] = sum((NE_prey .- NE_rate[i]) .> 0) # just a simple tally of test outcome
                #number_prey_testing_rule[i] = length(NE_prey) # total number of prey tested

            end
        end
    end
    # add the proportion test to match the sequence of h values 
    prop_prey_meeting_rule[gg] = num_NE_diffs_over_zero / num_NE_diffs_tested 
end

# ===================================================
# put all the processed data into the meta DataFrame
# ===================================================
df_meta[!, :dens_slopes] = dens_slopes
df_meta[!, :pval_dens] = pval_dens
df_meta[!, :numb_prey] = numb_prey
df_meta[!, :numb_prey_spp] = numb_prey_spp
df_meta[!, :baow_a] = baow_a_store
df_meta[!, :mass_range] = mass_range
df_meta[!, :obs_largest_prey_freq] = observed_largest_prey_freq
df_meta[!, :NE_intake] = NE_rate
df_meta[!, :sum_R] = sum_R
df_meta[!, :FreqBirds] = FreqBirds
df_meta[!, :NumPos] = NumPositiveInStudy

# ====================================================================
# save the data set as a csv
CSV.write("BarnOwlProcessed.csv", df_meta)
# ====================================================================

# ===============================================================
# estimate the lower bound of the handling time
# ===============================================================
h_lower_bound = 1/mean(skipmissing(df_meta.PreyPerPellet))/3

# ===============================================================
# coming back to the net energy gain figure - main OFT figure
# ===============================================================
f_net_energy
prop_prey_meeting_rule = num_NE_diffs_over_zero / num_NE_diffs_tested
save("Net_energy_return_of_prey_log.png",f_net_energy)
# ===================================================

# ====================================================================
# Plot h against the proportion of diet matching the rule - Figure 2
# only works if you have run the whole vector of candidate h's
# ====================================================================

f_h_by_OF = Figure()
    ax_h = Axis(f_h_by_OF[1, 1], xlabel = "Handling time (days)", ylabel = "Proportion meeting rule")
    xlims!(ax_h,[0,0.2])
    scatter!(ax_h,candidate_h, prop_prey_meeting_rule, color = :black)
    lines!(ax_h,candidate_h, prop_prey_meeting_rule, color = :black, alpha=0.4)
    lines!(ax_h,[0.018,0.018],[0,1], color = colorgrad[8], linewidth = 4)
        text!(ax_h,(0.018+0.001,0.25), text = "Estimated handling time")
    lines!(ax_h,[h_lower_bound,h_lower_bound],[0,1], color = :green, linewidth = 4)
        text!(ax_h,(h_lower_bound+0.001,0.75), text = "Lower bound")
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
# coming back to the rank difference against prey body mass figure
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
    df_tests = dropmissing(df_meta,:dens_slopes)

    length(df_tests.dens_slopes[findall(df_tests.dens_slopes .> 0.0)]) / size(df_tests,1)

    # draw a 1:1 line overtop
    lines!(ax_dens,1:20,1:20,color = :black, linewidth=2)
        text!(ax_dens,(20,18), text = "1:1")

    hist!(ax_hist_dens,df_tests.dens_slopes, color = :white, strokewidth = 1, strokecolor = colorgrad[8])
        lines!(ax_hist_dens,0:0,0:100,color = :black, linewidth=2)
    save("Density_count_correlations.png",f_ranks)

# ===================================================
# see if species diversity influences diet diversity
# ===================================================

df_meta[!, :l_re] = log10.(df_meta.RERichness) # add logged variable to dataframe
df_meta[!, :l_ns] = log10.(df_meta.numb_prey_spp) # add logged variable to dataframe

# there are some zeros, so drop these out
df_meta_subset = df_meta[findall(df_meta.l_ns .!= 0),:] # drop rows with 0 body mass
df_meta_subset = df_meta_subset[findall(df_meta_subset.l_re .!= 0),:] # drop rows with 0 body mass
df_meta_subset = df_meta_subset[findall(df_meta_subset.sum_R .!= 0),:] # drop rows with 0 body mass

model_richness = lm(@formula(l_ns ~ l_re), df_meta_subset)
model_richness = lm(@formula(l_ns ~ log(LocalRichness)), df_meta_subset)
r2(model_richness)

#model_richness = fit(MixedModel, @formula(l_ns ~ l_re + (1|ecozone)), df_meta_subset)

diversity_test = Figure()
    ax_div = Axis(diversity_test[1, 1], xlabel = "Richness of rodents and true insectivores", ylabel = "Species richness is diet")
    a1 = scatter!(ax_div, df_meta_subset.l_re, df_meta_subset.l_ns, color = colorgrad[8], alpha=0.4)
   
    new_x = DataFrame(l_re = range(minimum(df_meta_subset.l_re), maximum(df_meta_subset.l_re), length=10))
    df_predict_y = predict(model_richness, new_x, interval=:confidence, level = 0.95)
    # for some reason the predictions data type is coming out weird, so change it
    df_predict_y[!,:prediction] = convert.(Float64,df_predict_y[!,:prediction])
    df_predict_y[!,:lower] = convert.(Float64,df_predict_y[!,:lower])
    df_predict_y[!,:upper] = convert.(Float64,df_predict_y[!,:upper])

    band!(ax_div,new_x.l_re,df_predict_y.lower,df_predict_y.upper, color = (:gray, 0.2))
    rod = lines!(ax_div,new_x.l_re, df_predict_y.prediction, linewidth = 2, color = :black)

    save("Prey_richness.png",diversity_test)

# ===================================================
# try to explain variation in density slopes
# ===================================================

model_ecozone = lm(@formula(dens_slopes ~ ecozone), df_meta)
# some diffs by ecozone, so we'll use ecozone as a random effect

model_num_spp = fit(MixedModel, @formula(dens_slopes ~ numb_prey_spp + (1|ecozone)), df_meta)
model_sumR = fit(MixedModel, @formula(dens_slopes ~ sum_R + (1|ecozone)), df_meta)
model_RERichness = fit(MixedModel, @formula(dens_slopes ~ RERichness + (1|ecozone)), df_meta)
model_FreqBirds = fit(MixedModel, @formula(dens_slopes ~ FreqBirds + (1|ecozone)), df_meta)
model_RaptorRichness = fit(MixedModel, @formula(dens_slopes ~ RaptorRichness + (1|ecozone)), df_meta)

model_1 = fit(MixedModel, @formula(dens_slopes ~ sum_R + numb_prey_spp + RERichness + RaptorRichness + FreqBirds + (1|ecozone)), df_meta_subset)
model_1 = fit(MixedModel, @formula(dens_slopes ~ sum_R + numb_prey_spp + RERichness + RaptorRichness + (1|ecozone)), df_meta)

model_1 = fit(MixedModel, @formula(dens_slopes ~ sum_R + numb_prey_spp + RERichness + RaptorRichness + FreqBirds + (1|ecozone)), df_meta_subset)

model_1 = fit(MixedModel, @formula(sum_R ~ numb_prey_spp + (1|ecozone)), df_meta_subset)
model_1 = fit(MixedModel, @formula(sum_R ~ numb_prey_spp + (1|ecozone)), df_meta)

# test NE intake separately because this reduces the sample size to 330
model_NE_intake = fit(MixedModel, @formula(dens_slopes ~ NE_intake + (1|ecozone)), df_meta)





# check for some correlations among predictors
density_model1 = lm(@formula(sum_R ~ NE_intake), df_meta)
density_model1 = lm(@formula(sum_R ~ RERichness), df_meta)
density_model1 = lm(@formula(NE_intake ~ RERichness), df_meta)

model_1 = fit(MixedModel, @formula(dens_slopes ~ FreqBirds + RaptorRichness + sum_R + numb_prey_spp + (1|ecozone)), df_meta)



density_model1 = lm(@formula(dens_slopes ~ NE_intake), df_meta)



density_model1 = lm(@formula(dens_slopes ~ sum_R), df_meta)
density_model1 = lm(@formula(dens_slopes ~ LocalRichness), df_meta)
density_model1 = lm(@formula(dens_slopes ~ RERichness), df_meta)

density_model1 = lm(@formula(dens_slopes ~ sum_R + NE_intake + numb_prey_spp), df_meta)
density_model1 = lm(@formula(dens_slopes ~ sum_R * numb_prey_spp), df_meta)
r2(density_model1)



fit(MixedModel, @formula(dens_slopes ~ 1 + (1|Continent)), df_meta)
lmm1 = fit(MixedModel, @formula(dens_slopes ~ sum_R + numb_prey_spp + RERichness + (1|ecozone)), df_meta)
lmm1 = fit(MixedModel, @formula(dens_slopes ~ sum_R + numb_prey_spp + (1|ecozone)), df_meta)


r2(lmm1)



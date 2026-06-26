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

dir = "C:/Users/jdelong2/OneDrive - University of Nebraska/Projects in progress/Barn_owl_global_diet/"
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
IsTopPreyBiggest = zeros(Float64, howmanystudies)
IsTopPreyMostDense = zeros(Float64, howmanystudies)
largest_prey_mass = zeros(Float64, howmanystudies)
smallest_prey_mass = zeros(Float64, howmanystudies)
mass_of_most_common_dropones = zeros(Float64, howmanystudies)
mass_range = zeros(Float64, howmanystudies)

baow_a_store = zeros(Float64, howmanystudies)
number_prey_meeting_rule = zeros(Float64, howmanystudies)
number_prey_testing_rule = zeros(Float64, howmanystudies)
predicted_largest_prey_freq = zeros(Float64, howmanystudies)
observed_largest_prey_freq = zeros(Float64, howmanystudies)
predicted_smallest_prey_freq = zeros(Float64, howmanystudies)
observed_smallest_prey_freq = zeros(Float64, howmanystudies)
NE_rate = Array{Union{Float64,Missing}}(missing, howmanystudies)
sum_R = zeros(Float64, howmanystudies)
NumPositiveInStudy = zeros(Float64, howmanystudies)
family_matrix = zeros(Float64, howmanystudies,size(sums_by_family,1))

# ==========================================================================================
###### step 4, # cycle through the datasets to generate ranks, slopes, and diversity metrics
# ==========================================================================================

# read in the posterior sample for the mammal density allometry
df_chains = CSV.read("Density_allometry_chain.csv", DataFrame)

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
            df_diet[!, :Density] = exp.(df_diet.LogDensity) # convert to non-log University

            #= ####################################################################################
            # here we need to predict the density from the mass and the order
            # using the posteriors of the intercept and scaling parameters from the allometry analysis
            # need to estimate a density for each of the posterior samples (given by length of df_chains)
            # all these estimates need to be added to df_diet and then the can be compared to the total as the OFT test
            density_post = zeros(Float64,length(numb_diet),size(df_chains,1))
            df_predict_y = zeros(Float64,length(numb_diet))
            for jjj = 1:length(numb_diet)
                # grab the slopes
                column_name = string("orders",df_diet.group_idx[jjj]) # get the name of the right slope
                idx = columnindex(df_chains, column_name) # determine what column that is in
                fitted_beta_mass = df_chains[:,idx] # grab the value
                fitted_beta_mass_median = median(df_chains[:,idx]) # grab the value

                # grab the intercepts
                column_name = string("Intercept",df_diet.group_idx[jjj])
                idx = columnindex(df_chains, column_name)
                fitted_int = df_chains[:,idx]
                fitted_int_median = median(df_chains[:,idx])
                # calculate the predicted density
                predictions = fitted_int .+ fitted_beta_mass .* df_diet.LogMass[jjj]
                # flip it and exponentiate it to get the actual density
                density_post[jjj,1:size(df_chains,1)] = exp.(predictions)' # calculate the fitted y

                # also need to do it just for the median
                df_predict_y[jjj] = fitted_int_median + fitted_beta_mass_median * df_diet.LogMass[jjj]
            end
            
            # to add the whole set of densities calculated from the posterior sample,
            # first transform it the matrix to a dataframe and then hcat them
            mat_df = DataFrame(density_post, :auto)
            df_diet = hcat(df_diet, mat_df)
            =#

            #####################################################################################

            # calculate frequencies of diet items
            df_diet[!, :Frequencies] = df_diet.CountInt./ sum(df_diet.CountInt)
            df_diet[!, :Rel_density] = df_diet.Density./ sum(df_diet.Density)

            # here find the biggest prey and predict its frequency
            row_biggest = findall(df_diet.PreyMass .== maximum(df_diet.PreyMass))
            rows_except_biggest = findall(df_diet.PreyMass .< maximum(df_diet.PreyMass))
            phi = sum((df_diet.PreyMass[rows_except_biggest] ./ maximum(df_diet.PreyMass)) .^ 0 .* df_diet.Density[rows_except_biggest]) ./ df_diet.Density[row_biggest]
            predicted_largest_prey_freq[i] = 1 / (1 + phi[])
            observed_largest_prey_freq[i] = df_diet.Frequencies[row_biggest][]

            # here find the smallest prey and predict its frequency
            row_smallest = findall(df_diet.PreyMass .== minimum(df_diet.PreyMass))[1]
            rows_except_smallest = findall(df_diet.PreyMass .> minimum(df_diet.PreyMass))
            phi = sum((df_diet.PreyMass[rows_except_biggest] ./ maximum(df_diet.PreyMass)) .^ 0 .* df_diet.Density[rows_except_biggest]) ./ df_diet.Density[row_biggest]
            predicted_smallest_prey_freq[i] = 1 / (1 + phi[])
            observed_smallest_prey_freq[i] = df_diet.Frequencies[row_smallest][]

            # first calculate the rank of prey by density, highest to lowest
            df_rank_density = transform!(sort(df_diet, :Density), :Density, :Density => eachindex => :DensRank)
            # then calculate the rank of prey by count, highest to lowest
            df_ranked = transform!(sort(df_rank_density, :CountInt), :CountInt, :CountInt => eachindex => :CountRank)
            # then calculate the rank of prey by body mass, highest to lowest
            df_ranked = transform!(sort(df_ranked, :PreyMass), :PreyMass, :PreyMass => eachindex => :MassRank)

            # determine the body mass of the largest prey
                largest_prey_mass[i] = df_ranked.PreyMass[findall(df_ranked.MassRank .== length(numb_diet))][] # find row of most common
                smallest_prey_mass[i] = df_ranked.PreyMass[findall(df_ranked.MassRank .== 1)][] # find row of most common

                df_ranked_dropones = df_ranked[findall(df_ranked.CountInt .!= 1),:] # drop rows with 0 body mass
                mass_of_most_common_dropones[i] = df_ranked.PreyMass[findall(df_ranked.CountRank .== length(numb_diet))][] # find row of most common

            # APPROACH 1 -- determine if the most common prey is the biggest or the most dense
                row_max_count = findall(df_ranked.CountInt .== maximum(df_ranked.CountInt)) # find row of most common
                row_max_mass = findall(df_ranked.PreyMass .== maximum(df_ranked.PreyMass)) # find row of biggest
                row_max_dens = findall(df_ranked.Density .== maximum(df_ranked.Density)) # find row of most dense
                # if the prey is the biggest, then change to 1
                if row_max_count == row_max_mass
                    IsTopPreyBiggest[i] = 1
                end
                # if the prey is the densest, then change to 1
                if row_max_count == row_max_dens
                    IsTopPreyMostDense[i] = 1
                end

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

            # APPROACH 3 -- simulate the optimal diet

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

            # determine the range of masses in the diet
            mass_range[i] = maximum(df_diet.PreyMass) - minimum(df_diet.PreyMass)
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
df_meta[!, :mass_of_most_common_dropones] = mass_of_most_common_dropones
df_meta[!, :IsTopPreyBiggest] = IsTopPreyBiggest
df_meta[!, :smallest_prey_mass] = smallest_prey_mass
df_meta[!, :largest_prey_mass] = largest_prey_mass
df_meta[!, :baow_a] = baow_a_store
df_meta[!, :mass_range] = mass_range
df_meta[!, :pred_largest_prey_freq] = predicted_largest_prey_freq
df_meta[!, :obs_largest_prey_freq] = observed_largest_prey_freq
df_meta[!, :NE_intake] = NE_rate
df_meta[!, :sum_R] = sum_R
df_meta[!, :FreqBirds] = FreqBirds
df_meta[!, :NumPos] = NumPositiveInStudy

# ====================================================================
# save the data set as a csv
CSV.write("BarnOwlProcessed.csv", df_meta)
# ====================================================================

minimum(df_meta.numb_prey_spp)
maximum(df_meta.numb_prey_spp)
maximum(df_meta.NumPos)

h_lower_bound = 1/mean(skipmissing(df_meta.PreyPerPellet))/3
#yscale = log10, xscale = log10

# ===============================================================
# coming back to the net energy gain figure - main OFT figure
# ===============================================================
f_net_energy
prop_prey_meeting_rule = num_NE_diffs_over_zero / num_NE_diffs_tested
    ax_ofthist = Axis(f_net_energy[1, 2], xlabel = "Difference", ylabel = "Observed frequency")
        #xlims!(ax_ofthist,[-100, 10000])
    hist!(ax_ofthist,OptForTests[:], color = :white, strokewidth = 1, strokecolor = colorgrad[8])
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
# Plot the families against the number of diets that family occurs in
# ====================================================================
sums_by_family[!, :Diets] = sum(family_matrix, dims=1)'[:] # add the sum of the family matrix to the families summary
sums_by_family = sort(sums_by_family, :Diets, rev=true) # sort the diets from highest to lowest

f_families = Figure()
    ax_fams = Axis(f_families[1, 1], xticks = (1:46, sums_by_family.PreyFamily), xticklabelrotation = 90*pi/180, xlabel = "Family", ylabel = "Number of diets")
    barplot!(ax_fams,1:1:46, sums_by_family.Diets)
    save("FamiliesInDiets.png",f_families)
# ====================================================================

# ====================================================================
# Plot the body mass distribution of prey at the species level
# ====================================================================
df_mass_by_species = combine(groupby(df_baow, :PreyScientificName), :PreyMass .=> mean)

f_miss_dist = Figure()
    ax_mass_dist = Axis(f_miss_dist[1, 1], xlabel = "Body mass (g)", ylabel = "Frequency")
    hist!(log10.(df_mass_by_species.PreyMass_mean), color = :white, strokewidth = 1, strokecolor = colorgrad[8])
# ====================================================================

# ====================================================================
# Analyze the variation in the number of sub-optimal cases per diet
# ====================================================================

# subset the data
df_meta_subset1 = dropmissing(df_meta,:PreyPerPellet)
    df_meta_subset1 = df_meta_subset1[findall(df_meta_subset1.numb_prey_spp .!= 0),:] # drop rows with 0 prey
    minimum(df_meta_subset1.NumPos)
    maximum(df_meta_subset1.NumPos)

    df_meta_subset1 = df_meta_subset1[findall(df_meta_subset1.NumPos .!= 0),:] # drop rows with 0 suboptimal

f_negs = Figure()
    ax_negs = Axis(f_negs[1, 1], xlabel = "Number of negs", ylabel = "Frequency")
    hist!(df_meta_subset1.NumPos, color = :white, strokewidth = 1, strokecolor = colorgrad[8])

    sum(df_meta_subset1.NumPos)
    num_NE_diffs_tested - num_NE_diffs_over_zero

# use a mixed glm to test predictors
neg_model1 = glm(@formula(NumPos ~ numb_prey_spp), df_meta_subset1, Poisson(), LogLink())
neg_model1 = glm(@formula(NumPos ~ RERichness), df_meta_subset1, Poisson(), LogLink())
neg_model1 = glm(@formula(NumPos ~ FreqBirds), df_meta_subset1, Poisson(), LogLink())



neg_model1 = glm(@formula(NumPos ~ numb_prey_spp + NE_intake + sum_R + FreqBirds + RERichness), df_meta_subset1, Poisson(), LogLink())
model_1 = fit(MixedModel, @formula(NumPos ~ sum_R + log(numb_prey_spp) + NE_intake + FreqBirds + RERichness + (1|ecozone)), df_meta_subset1, NegativeBinomial(2.0), LogLink())
model_1 = fit(MixedModel, @formula(NumPos ~ log(sum_R) + log(numb_prey_spp) + log(NE_intake) + FreqBirds + RERichness + (1|ecozone)), df_meta_subset1, Poisson(), LogLink())

model_1 = fit(MixedModel, @formula(NumPos ~ sum_R + numb_prey_spp + FreqBirds + RERichness + (1|ecozone)), df_meta_subset1, Poisson(), LogLink())
neg_model1 = glm(@formula(NumPos ~ numb_prey_spp + FreqBirds + RERichness), df_meta_subset1, Poisson(), LogLink())

neg_model1 = lm(@formula(sum_R ~ numb_prey_spp), df_meta_subset1)





# ====================================================================
df_mass_by_species = combine(groupby(df_baow, :PreyScientificName), :PreyMass .=> mean)
# ====================================================================


f_test = Figure()
    ax_test = Axis(f_test[1, 1], xlabel = "Predicted frequency of largest prey", ylabel = "Observed frequency of largest prey")
    a1 = scatter!(ax_test,predicted_largest_prey_freq, observed_largest_prey_freq, color = :black, alpha=0.4)
    lines!(ax_test,0:1,0:1,color = :black, linewidth=2)
    
    ax_smallest = Axis(f_test[1, 2], xlabel = "Predicted frequency of smallest prey", ylabel = "Observed frequency of smallest prey")
    a1 = scatter!(ax_smallest,predicted_smallest_prey_freq, observed_smallest_prey_freq, color = :black, alpha=0.4)
    lines!(ax_smallest,0:1,0:1,color = :black, linewidth=2)
    
    ax_cross = Axis(f_test[1, 3], xlabel = "Observed frequency of largest prey", ylabel = "Observed frequency of smallest prey")
    a1 = scatter!(ax_cross,observed_largest_prey_freq, observed_smallest_prey_freq, color = :black, alpha=0.4)
    lines!(ax_cross,0:1,0:1,color = :black, linewidth=2)

    save("Largest_prey_predicted_observed.png",f_test)



sum(df_meta.IsTopPreyMostDense)
sum(df_meta.IsTopPreyBiggest)



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
    # some regressions don't have enough points to work
    df_tests = dropmissing(df_meta,:dens_slopes)
    #df_tests.pval_mass = replace(df_tests.pval_mass, NaN=>missing)
    #df_tests = dropmissing(df_tests,:pval_mass)

    length(df_tests.dens_slopes[findall(df_tests.dens_slopes .> 0.0)]) / size(df_tests,1)

    # draw a 1:1 line overtop
    lines!(ax_dens,1:20,1:20,color = :black, linewidth=2)
        text!(ax_dens,(20,18), text = "1:1")
    #lines!(ax_mass,1:20,1:20,color = :black, linewidth=2)
    #    text!(ax_mass,(20,18),text = "1:1")

    hist!(ax_hist_dens,df_tests.dens_slopes, color = :white, strokewidth = 1, strokecolor = colorgrad[8])
        lines!(ax_hist_dens,0:0,0:100,color = :black, linewidth=2)
    #hist!(ax_hist_mass,df_tests.mass_slopes, color = :white, strokewidth = 1, strokecolor = :blue)
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
    rod = lines!(ax_div,new_x.l_re, df_predict_y.prediction, linewidth = 2, color = :black)
        band!(ax_div,new_x.l_re,df_predict_y.lower,df_predict_y.upper; color = (:black, 0.2))

    save("Prey_richness.png",diversity_test)

#=
ShannEnt_regression = lm(@formula(ShannEnt_rare ~ DietRichness_rare + PreyPerPellet + largest_prey_mass), df_meta)
density_regression = lm(@formula(dens_slopes ~ DietRichness_rare + PreyPerPellet + largest_prey_mass), df_meta)
ShannEnt_residuals = df_meta.ShannEnt_rare - predict(ShannEnt_regression, df_meta)
density_residuals = df_meta.dens_slopes - predict(density_regression, df_meta)

df_resid = DataFrame(ShannEnt_rare = ShannEnt_residuals, dens_slopes = density_residuals)
    of1 = lm(@formula(dens_slopes ~ ShannEnt_rare), df_resid)

f_partials = Figure()
    ax_part1 = Axis(f_partials[1, 1], xlabel = "Residual Shannon entropy", ylabel = "Residual density slopes")
    p1 = scatter!(ax_part1,df_resid.ShannEnt_rare, df_resid.dens_slopes,alpha=0.6)
    new_x_for_lsline = DataFrame(ShannEnt_rare = [minimum(skipmissing(df_resid.ShannEnt_rare)):0.1:maximum(skipmissing(df_resid.ShannEnt_rare));]) # make new x
            predict_y = predict(of1,new_x_for_lsline) # predict new y variable
            lines!(ax_part1,new_x_for_lsline.ShannEnt_rare,predict_y)
=#

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




# first test the prediction that mass range would be
density_model1 = lm(@formula(dens_slopes ~ mass_range), df_meta)
f_mass_range = Figure()
    ax_mr = Axis(f_mass_range[1, 1], xlabel = "Mass range", ylabel = "Density slopes")
                #ylims!(ax_mr,0,1)
    p1 = scatter!(ax_mr,df_meta.mass_range, df_meta.dens_slopes,alpha=0.6)
    
r2(density_model1)



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





# partial regression dens_slopes against Shannon entropy
mass_range_regression = lm(@formula(mass_range ~ DietRichness_rare + PreyPerPellet + largest_prey_mass), df_meta)
density_regression = lm(@formula(dens_slopes ~ DietRichness_rare + PreyPerPellet + largest_prey_mass), df_meta)
mass_range_residuals = df_meta.mass_range - predict(ShannEnt_regression, df_meta)
density_residuals = df_meta.dens_slopes - predict(density_regression, df_meta)

df_resid = DataFrame(mass_range = ShannEnt_residuals, dens_slopes = density_residuals)
    of1 = lm(@formula(dens_slopes ~ mass_range), df_resid)

f_partials = Figure()
    ax_part1 = Axis(f_partials[1, 1], xlabel = "Residual Shannon entropy", ylabel = "Residual density slopes")
    p1 = scatter!(ax_part1,df_resid.ShannEnt_rare, df_resid.dens_slopes,alpha=0.6)
    new_x_for_lsline = DataFrame(ShannEnt_rare = [minimum(skipmissing(df_resid.ShannEnt_rare)):0.1:maximum(skipmissing(df_resid.ShannEnt_rare));]) # make new x
            predict_y = predict(of1,new_x_for_lsline) # predict new y variable
            lines!(ax_part1,new_x_for_lsline.ShannEnt_rare,predict_y)




# run some linear models to see if things are related to density slopes
density_model1 = lm(@formula(dens_slopes ~ ShannEnt_rare), df_meta)
density_model2 = lm(@formula(dens_slopes ~ DietRichness_rare), df_meta)
density_model1 = lm(@formula(dens_slopes ~ ShannEnt_rare + DietRichness_rare), df_meta)

density_model1 = lm(@formula(dens_slopes ~ numb_prey), df_meta)
density_model1 = lm(@formula(dens_slopes ~ largest_prey_mass), df_meta)
density_model1 = lm(@formula(dens_slopes ~ smallest_prey_mass), df_meta)

density_model1 = lm(@formula(dens_slopes ~ PreyPerPellet), df_meta)


density_model1 = lm(@formula(dens_slopes ~ ShannEnt_rare + DietRichness_rare + PreyPerPellet + mass_range), df_meta)


r2(density_model1)

density_model1 = lm(@formula(dens_slopes ~ DietRichness_rare), df_meta)
density_model1 = lm(@formula(mass_range ~ largest_prey_mass), df_meta)


# =========================================================
# partial regression dens_slopes against Shannon entropy
# =========================================================
ShannEnt_regression = lm(@formula(ShannEnt_rare ~ DietRichness_rare + PreyPerPellet + largest_prey_mass), df_meta)
density_regression = lm(@formula(dens_slopes ~ DietRichness_rare + PreyPerPellet + largest_prey_mass), df_meta)
ShannEnt_residuals = df_meta.ShannEnt_rare - predict(ShannEnt_regression, df_meta)
density_residuals = df_meta.dens_slopes - predict(density_regression, df_meta)

df_resid = DataFrame(ShannEnt_rare = ShannEnt_residuals, dens_slopes = density_residuals)
    of1 = lm(@formula(dens_slopes ~ ShannEnt_rare), df_resid)

f_partials = Figure()
    ax_part1 = Axis(f_partials[1, 1], xlabel = "Residual Shannon entropy", ylabel = "Residual density slopes")
    p1 = scatter!(ax_part1,df_resid.ShannEnt_rare, df_resid.dens_slopes,alpha=0.6)
    new_x_for_lsline = DataFrame(ShannEnt_rare = [minimum(skipmissing(df_resid.ShannEnt_rare)):0.1:maximum(skipmissing(df_resid.ShannEnt_rare));]) # make new x
            predict_y = predict(of1,new_x_for_lsline) # predict new y variable
            lines!(ax_part1,new_x_for_lsline.ShannEnt_rare,predict_y)


density_model1 = lm(@formula(PreyPerPellet ~ DietRichness_rare), df_meta)



    of1 = lm(@formula(dens_slopes ~ Sim_slopes_opt), df_meta)
    of1 = lm(@formula(dens_slopes ~ Sim_slopes_rand), df_meta)



f_entropy = Figure()
    ax_ent = Axis(f_entropy[1, 1], xlabel = "Sim_slopes_rand", ylabel = "dens_slopes")
    hm = scatter!(ax_ent,df_meta.Sim_slopes_opt, df_meta.dens_slopes,alpha=0.4)
    hm2 = scatter!(ax_ent,df_meta.Sim_slopes_rand, df_meta.dens_slopes,alpha=0.4)


df_meta[!, :pval_mass] = pval_mass

    include("min_max_entropy.jl")

    df_meta_reduced = df_meta[findall(df_meta.Simulated_opt_richness .>= 1.0),:] # drop rows with 0 body mass


    df_meta_reduced = dropmissing(df_meta,:mediandeltaAIC)

f_entropy = Figure()
    ax_ent = Axis(f_entropy[1, 1], xlabel = "Richness", ylabel = "Entropy")
    hm = scatter!(ax_ent,df_meta.Simulated_opt_richness,df_meta.Simulated_opt_SE, alpha=0.4)
    hm = scatter!(ax_ent,df_meta.Simulated_rand_richness,df_meta.Simulated_rand_SE, alpha=0.4)
    hm2 = scatter!(ax_ent,df_meta.DietRichness_rare,df_meta.ShannEnt_rare, alpha=0.4)
    max_richness = round(maximum(df_meta.DietRichness_rare),digits = 0)
    SE_max, SE_min, x_rich_max, x_rich_min = min_max_entropy(max_richness,200)
    hm4 = lines!(ax_ent,x_rich_max,SE_max)
    hm5 = lines!(ax_ent,x_rich_min,SE_min)

    ax_obs_exp = Axis(f_entropy[1, 2], xlabel = "Predicted entropy", ylabel = "Observed entropy")
    hm3 = scatter!(ax_obs_exp,df_meta_reduced.Simulated_opt_SE,df_meta_reduced.ShannEnt_rare)
    hm6 = lines!(ax_obs_exp,[0, 2],[0, 2])

        hm3 = scatter!(ax_obs_exp,df_meta_reduced.Simulated_opt_richness,df_meta_reduced.DietRichness_rare)



    of1 = lm(@formula(ShannEnt_rare ~ Simulated_rand_SE), df_meta_reduced)
    r2(of1)
    of1 = lm(@formula(ShannEnt_rare ~ Simulated_opt_SE), df_meta_reduced)
    r2(of1)


lm2 = lm(@formula(smallest_prey_mass ~ PreyPerPellet + numb_prey), df_meta)
lm2 = lm(@formula(largest_prey_mass ~ PreyPerPellet), df_meta)
lm2 = lm(@formula(ShannEnt_rare ~ PreyPerPellet + DietRichness_rare), df_meta)


lm2 = lm(@formula(mass_slopes ~ ShannEnt_rare), df_meta)
lm2 = lm(@formula(dens_slopes ~ ShannEnt_rare), df_meta)
lm2 = lm(@formula(abs(mass_slopes) ~ ShannEnt_rare), df_meta)

lm2 = lm(@formula(mass_slopes ~ ShannEnt_rare + PreyPerPellet), df_meta)
lm2 = lm(@formula(ShannEnt_rare ~ PreyPerPellet + DietRichness_rare), df_meta)


lm2 = lm(@formula(dens_slopes ~ ShannEnt_rare * mass_of_most_common_dropones), df_meta)
lm2 = lm(@formula(dens_slopes ~ DietRichness_rare * ShannEnt_rare), df_meta)
mainr2 = r2(lm2)

logit = glm(@formula(IsTopPreyBiggest ~ mass_of_most_common_dropones), df_meta, Binomial(), ProbitLink())
lm2 = lm(@formula(mass_slopes ~ mass_of_most_common_dropones + ShannEnt_rare), df_meta)
lm2 = lm(@formula(ShannEnt_rare ~ mass_of_most_common_dropones), df_meta)
mainr2 = r2(lm2)



f_mass = Figure()
    ax_mass = Axis(f_mass[1, 1], xlabel = "Mass of largest prey", ylabel = "Largest prey is most common")
    hm = scatter!(ax_mass,df_meta.mass_of_most_common,df_meta.IsTopPreyBiggest)
    ax_mass2 = Axis(f_mass[1, 2], xlabel = "Mass of largest prey", ylabel = "Mass slope")
    hm = scatter!(ax_mass2,df_meta.mass_of_most_common,df_meta.mass_slopes)


f_ranks


#=
# how many positive slopes?
sum(dens_slopes .> 0.0)
sum(dens_slopes .< 0.0)
sum(dens_slopes .== 0.0)

# does the slope depend on the sample size?
CorrelationTest(mass_slopes,numb_prey)
# does the slope depend on the diversity?
CorrelationTest(mass_slopes,numb_prey_spp)

# does the slope depend on the sample size?
CorrelationTest(dens_slopes,numb_prey)
# does the slope depend on the diversity?
CorrelationTest(dens_slopes,numb_prey_spp)
=#

# run a lm to determine if owls trade off foraging randomly versus foraging optimally
lm(@formula(dens_slopes ~ numb_prey), df_tests)

f_corr = Figure()
ax_corrs = Axis(f_corr[1, 1], xlabel = "Slope of body mass line", ylabel = "Slope of density line")
hm = scatter!(ax_corrs,df_tests.mass_slopes,df_tests.dens_slopes, color = convert(Array{Float64,1},df_tests.pval_dens), colormap = :lighttest)
Colorbar(f_corr[1, 2], hm, label = "Prey diversity in diet")

ax_corrs2 = Axis(f_corr[1, 2], xlabel = "Prey diversity in diet", ylabel = "Slope of density line")
    #xlims!(ax_corrs2,0,5e3)
hm = scatter!(ax_corrs2,df_tests.numb_prey_spp, df_tests.dens_slopes,color = convert(Array{Float64,1},df_tests.pval_dens), colormap = :lighttest)
Colorbar(f_corr[1, 3], hm, label = "p value")
    scatter!(ax_corrs2,df_tests.numb_prey_spp[df_tests.IsTopPreyBiggest .== 1], df_tests.dens_slopes[df_tests.IsTopPreyBiggest .== 1], color = :black)
    scatter!(ax_corrs2,df_tests.numb_prey_spp[df_tests.IsTopPreyMostDense .== 1], df_tests.dens_slopes[df_tests.IsTopPreyMostDense .== 1], color = :gray)


f_corr
save("Count_corrs_linearmodel.png",f_corr)


#colormap = :lighttest

# process the diet data from the OS-Prey database

dir = "C:/Users/jdelong2/OneDrive - University of Nebraska/Projects in progress/Raptor foraging and diets/"

dir = "C:/Users/johnp/Documents/GitHub/OSPrey-database/Barn owl optimal foraging manuscript"

cd(dir)

using CSV
using DataFrames
using Tables
using Statistics
using Plots
using FreqTables

df_diets = CSV.read("RaptorDiets_V3.csv",DataFrame) # read in diet data
# drop any rows with missing count data
df_diets = dropmissing(df_diets,:Count)
    df_diets.Count = replace(df_diets.Count, "NA"=>missing) # switch NAs to missing
    # switch the string numbers to Int
    df_diets[!,:CountInt] = passmissing(parse).([Int64],df_diets[!,:Count])

# drop any rows with count data as NA
#df_diets = df_diets[findall(df_diets.CountNumber .!= "NA"),:] # drop everything that is rows with just NA

# remove likely carrion from all diets, identified as being class = artiodactyla
#filter!(row -> !(row.PreyOrder == "Artiodactyla"),  df_diets)

# pull some useful columns
data_sets = df_diets.DataSet
prey_counts = df_diets.Count
prey_id_level = df_diets.IDLevel
prey_class = df_diets.PreyClass
prey_order = df_diets.PreyOrder
prey_sci_name = df_diets.PreyScientificName

# how many diets are there?
num_studies = maximum(data_sets)

# what is the size of the diets database
length_of_diet_df = size(df_diets,1)

df_meta = CSV.read("RaptorDiets_Metadata_V3.csv",DataFrame) # read in study overview data
    df_meta = dropmissing(df_meta,:DataSet)
    df_meta.PreyPerPellet = replace(df_meta.PreyPerPellet, "NA"=>missing) # switch NAs to missing

df_biome = CSV.read("biome_world.csv",DataFrame) # read in the biome data
df_meta[!, :ecozone] .= df_biome.ecozone # add the biome ecozone type to the meta dataset

# first need to get some genus and family-level averages for mammal masses
    df_mamms = CSV.read("EltonTraits_mammals.csv",DataFrame) # read in mammal body trait data

    # genus level
    gd = groupby(df_mamms, [:Scientific])
    mass_means_genus = combine(gd, :BodyMassValue => mean)
    vec_mod = [x * suffix for x in mass_means_genus.Scientific for suffix in [" sp"]] # this appends "sp" to each genus
    mass_means_genus[!, :new_genus] .= vec_mod # add it to the data frame

    #= family level
    gd = groupby(df_mamms, [:Family])
    mass_means_family = combine(gd, :BodyMassValue => mean)
    rename!(mass_means_family, :Family => :Scientific)

    df_mean_mass = append!(mass_means_family,mass_means_genus)
    =#

###### pass in mammal masses
    unique_prey = unique(prey_sci_name) # how many unique prey are there?
    prey_name_Elton = df_mamms.SciName
    prey_mass_Elton = df_mamms.BodyMassValue
    prey_family_Elton = df_mamms.Family

    prey_mass_diet = zeros(Float64, length_of_diet_df) # open an empty vector to hold mass values
    prey_family = Array{Union{String,Missing}}(missing, length_of_diet_df) # open an empty vector to hold family names

    # first run it through the Elton database as is to find species-level masses and families
    for i = 1:length(unique_prey)

        # passing the masses for prey id'd to species
        prey_row1 = findall(prey_name_Elton .== unique_prey[i]) # grab the ith prey name from the unique list and find those rows in the Elton database
        matches = findall(prey_sci_name .== unique_prey[i]) # find where that prey occurs in df_diet
        
        mass_to_pass = prey_mass_Elton[prey_row1] # snag the mass of the prey in row i
        if length(mass_to_pass) > 0 # only do this if there is an actual value in for mass
            for j = 1:length(matches)
                prey_mass_diet[matches[j]] = mass_to_pass[1] # pass the value to every row for that species
            end
        end

        # passing the masses for prey id'd to genus
        prey_row2 = findall(mass_means_genus.new_genus .== unique_prey[i]) # grab the ith prey name from the unique list and find those rows in the Elton database
        matches = findall(prey_sci_name .== unique_prey[i]) # find where that prey occurs in df_diet
        mass_to_pass = mass_means_genus.BodyMassValue_mean[prey_row2] # snag the mass of the prey in row i
        if length(mass_to_pass) > 0 # only do this if there is an actual value in for mass
            for j = 1:length(matches)
                prey_mass_diet[matches[j]] = mass_to_pass[1] # pass the value to every row for that species
            end
        end        

        # passing the family names
        family_to_pass = prey_family_Elton[prey_row1]
        if ~isempty(family_to_pass)
            for j = 1:length(matches)
                prey_family[matches[j]] = family_to_pass[1] # pass the value to every row for that species
            end
        end
    end


###### run through each diet and get diet-level characteristics
df_polyfids = CSV.read("POLYFID_locs.csv",DataFrame) # read in the location ID's
df_mammal_richness = CSV.read("mammal-richness.csv",DataFrame) # read in the location ID's
df_RE_richness = CSV.read("mammal-richness_Rodentia_Eulipotyphla.csv",DataFrame) # read in the location ID's
df_raptor_richness = CSV.read("bird-breeding-richness_raptors.csv",DataFrame) # read in the location ID's


ShannEnt = zeros(Float64, num_studies)
DietRichness = zeros(Float64, num_studies)
ShannEnt_rare = zeros(Float64, num_studies)
DietRichness_rare = zeros(Float64, num_studies)

LocalRichness = zeros(Float64, num_studies)
RERichness = zeros(Float64, num_studies)
RaptorRichness = zeros(Float64, num_studies)

# rarefaction target numbers
    rarefy_reps = 200
    rarefy_to = 100

for i = 1:num_studies
    println(i)
    counts_to_use = df_diets.CountInt[findall(df_diets.DataSet .== i)] # pull out diet count data 
    freqs_to_use = counts_to_use ./ sum(counts_to_use) # turn counts into frequencies

    # only run the diets where the data is in counts (excluding diets only reported as frequencies)
    if !ismissing(counts_to_use[1])
        ShannEnt[i] = -sum(freqs_to_use .* log.(freqs_to_use)) # calculate Shannon entropy
        DietRichness[i] = length(counts_to_use)

        #prey_to_use = prey_name(data_sets == i);
        #class_to_use = prey_class(data_sets == i);
        #mass_to_use = prey_mass_diet(data_sets == i);
        #prey_id_level_to_use = prey_id_level(data_sets == i)

        # grab the mammal richness via the POLYFID
        diet_polyfid = df_polyfids.POLYFID[df_polyfids.DataSet .== i]
        if !isempty(diet_polyfid)
            mammalcheck = df_mammal_richness.species[findall(df_mammal_richness.POLYFID .== diet_polyfid)]
            if !isempty(mammalcheck)
                LocalRichness[i] = mammalcheck[]
            end
            mammalcheckRE = df_RE_richness.species[findall(df_RE_richness.POLYFID .== diet_polyfid)]
            if !isempty(mammalcheckRE)
                RERichness[i] = mammalcheckRE[]
            end
            raptorcheck = df_raptor_richness.species[findall(df_raptor_richness.POLYFID .== diet_polyfid)]
            if !isempty(mammalcheckRE)
                RaptorRichness[i] = raptorcheck[]
            end
        end

        if DietRichness[i] > 0
            # rarefy datasets and calculate diversity indices
            CS_vector = cumsum(counts_to_use)
            Slice_widths = CS_vector ./ CS_vector[end] # get pie slices for the diet

            RareSample = zeros(Int64, rarefy_to, rarefy_reps) # open up an emptry matrix
            for j = 1:rarefy_reps # loop over the number of rarefied samples you want to take
                LI = rand(rarefy_to) .< Slice_widths' # broadcast random samples across the pie slices of the diet
                RareSample[:,j] = findfirst.(isequal(1),eachrow(LI)) # find the first 1 in each sample - that's an index of diet type
            end
            Counts_rare = freqtable.(eachcol(RareSample)) # broadcasts a count summary of each rarefied sample diet
            Freqs_rare = eachrow(Counts_rare) ./ rarefy_to # broadcasts a frequency summary of each rarefied sample diet

            # here are the diet-level characteristics
            DietRichness_rare_reps = map(length, Counts_rare) # use map to find the lengths of the inner vectors, which are sample diets
            DietRichness_rare[i] = mean(DietRichness_rare_reps)

            # broadcasting not working for log, so use a loop to extract freqs and calculate entropy for each replicate sample
            ShannEnt_rare_reps = zeros(Float64,rarefy_reps)
                for k = 1:rarefy_reps
                    p = Freqs_rare[k,:][][]
                    logp = log.(p)
                    ShannEnt_rare_reps[k] = -sum(p .* logp)
                end
                ShannEnt_rare[i] = mean(ShannEnt_rare_reps)

        end
    end
end



    #= now run it through df_mean_mass to assign mean genus or family body masses to prey identified to those levels
    prey_mass_mean = df_mean_mass.BodyMassValue_mean
    prey_name_mean = df_mean_mass.Scientific

    for i = 1:length(unique_prey)
        prey_row = findall(prey_name_mean .== unique_prey[i]) # grab the ith prey name from the unique list
        mass_to_pass = prey_mass_mean[prey_row] # snag the mass of the prey in row i
        matches = findall(prey_sci_name .== unique_prey[i]) # find where that prey occurs in df_diet
        
        if length(mass_to_pass) > 0 # only do this if there is an actual value in for mass
            for j = 1:length(matches)
                prey_mass_diet[matches[j]] = mass_to_pass[1] # pass the value to every row for that species
            end
        end
    end

    =#

# build the barn owl specific diets overview
df_baow_overview = select(df_meta,:DataSet,:RaptorScientificName,:Continent,:Num_obs,:PreyPerPellet,:ecozone)
df_baow_overview[!, :ShannEnt_rare] .= ShannEnt_rare
df_baow_overview[!, :DietRichness_rare] .= DietRichness_rare
df_baow_overview[!, :LocalRichness] .= LocalRichness
df_baow_overview[!, :RERichness] .= RERichness
df_baow_overview[!, :RaptorRichness] .= RaptorRichness

df_baow_overview = df_baow_overview[occursin.("Tyto", df_baow_overview.RaptorScientificName), :] # reduce the data set to only rows that contain "Tyto"

CSV.write("BarnOwlMeta_100.csv", df_baow_overview)

# check on effect of rarefaction
f1 = scatter(DietRichness,ShannEnt)
f2 = scatter(DietRichness_rare,ShannEnt_rare)
plot(f1,f2)

# build the barn owl specific database of diets
df_baow = select(df_diets,:DataSet,:RaptorScientificName,:IDLevel,:PreyClass,:PreyOrder,:PreyScientificName,:CountInt) # start with a subset of the imported data columns
df_baow[!, :PreyMass] .= prey_mass_diet # add the prey matched body masses
df_baow[!, :PreyFamily] .= prey_family # add the prey matched body masses

df_baow = df_baow[occursin.("Tyto", df_baow.RaptorScientificName), :] # reduce the data set to only rows that contain "Tyto"

unique(df_baow.DataSet)
CSV.write("BarnOwlDiets_100.csv", df_baow)

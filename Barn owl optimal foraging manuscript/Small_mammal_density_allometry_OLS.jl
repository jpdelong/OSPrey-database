# this script processes the link between abundances of mammal prey and body size
# this is broken up into a Bayesian section and a GLM section, which will use different packages

##### specify folder and load data
dir = "C:/Users/93652672/OneDrive - University of Nebraska/Projects in progress/Barn_owl_global_diet/"
cd(dir)

# need these for all parts
using CSV
using DataFrames
using Statistics
using CategoricalArrays
using GLMakie
using GLM

using CairoMakie

colorgrad = cgrad(:roma, 15, categorical = true, scale = :exp)

# read in the data
df_density = CSV.read("Dataset_density2.csv",DataFrame) # read in density data
df_density = df_density[findall(df_density.Order .!= "NA"),:] # drop everything that is rows with just NA
#df_dens_rodents = df_density[findall(df_density.Order .== "Rodentia" .|| df_density.Order .== "Eulipotyphla" .|| df_density.Order .== "Lagomorpha" .|| df_density.Order .== "Dasyuromorphia" .|| df_density.Order .== "Didelphimorphia"),:] # drop everything that is rows with just NA

number_rows = size(df_density,1)
masses = zeros(Float64,number_rows)

df_bodymass = CSV.read("Dataset_predictions.csv",DataFrame) # read in density data

# litte loop to pull mammal body masses from the Elton database and match them to the density database
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

# add the body mass and logged density data to the main database
df_density[!, :BodyMass] .= masses
df_density[!, :LogMass] = log10.(df_density.BodyMass) # 
df_density[!, :LogDensity] = log10.(df_density.Density_km) # 
df_density = df_density[findall(df_density.BodyMass .!= 0),:] # drop rows with 0 body mass


# write a little function to make sure that the predicitons are floats
function setpredictiontypes(df_predict_y) 
    df_predict_y[!,:prediction] = convert.(Float64,df_predict_y[!,:prediction])
    df_predict_y[!,:lower] = convert.(Float64,df_predict_y[!,:lower])
    df_predict_y[!,:upper] = convert.(Float64,df_predict_y[!,:upper])
end

################################################################
######## set up to do density allometry with a OLS approach
################################################################

########### first do standard Damuth's rule ###############################

f_density = Figure()
ax_all = Axis(f_density[1, 1], xlabel = "log10 Body mass (g)", ylabel = "log10 Density (individuals/km2)") #,yscale = log10,xscale = log10
scatter!(ax_all,df_density.LogMass,df_density.LogDensity, color = colorgrad[8], alpha = 0.6)

lm_all = lm(@formula(LogDensity ~ LogMass), df_density)
allr2 = r2(lm_all)

# make model predictions and plot
new_x = DataFrame(LogMass = range(minimum(df_density.LogMass), maximum(df_density.LogMass), length=10))
df_predict_y = predict(lm_all,new_x, interval=:confidence, level = 0.95)
# ensure that all data types are floats with our above function
setpredictiontypes(df_predict_y)
band!(ax_all,new_x.LogMass,df_predict_y.lower,df_predict_y.upper; color = colorgrad[8])
rod = lines!(ax_all,new_x.LogMass, df_predict_y.prediction, linewidth = 2, color = :black)

text!(ax_all, 0.2, 4, text = "A")
############# now do it by order ########################

# add a panel to the figure
ax_order = Axis(f_density[2, 1], xlabel = "log10 Body mass (g)", ylabel = "log10 Density (individuals/km2)") #,yscale = log10,xscale = log10

# subset the data by the orders actually in the barn owl diet
subset_df = df_density

# tally up rows by order to check on sample sizes
sums_by_order = combine(groupby(subset_df, :Order), nrow)

prey_orders = ["Eulipotyphla","Rodentia","Didelphimorphia","Chiroptera","Lagomorpha","Carnivora","Dasyuromorphia","Macroscelidea","Diprotodontia","Afrosoricida","Primates","Peramelemorphia","Microbiotheria"]
missing_words = setdiff(subset_df.Order, prey_orders)
for j = 1:length(missing_words)
    subset_df = subset_df[findall(subset_df.Order .!= missing_words[j]),:] # drop rows with orders that don't occur in the diets
end

    subset_df = subset_df[findall(subset_df.Order .!= "Microbiotheria"),:] # drop rows, only two diet items and no densities
    subset_df = subset_df[findall(subset_df.Order .!= "Peramelemorphia"),:] # drop rows with 0 body mass

unique(subset_df.Order)

# fit the linear models
lm_1 = lm(@formula(LogDensity ~ LogMass + Order), subset_df)
mainr2 = r2(lm_1)

lm_2 = lm(@formula(LogDensity ~ LogMass * Order), subset_df)
mainr2 = r2(lm_2)

# plot all the points first, then come back to the lines
# rodents
    df_subset = subset_df[findall(subset_df.Order .== "Rodentia"),:]
    scatter!(ax_order,df_subset.LogMass,df_subset.LogDensity, color = ("White", 0.1), strokewidth = 0.5, strokecolor = colorgrad[1])
# rabbits
    df_subset = subset_df[findall(subset_df.Order .== "Lagomorpha"),:]
    scatter!(ax_order,df_subset.LogMass,df_subset.LogDensity, color = ("White", 0.1), strokewidth = 0.5, strokecolor = colorgrad[3])
# insectivores
    df_subset = subset_df[findall(subset_df.Order .== "Eulipotyphla"),:]
    scatter!(ax_order,df_subset.LogMass,df_subset.LogDensity, color = ("White", 0.1), strokewidth = 0.5, strokecolor = colorgrad[5])
# carnivorous marsupials
    df_subset = subset_df[findall(subset_df.Order .== "Dasyuromorphia"),:]
    scatter!(ax_order,df_subset.LogMass,df_subset.LogDensity, color = ("White", 0.1), strokewidth = 0.5, strokecolor = colorgrad[7])
# carnivora
    df_subset = subset_df[findall(subset_df.Order .== "Carnivora"),:]
    scatter!(ax_order,df_subset.LogMass,df_subset.LogDensity, color = ("White", 0.1), strokewidth = 0.5, strokecolor = colorgrad[9])
# primates
    df_subset = subset_df[findall(subset_df.Order .== "Primates"),:]
    scatter!(ax_order,df_subset.LogMass,df_subset.LogDensity, color = ("White", 0.1), strokewidth = 0.5, strokecolor = colorgrad[11])
# Diprotodontia
    df_subset = subset_df[findall(subset_df.Order .== "Diprotodontia"),:]
    scatter!(ax_order,df_subset.LogMass,df_subset.LogDensity, color = ("White", 0.1), strokewidth = 0.5, strokecolor = colorgrad[13])
# Didelphimorphia
    df_subset = subset_df[findall(subset_df.Order .== "Didelphimorphia"),:]
    scatter!(ax_order,df_subset.LogMass,df_subset.LogDensity, color = ("White", 0.1), strokewidth = 0.5, strokecolor = colorgrad[15])

# now get and plot the fitted lines
    # rodents
        df_subset = subset_df[findall(subset_df.Order .== "Rodentia"),:]
        new_x = DataFrame(LogMass = range(minimum(df_subset.LogMass), maximum(df_subset.LogMass), length=10))
        new_x[!, :Order] = vcat(fill.("Rodentia",10))
        df_predict_y = predict(lm_2,new_x, interval=:confidence, level = 0.95)
        # ensure that all data types are floats with our above function
        setpredictiontypes(df_predict_y)
        band!(ax_order,new_x.LogMass,df_predict_y.lower,df_predict_y.upper; color = colorgrad[1])
        rod = lines!(ax_order,new_x.LogMass, df_predict_y.prediction, linewidth = 2, color = :black)
    # rabbits
        df_subset = subset_df[findall(subset_df.Order .== "Lagomorpha"),:]
        new_x = DataFrame(LogMass = range(minimum(df_subset.LogMass), maximum(df_subset.LogMass), length=10))
        new_x[!, :Order] = vcat(fill.("Lagomorpha",10))
        df_predict_y = predict(lm_2,new_x, interval=:confidence, level = 0.95)
        # ensure that all data types are floats with our above function
        setpredictiontypes(df_predict_y)
        band!(ax_order,new_x.LogMass,df_predict_y.lower,df_predict_y.upper; color = colorgrad[3])
        lag = lines!(ax_order,new_x.LogMass, df_predict_y.prediction, linewidth = 2, color = :black)
    # plot the line for insectivores
        df_subset = subset_df[findall(subset_df.Order .== "Eulipotyphla"),:]
        new_x = DataFrame(LogMass = range(minimum(df_subset.LogMass), maximum(df_subset.LogMass), length=10))
        new_x[!, :Order] = vcat(fill.("Eulipotyphla",10))
        df_predict_y = predict(lm_2,new_x, interval=:confidence, level = 0.95)
        # ensure that all data types are floats with our above function
        setpredictiontypes(df_predict_y)
        band!(ax_order,new_x.LogMass,df_predict_y.lower,df_predict_y.upper; color = colorgrad[5])
        ins = lines!(ax_order,new_x.LogMass, df_predict_y.prediction, linewidth = 2, color = :black)
    # plot the line for carnivorous marsupials
        df_subset = subset_df[findall(subset_df.Order .== "Dasyuromorphia"),:]
        new_x = DataFrame(LogMass = range(minimum(df_subset.LogMass), maximum(df_subset.LogMass), length=10))
        new_x[!, :Order] = vcat(fill.("Dasyuromorphia",10))
        df_predict_y = predict(lm_2,new_x, interval=:confidence, level = 0.95)
        # ensure that all data types are floats with our above function
        setpredictiontypes(df_predict_y)
        band!(ax_order,new_x.LogMass,df_predict_y.lower,df_predict_y.upper; color = colorgrad[7])
        cm = lines!(ax_order,new_x.LogMass, df_predict_y.prediction, linewidth = 2, color = :black)
    # plot the line for carnivora
        df_subset = subset_df[findall(subset_df.Order .== "Carnivora"),:]
        new_x = DataFrame(LogMass = range(minimum(df_subset.LogMass), maximum(df_subset.LogMass), length=10))
        new_x[!, :Order] = vcat(fill.("Carnivora",10))
        df_predict_y = predict(lm_2,new_x, interval=:confidence, level = 0.95)
        # ensure that all data types are floats with our above function
        setpredictiontypes(df_predict_y)
        band!(ax_order,new_x.LogMass,df_predict_y.lower,df_predict_y.upper; color = colorgrad[9])
        cv = lines!(ax_order,new_x.LogMass, df_predict_y.prediction, linewidth = 2, color = :black)
    # plot the line for Primates
        df_subset = subset_df[findall(subset_df.Order .== "Primates"),:]
        new_x = DataFrame(LogMass = range(minimum(df_subset.LogMass), maximum(df_subset.LogMass), length=10))
        new_x[!, :Order] = vcat(fill.("Primates",10))
        df_predict_y = predict(lm_2,new_x, interval=:confidence, level = 0.95)
        # ensure that all data types are floats with our above function
        setpredictiontypes(df_predict_y)
        band!(ax_order,new_x.LogMass,df_predict_y.lower,df_predict_y.upper; color = colorgrad[11])
        pri = lines!(ax_order,new_x.LogMass, df_predict_y.prediction, linewidth = 2, color = :black)
    # plot the line for Diprotodontia
        df_subset = subset_df[findall(subset_df.Order .== "Diprotodontia"),:]
        new_x = DataFrame(LogMass = range(minimum(df_subset.LogMass), maximum(df_subset.LogMass), length=10))
        new_x[!, :Order] = vcat(fill.("Diprotodontia",10))
        df_predict_y = predict(lm_2,new_x, interval=:confidence, level = 0.95)
        # ensure that all data types are floats with our above function
        setpredictiontypes(df_predict_y)
        band!(ax_order,new_x.LogMass,df_predict_y.lower,df_predict_y.upper; color = colorgrad[13])
        dip = lines!(ax_order,new_x.LogMass, df_predict_y.prediction, linewidth = 2, color = :black)
    # plot the line for Didelphimorphia
        df_subset = subset_df[findall(subset_df.Order .== "Didelphimorphia"),:]
        new_x = DataFrame(LogMass = range(minimum(df_subset.LogMass), maximum(df_subset.LogMass), length=10))
        new_x[!, :Order] = vcat(fill.("Didelphimorphia",10))
        df_predict_y = predict(lm_2,new_x, interval=:confidence, level = 0.95)
        # ensure that all data types are floats with our above function
        setpredictiontypes(df_predict_y)
        band!(ax_order,new_x.LogMass,df_predict_y.lower,df_predict_y.upper; color = colorgrad[15])
        did = lines!(ax_order,new_x.LogMass, df_predict_y.prediction, linewidth = 2, color = :black)

text!(ax_order, 0.2, 4, text = "B")

f_density

Legend(f_density[2, 2],
    [PolyElement(color = colorgrad[1]),PolyElement(color = colorgrad[3]),PolyElement(color = colorgrad[5]),
    PolyElement(color = colorgrad[7]),PolyElement(color = colorgrad[9]),PolyElement(color = colorgrad[1]),
    PolyElement(color = colorgrad[13]),PolyElement(color = colorgrad[15])],
    ["Rodentia", "Lagomorphs", "Insectivores", "Carnivorous marsupials", "Carnivora", 
    "Primates", "Diprotodontia", "Didelphimorphia"])

    save("Mammal_density_allometry.pdf",f_density)


    #[rod, lag, ins, cm, cv, pri, dip, did],

################################################################
######## set up to do density allometry with a Bayesian approach
################################################################


#= Need to convert the "orders" column to categorical type
subset_df.group = categorical(subset_df.Order)
# Then create a new column of the underlying integer codes
subset_df.group_idx = levelcode.(subset_df.group)=#

#= pull out needed columns and turn them into matrices
LogMass = reshape(subset_df.LogMass, :, 1)
LogDensity = reshape(subset_df.LogDensity, :, 1)
OrderIndex = reshape(subset_df.group_idx, :, 1)=#

# add these packages
using Turing
using StatsPlots

################################################################
# first do standard Damuth's rule

#= pull out needed columns and turn them into matrices
LogMass = reshape(df_density.LogMass, :, 1)
LogDensity = reshape(df_density.LogDensity, :, 1)=#

# pull out needed columns and turn them into matrices
LogMass = reshape(df_density.LogMass, :, 1)
LogDensity = reshape(df_density.LogDensity, :, 1)

# Bayesian linear regression.
@model function Damuth_rule(x, y)
    # Set variance prior
    σ² ~ truncated(Normal(0, 100); lower=0)

    # Set intercept prior
    intercept ~ Normal(1, sqrt(3))

    # Set the priors on our coefficients
    β ~ Normal(-.75, 1)

    for i in 1:length(y)
        y[i] ~ Normal(intercept .+ x[i] .* β, σ²)
    end
end

model_Damuth = Damuth_rule(LogMass, LogDensity)
chain = sample(model_Damuth, NUTS(),2_000)
plot(chain)
chain_summary = summarystats(chain)

# pull out the parameters
fitted_int = median(chain[:intercept]) # grab the intercepts
fitted_beta_mass = median(chain[:β]) # grab the slope
xrange = LinRange(minimum(df_density.LogMass), maximum(df_density.LogMass), 10) # create an x range
fitted_line = fitted_int .+ fitted_beta_mass .* xrange # calculate the fitted y

# plot the basic allometry
p1 = scatter(df_density.LogMass,df_density.LogDensity, xlims = (0.5, 7), label="", markerstrokecolor="blue4",markercolor="blue4",alpha=0.2,
    legend=:outerright, xlabel="ln body mass", ylabel="ln density",size=(600,300)) # plot the data
plot!(p1, xrange, fitted_line, linewidth=2, label="Overall fit", legend=:outerright) # plot the fitted line over the data

#= detour to maybe make the plots in Makie
using GLMakie
f = Figure(size = (1000, 400))
    ax_Damuth = Axis(f[1, 1], title = "Standard allometry", ylabel = "ln density")
    ax_orders = Axis(f[2, 1], xlabel = "ln body mass")

e1 = scatter!(ax_Damuth, df_density.LogMass, df_density.LogDensity, color=:blue,
        markersize = 20)
e2 = series!(ax_Damuth, xrange, fitted_line, solid_color=col_57,label="Line 57",
        linewidth=4)
=#



# Bayesian linear regression.
@model function allometry_by_order(x, y, order)
    orders = length(unique(order))
    # Set variance prior
    σ² ~ truncated(Normal(0, 100); lower=0)

    # Set intercept prior
    intercept ~ filldist(Normal(0, sqrt(3)), orders)

    # Set the priors on our coefficients
    β ~ filldist(Normal(0, 10), orders)

    for i in 1:length(y)
        y[i] ~ Normal(intercept[order[i]] .+ x[i] .* β[order[i]], σ²)
    end

end

model_orders = allometry_by_order(LogMass, LogDensity, OrderIndex)
chain_orders = sample(model_orders, NUTS(),2_000)
plot(chain)
chain_summary = summarystats(chain)

howmanysamples = 10
int1_samp = zeros(Float64, howmanysamples)
int2_samp = zeros(Float64, howmanysamples)
int3_samp = zeros(Float64, howmanysamples)
int4_samp = zeros(Float64, howmanysamples)
int5_samp = zeros(Float64, howmanysamples)
int6_samp = zeros(Float64, howmanysamples)
int7_samp = zeros(Float64, howmanysamples)
int8_samp = zeros(Float64, howmanysamples)
int9_samp = zeros(Float64, howmanysamples)
int10_samp = zeros(Float64, howmanysamples)
int11_samp = zeros(Float64, howmanysamples)
int12_samp = zeros(Float64, howmanysamples)
β1_samp = zeros(Float64, howmanysamples)
β2_samp = zeros(Float64, howmanysamples)
β3_samp = zeros(Float64, howmanysamples)
β4_samp = zeros(Float64, howmanysamples)
β5_samp = zeros(Float64, howmanysamples)
β6_samp = zeros(Float64, howmanysamples)
β7_samp = zeros(Float64, howmanysamples)
β8_samp = zeros(Float64, howmanysamples)
β9_samp = zeros(Float64, howmanysamples)
β10_samp = zeros(Float64, howmanysamples)
β11_samp = zeros(Float64, howmanysamples)
β12_samp = zeros(Float64, howmanysamples)

# sample the posteriors of each parameter and store
for i in 1:howmanysamples # grab howmanysamples random samples for clarity
    # Get parameter values for the current sample
    int1_samp[i] = chain_orders[:"intercept[1]"][i]
    int2_samp[i] = chain_orders[:"intercept[2]"][i]
    int3_samp[i] = chain_orders[:"intercept[3]"][i]
    int4_samp[i] = chain_orders[:"intercept[4]"][i]
    int5_samp[i] = chain_orders[:"intercept[5]"][i]
    int6_samp[i] = chain_orders[:"intercept[6]"][i]
    int7_samp[i] = chain_orders[:"intercept[7]"][i]
    int8_samp[i] = chain_orders[:"intercept[8]"][i]
    int9_samp[i] = chain_orders[:"intercept[9]"][i]
    int10_samp[i] = chain_orders[:"intercept[10]"][i]
    int11_samp[i] = chain_orders[:"intercept[11]"][i]
    int12_samp[i] = chain_orders[:"intercept[12]"][i]
    β1_samp[i] = chain_orders[:"β[1]"][i] # grab the slope
    β2_samp[i] = chain_orders[:"β[2]"][i] # grab the slope
    β3_samp[i] = chain_orders[:"β[3]"][i] # grab the slope
    β4_samp[i] = chain_orders[:"β[4]"][i] # grab the slope
    β5_samp[i] = chain_orders[:"β[5]"][i] # grab the slope
    β6_samp[i] = chain_orders[:"β[6]"][i] # grab the slope
    β7_samp[i] = chain_orders[:"β[7]"][i] # grab the slope
    β8_samp[i] = chain_orders[:"β[8]"][i] # grab the slope
    β9_samp[i] = chain_orders[:"β[9]"][i] # grab the slope
    β10_samp[i] = chain_orders[:"β[10]"][i] # grab the slope
    β11_samp[i] = chain_orders[:"β[11]"][i] # grab the slope
    β12_samp[i] = chain_orders[:"β[12]"][i] # grab the slope

end

df_allometry_chain = DataFrame()
df_allometry_chain[!,:Intercept1] = int1_samp
df_allometry_chain[!,:Intercept2] = int2_samp
df_allometry_chain[!,:Intercept3] = int3_samp
df_allometry_chain[!,:Intercept4] = int4_samp
df_allometry_chain[!,:Intercept5] = int5_samp
df_allometry_chain[!,:Intercept6] = int6_samp
df_allometry_chain[!,:Intercept7] = int7_samp
df_allometry_chain[!,:Intercept8] = int8_samp
df_allometry_chain[!,:Intercept9] = int9_samp
df_allometry_chain[!,:Intercept10] = int10_samp
df_allometry_chain[!,:Intercept11] = int11_samp
df_allometry_chain[!,:Intercept12] = int12_samp
df_allometry_chain[!,:orders1] = β1_samp
df_allometry_chain[!,:orders2] = β2_samp
df_allometry_chain[!,:orders3] = β3_samp
df_allometry_chain[!,:orders4] = β4_samp
df_allometry_chain[!,:orders5] = β5_samp
df_allometry_chain[!,:orders6] = β6_samp
df_allometry_chain[!,:orders7] = β7_samp
df_allometry_chain[!,:orders8] = β8_samp
df_allometry_chain[!,:orders9] = β9_samp
df_allometry_chain[!,:orders10] = β10_samp
df_allometry_chain[!,:orders11] = β11_samp
df_allometry_chain[!,:orders12] = β12_samp

CSV.write("Density_allometry_chain.csv", df_allometry_chain)

# now plot the data again but just the orders included here
p2 = scatter(subset_df.LogMass,subset_df.LogDensity,markerstrokecolor="blue4",markercolor="blue4",alpha=0.2,
    label="",xlims = (0.5, 7),xlabel="ln body mass", ylabel="ln density",size=(600,300)) # plot the data

for i = 1:12
    subset_mass = filter(row -> row.group_idx in i, subset_df)
    xrange = LinRange(minimum(subset_mass.LogMass), maximum(subset_mass.LogMass), 10) # create an x range

    column_name = string("β[",i,"]")
    fitted_beta_mass = median(chain_orders[column_name]) # grab the slope

    column_name = string("intercept[",i,"]")
    fitted_int = median(chain_orders[column_name]) # grab the intercept

    fitted_line = fitted_int .+ fitted_beta_mass .* xrange # calculate the fitted y

    plot!(p2, xrange, fitted_line, label=sums_by_order.Order[i], legendposition=:outerright, linewidth=2) # plot the fitted line over the data
end

p2

allometry_figure = plot(p1,p2, layout=grid(2,1), size=(700,500),dpi=600) 
savefig(allometry_figure,"Mammal_allometry.png")


################################################################
# older efforts
################################################################

#=segments = shortest_path(
    log10.(df_dens_rodents.BodyMass), 
    log10.(df_dens_rodents.Density_km); 
    min_segment_length = 0.5, 
    fit_threshold = 0.5,
    fit_function = :r2,
    overlap = true,
)=#
#=
using GLMakie
using LinearSegmentation
using GLM


# trying to make this work but so far not --- filter(row -> all(x -> !(x .= "Rodentia"), row), df_density)


=#


#= an attempt using TuringGLM
######################### try simple mass allometry first
f = @formula LogDensity ~ LogMass # define the model formula
model1 = turing_model(f, df_density) # pair it with the dataset
chain = sample(model1, NUTS(),2_000) # call the sampler
plot(chain) # plot the posteriors and chains to check

fitted_int = median(chain[:α]) # grab the intercepts
fitted_beta_mass = median(chain[:"β[1]"]) # grab the slope

xrange = LinRange(minimum(df_density.LogMass), maximum(df_density.LogMass), 10) # create an x range
fitted_line = fitted_int .+ fitted_beta_mass .* xrange # calculate the fitted y

p1 = scatter(df_density.LogMass,df_density.LogDensity) # plot the data
plot!(p1, xrange, fitted_line) # plot the fitted line over the data

# looks good, shockingly close to class Damuth's law (-3/4 power scaling)
# but this is way to noisy to be useful for predicting densities
# need to add mammal order as a predictor

######################### now add order as a predictor
f = @formula LogDensity ~ (1 / Order | LogMass)
model2_orders = turing_model(f, df_density)
chain = sample(linear_model, NUTS(),2_000)
plot(chain)
chain_summary = summarystats(chain)

fitted_int = median(chain[:α]) # grab the intercepts
fitted_beta_mass = median(chain[:"β[3]"]) # grab the slope

fitted_line = fitted_int .+ fitted_beta_mass .* xrange # calculate the fitted y


p2 = scatter(df_density.LogMass,df_density.LogDensity) # plot the data

for i = 1:3
    column_name = string("β[",i,"]")
    fitted_beta_mass = median(chain[column_name]) # grab the slope
    fitted_line = fitted_int .+ fitted_beta_mass .* xrange # calculate the fitted y

    plot!(p2, xrange, fitted_line) # plot the fitted line over the data


end
p2

orders = unique(df_density.Order)
sort!(orders)

# Convert to a CategoricalVector
cat_vec = categorical(orders)

# Get the underlying integer codes
encoded_vec = levelcode.(cat_vec)



howmanysamples = 10
α_samp = zeros(Float64, howmanysamples)
β1_samp = zeros(Float64, howmanysamples)
β2_samp = zeros(Float64, howmanysamples)
β3_samp = zeros(Float64, howmanysamples)
β4_samp = zeros(Float64, howmanysamples)
β5_samp = zeros(Float64, howmanysamples)
β6_samp = zeros(Float64, howmanysamples)
β7_samp = zeros(Float64, howmanysamples)
β8_samp = zeros(Float64, howmanysamples)
β9_samp = zeros(Float64, howmanysamples)
β10_samp = zeros(Float64, howmanysamples)
β11_samp = zeros(Float64, howmanysamples)
β12_samp = zeros(Float64, howmanysamples)
β13_samp = zeros(Float64, howmanysamples)
β14_samp = zeros(Float64, howmanysamples)
β15_samp = zeros(Float64, howmanysamples)
β16_samp = zeros(Float64, howmanysamples)
β17_samp = zeros(Float64, howmanysamples)
β18_samp = zeros(Float64, howmanysamples)
β19_samp = zeros(Float64, howmanysamples)
β20_samp = zeros(Float64, howmanysamples)

# sample the posteriors of each parameter and store
for i in 1:howmanysamples # grab howmanysamples random samples for clarity
    # Get parameter values for the current sample
    α_samp[i] = chain[:α][i]
    β1_samp[i] = chain[:"β[1]"][i] # grab the slope
    β2_samp[i] = chain[:"β[2]"][i] # grab the slope
    β3_samp[i] = chain[:"β[3]"][i] # grab the slope
    β4_samp[i] = chain[:"β[4]"][i] # grab the slope
    β5_samp[i] = chain[:"β[5]"][i] # grab the slope
    β6_samp[i] = chain[:"β[6]"][i] # grab the slope
    β7_samp[i] = chain[:"β[7]"][i] # grab the slope
    β8_samp[i] = chain[:"β[8]"][i] # grab the slope
    β9_samp[i] = chain[:"β[9]"][i] # grab the slope
    β10_samp[i] = chain[:"β[10]"][i] # grab the slope
    β11_samp[i] = chain[:"β[11]"][i] # grab the slope
    β12_samp[i] = chain[:"β[12]"][i] # grab the slope
    β13_samp[i] = chain[:"β[13]"][i] # grab the slope
    β14_samp[i] = chain[:"β[14]"][i] # grab the slope
    β15_samp[i] = chain[:"β[15]"][i] # grab the slope
    β16_samp[i] = chain[:"β[16]"][i] # grab the slope
    β17_samp[i] = chain[:"β[17]"][i] # grab the slope
    β18_samp[i] = chain[:"β[18]"][i] # grab the slope
    β19_samp[i] = chain[:"β[19]"][i] # grab the slope
    β20_samp[i] = chain[:"β[20]"][i] # grab the slope
end

df_allometry_chain = DataFrame()
df_allometry_chain[!,:Intercept] = α_samp
df_allometry_chain[!,orders[1]] = β1_samp
df_allometry_chain[!,orders[2]] = β2_samp
df_allometry_chain[!,orders[3]] = β3_samp
df_allometry_chain[!,orders[4]] = β4_samp
df_allometry_chain[!,orders[5]] = β5_samp
df_allometry_chain[!,orders[6]] = β6_samp
df_allometry_chain[!,orders[7]] = β7_samp
df_allometry_chain[!,orders[8]] = β8_samp
df_allometry_chain[!,orders[9]] = β9_samp
df_allometry_chain[!,orders[10]] = β10_samp
df_allometry_chain[!,orders[11]] = β11_samp
df_allometry_chain[!,orders[12]] = β12_samp
df_allometry_chain[!,orders[13]] = β13_samp
df_allometry_chain[!,orders[14]] = β14_samp
df_allometry_chain[!,orders[15]] = β15_samp
df_allometry_chain[!,orders[16]] = β16_samp
df_allometry_chain[!,orders[17]] = β17_samp
df_allometry_chain[!,orders[18]] = β18_samp
df_allometry_chain[!,orders[19]] = β19_samp
df_allometry_chain[!,orders[20]] = β20_samp

CSV.write("Density_allometry_chain.csv", df_allometry_chain)
=#

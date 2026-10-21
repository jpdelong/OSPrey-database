### some extra code







# with yet another subset, try to predict intake rate
intake_model1 = lm(@formula(NE_intake ~ numb_prey_spp + RERichness + RaptorRichness + FreqBirds), df_meta_subset)
intake_model1 = lm(@formula(NE_intake ~ RERichness + RaptorRichness + FreqBirds), df_meta_subset)
intake_model1 = lm(@formula(NE_intake ~ RERichness + FreqBirds), df_meta_subset)
intake_model1 = lm(@formula(NE_intake ~ FreqBirds), df_meta_subset)




density_model1 = lm(@formula(dens_slopes ~ NE_intake + numb_prey_spp + RERichness + RaptorRichness + FreqBirds), df_meta_subset)
density_model1 = lm(@formula(dens_slopes ~ NE_intake + numb_prey_spp + RERichness + RaptorRichness), df_meta_subset)
density_model1 = lm(@formula(dens_slopes ~ sum_R + NE_intake + numb_prey_spp + RERichness), df_meta_subset)
density_model1 = lm(@formula(dens_slopes ~ sum_R + NE_intake + numb_prey_spp), df_meta_subset)
density_model1 = lm(@formula(dens_slopes ~ sum_R + numb_prey_spp), df_meta_subset)
density_model1 = lm(@formula(dens_slopes ~ sum_R * numb_prey_spp), df_meta_subset)

density_model1 = lm(@formula(dens_slopes ~ numb_prey_spp + RERichness + RaptorRichness), df_meta_subset)







model_2 = fit(MixedModel, @formula(dens_slopes ~ numb_prey_spp + RERichness + RaptorRichness + (1|ecozone)), df_meta_subset)


# check for some correlations among predictors
density_model1 = lm(@formula(sum_R ~ NE_intake), df_meta)
density_model1 = lm(@formula(sum_R ~ RERichness), df_meta)
density_model1 = lm(@formula(NE_intake ~ RERichness), df_meta)




density_model1 = lm(@formula(dens_slopes ~ NE_intake), df_meta)


r2(lmm1)


# considered also using sum_R, but this is highly correlated with numb_prey_spp
# obviously because sum_R sums over the prey species, so don't use it
corrcheck1 = lm(@formula(sum_R ~ numb_prey_spp), df_meta_subset)


    # for some reason the predictions data type is coming out weird, so change it
    df_predict_y[!,:prediction] = convert.(Float64,df_predict_y[!,:prediction])
    df_predict_y[!,:lower] = convert.(Float64,df_predict_y[!,:lower])
    df_predict_y[!,:upper] = convert.(Float64,df_predict_y[!,:upper])
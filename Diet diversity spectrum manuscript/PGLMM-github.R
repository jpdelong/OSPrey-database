################################################################################
##
##  Title: R script to prepare phylogenetic, diet, and trait data, and 
##         implement PGLMM models
##  Author: Frank A. La Sorte <fal42@cornell.edu>
##  Date: 11/4/2024
##  Publication: DeLong, J.P., Coblentz, K.E., La Sorte, F.A., & Uiterwaal, S.F.
##               (2024) The global diet diversity spectrum in avian apex 
##               predators. Proceedings of the Royal Society B: Biological 
##               Sciences.
##
################################################################################


## load R libraries
library(phyr)
library(ape)


##-----------------------
##
## data preparation
##

## load phylogenetic data 
tre <- ape::read.nexus("sumtrees-all.nex")
spp <- sort(tre$tip.label)

## load raptor diet data
dat <- read.table("Tallbutinsects.txt", sep=",", as.is=TRUE, header=TRUE)
dat$sp <- gsub(" ", "_", dat$Species)
spp2 <- sort(unique(dat$sp))
spp3 <- spp[!spp %in% spp2]

## update raptor diet taxonomy to match phylogeny
dat$sp <- ifelse(dat$sp=="Aquila_fasciata", "Aquila_fasciatus", dat$sp)
dat$sp <- ifelse(dat$sp=="Bubo_scandiacus", "Bubo_scandiaca", dat$sp)
dat$sp <- ifelse(dat$sp=="Buteogallus_coronatus", "Harpyhaliaetus_coronatus", dat$sp)
dat$sp <- ifelse(dat$sp=="Circus_hudsonius", "Circus_cyaneus", dat$sp)
dat$sp <- ifelse(dat$sp=="Clanga_clanga", "Aquila_clanga", dat$sp)
dat$sp <- ifelse(dat$sp=="Geranoaetus_polyosoma", "Buteo_polyosoma", dat$sp)
dat$sp <- ifelse(dat$sp=="Glaucidium_nana", "Glaucidium_brasilianum", dat$sp)
dat$sp <- ifelse(dat$sp=="Ictinaetus_malaiensis", "Ictinaetus_malayensis", dat$sp)
dat$sp <- ifelse(dat$sp=="Rupornis_magnirostris", "Buteo_magnirostris", dat$sp)

spp2 <- sort(unique(dat$sp))
spp3 <- spp[!spp %in% spp2]

## trim phylogeny to only include raptor species
tre2 <- drop.tip(tre, spp3)

## load AVONET trait data
tra <- read.csv("Supplementary dataset BirdLife raptor.csv")
tra$sp <- gsub(" ", "_", tra$Species1)

tra2 <- data.frame(sp=sort(unique(dat$sp)))
tra2 <- merge(tra2, tra)

## update AVONET taxonomy to match raptor diet taxonomy
tra$sp <- ifelse(tra$sp=="Clanga_clanga", "Aquila_clanga", tra$sp)
tra$sp <- ifelse(tra$sp=="Aquila_fasciata", "Aquila_fasciatus", tra$sp)
tra$sp <- ifelse(tra$sp=="Bubo_scandiacus", "Bubo_scandiaca", tra$sp)
tra$sp <- ifelse(tra$sp=="Rupornis_magnirostris", "Buteo_magnirostris", tra$sp)
tra$sp <- ifelse(tra$sp=="Geranoaetus_polyosoma", "Buteo_polyosoma", tra$sp)
tra$sp <- ifelse(tra$sp=="Buteogallus_coronatus", "Harpyhaliaetus_coronatus", tra$sp)
tra$sp <- ifelse(tra$sp=="Ictinaetus_malaiensis", "Ictinaetus_malayensis", tra$sp)

tra2 <- data.frame(sp=sort(unique(dat$sp)))
tra2 <- merge(tra2, tra)


##--------------------
##
## PGLMMs
##

## define factor variables
dat$MainDiet <- factor(dat$MainDiet, levels=c("Mix", "Birds", "Mamms"))
dat$Biome <- as.factor(dat$Biome)

##
## local prey diversity (Diversity_PCA1) and richness (rare_NT)
##
mdl <- pglmm(rare_NT ~ Diversity_PCA1 + (1|sp__) + (1|Biome), 
             data=dat,
             cov_ranef = list(sp = tre2))

##
## local prey diversity (Diversity_PCA1) and entropy (rare_SE) 
## 
mdl <- pglmm(rare_SE ~ Diversity_PCA1 + rare_NT + (1|sp__) + (1|Biome), 
             data=dat,
             cov_ranef = list(sp = tre2))

##
## raptor body size (Mass) and richness (rare_NT)
##
dat2 <- merge(dat[,c("sp","rare_NT","Biome")], tra2[,c("sp","Mass")])
mdl <- pglmm(rare_NT ~ Mass + (1|sp__) + (1|Biome), 
             data=dat2,
             cov_ranef = list(sp = tre2))

##
## raptor body size (Mass) and entropy (rare_SE) 
##
dat2 <- merge(dat[,c("sp","rare_SE","rare_NT","Biome")], tra2[,c("sp","Mass")])
mdl <- pglmm(rare_SE ~ Mass + rare_NT + (1|sp__) + (1|Biome), 
             data=dat2,
             cov_ranef = list(sp = tre2))

##
## evolutionary history (FairProp) and richness (rare_NT)
##
mdl <- pglmm(rare_NT ~ FairProp + (1|sp__) + (1|Biome), 
             data=dat,
             cov_ranef = list(sp = tre2))

##
## evolutionary history (FairProp) and entropy (rare_SE)
## 
mdl <- pglmm(rare_SE ~ FairProp + rare_NT + (1|sp__) + (1|Biome), 
             data=dat,
             cov_ranef = list(sp = tre2))


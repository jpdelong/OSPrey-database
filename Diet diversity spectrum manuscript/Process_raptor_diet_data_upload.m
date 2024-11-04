clear; clc;

[num words] = xlsread('OSPreydatabase_11_13_23.xlsx','Diets'); % read in diet data

data_sets = num(:,1);
prey_counts = num(:,9);
prey_id_level = words(2:length(data_sets)+1,4);
prey_class = words(2:length(data_sets)+1,5);
prey_order = words(2:length(data_sets)+1,6);
prey_name = words(2:length(data_sets)+1,8);

% remove likely carrion from all diets
% identified as being artiodactyla
data_sets(ismember(prey_order,'Artiodactyla')) = [];
prey_counts(ismember(prey_order,'Artiodactyla')) = [];
prey_class(ismember(prey_order,'Artiodactyla')) = [];
prey_name(ismember(prey_order,'Artiodactyla')) = [];
prey_order(ismember(prey_order,'Artiodactyla')) = [];
prey_mass_diet = nan(size(data_sets)); % create an empty matrix to fill later

[num words] = xlsread('OSPreydatabase_11_13_23.xlsx','Meta data'); % read in dataset data
data_id = num(:,1);
latitude = num(:,11);
%latitude = abs(latitude);
longitude = num(:,12);
raptor_mass = num(:,35);
num_obs = num(:,36);
order = words(2:length(raptor_mass)+1,4);
family = words(2:length(raptor_mass)+1,5);
genus = words(2:length(raptor_mass)+1,6);
species = words(2:length(raptor_mass)+1,7);
season = words(2:length(raptor_mass)+1,26);
habitats = words(2:length(raptor_mass)+1,14:25);
num_habitats = nan(length(raptor_mass),1); % create an empty matrix to fill later
for i = 1:length(raptor_mass)
    num_habitats(i) = sum(ismember(habitats(i,:),'X'));
end

pred_range_size = nan(length(raptor_mass),1); % create an empty matrix to fill later
pred_equal_split = nan(length(raptor_mass),1); % create an empty matrix to fill later
pred_fair_prop = nan(length(raptor_mass),1); % create an empty matrix to fill later

[orders num_records] = grpstats(raptor_mass,species,{'gname','numel'});
ds = table(orders,num_records);
sortrows(ds,2,'descend');

%% pass in bird masses

[num text] = xlsread('EltonTraits_birds.xlsx','Sheet1'); % read in treatment data
prey_mass = num(:,36);
prey_sci_name = text(2:length(prey_mass)+1,8);
unique_prey = unique(prey_sci_name);

for i = 1:length(unique_prey)
    prey_row = find(strcmp(unique_prey(i),prey_sci_name));    
    mass_to_pass = prey_mass(prey_row);
    matches = find(strcmp(unique_prey(i),prey_name));
    
    if isempty(mass_to_pass) == 0
        for j = 1:length(matches)
            prey_mass_diet(matches(j)) = mass_to_pass;
        end
    end
    
end

%% pass in mammal masses

[num text] = xlsread('EltonTraits_mammals.xlsx','Sheet1'); % read in treatment data
prey_mass = num(:,25);
prey_sci_name = text(2:length(prey_mass)+1,4);
unique_prey = unique(prey_sci_name);

for i = 1:length(unique_prey)
    prey_row = find(strcmp(unique_prey(i),prey_sci_name));    
    mass_to_pass = prey_mass(prey_row);
    matches = find(strcmp(unique_prey(i),prey_name));
    
    if isempty(mass_to_pass) == 0
        for j = 1:length(matches)
            prey_mass_diet(matches(j)) = mass_to_pass;
        end
    end
    
end

%% pass in range sizes

[num words] = xlsread('AVONET-raptors.xlsx','AVONET-raptors'); % read in treatment data
range_size = num(:,36);
pred_sci_name = words(2:length(range_size)+1,2);
unique_pred = unique(pred_sci_name);

for i = 1:length(unique_pred)
    pred_row = find(strcmp(unique_pred(i),pred_sci_name));    
    range_to_pass = range_size(pred_row);
    matches = find(strcmp(unique_pred(i),species));
    
    if isempty(range_size) == 0
        for j = 1:length(matches)
            pred_range_size(matches(j)) = range_to_pass;
        end
    end
    
end

%% pass in evolutionary distinctiveness

[num words] = xlsread('evolutionary-distinctiveness.xlsx','evolutionary-distinctiveness'); % read in treatment data
equal_split = num(:,1);
fair_proportion = num(:,2);
pred_sci_name = words(2:length(equal_split)+1,1);
unique_pred = unique(pred_sci_name);

for i = 1:length(unique_pred)
    pred_row = find(strcmp(unique_pred(i),pred_sci_name));
    eq_to_pass = equal_split(pred_row);
    fp_to_pass = fair_proportion(pred_row);
    matches = find(strcmp(unique_pred(i),species));
    
    if isempty(equal_split) == 0
        for j = 1:length(matches)
            pred_equal_split(matches(j)) = eq_to_pass;
            pred_fair_prop(matches(j)) = fp_to_pass;
        end
    end
    
end

%% pass in prey species richness

BB_richness = csvread('bird-breeding-richness.csv',1,0);
NBB_richness = csvread('bird-nonbreeding-richness.csv',1,0);
M_richness = csvread('mammal-richness.csv',1,0);
rod_richness = csvread('rodentia-richness.csv',1,0);

strig_BB_richness = csvread('Strigiformes-breeding-richness.csv',1,0);
strig_NBB_richness = csvread('Strigiformes-nonbreeding-richness.csv',1,0);
acc_BB_richness = csvread('Accipitriformes-breeding-richness.csv',1,0);
acc_NBB_richness = csvread('Accipitriformes-nonbreeding-richness.csv',1,0);

[M words] = xlsread('POLYFID.xlsx','POLYFID'); % read in treatment data
polyfid_data = M(:,2);

richness_bb = NaN(max(data_sets),1); % open empty vector
richness_nbb = NaN(max(data_sets),1); % open empty vector
richness_ma = NaN(max(data_sets),1); % open empty vector
richness_strig_BB = NaN(max(data_sets),1); % open empty vector
richness_strig_NBB = NaN(max(data_sets),1); % open empty vector
richness_acc_BB = NaN(max(data_sets),1); % open empty vector
richness_acc_NBB = NaN(max(data_sets),1); % open empty vector
richness_rodent = NaN(max(data_sets),1); % open empty vector

for i = 1:length(polyfid_data) % one pass for every row in polyfid
        polyfid_of_next_diet = polyfid_data(i); % identify the next polygon
        
        test_anything_there = find(BB_richness(:,1)==polyfid_of_next_diet); % not all hexagons have breeding birds, so first check that
        if isempty(test_anything_there) == 0 % if info is available
            richness_bb(M(i,1)) = BB_richness(test_anything_there,2); % go ahead and pass that info into the column
        end
        
        test_anything_there = find(NBB_richness(:,1)==polyfid_of_next_diet); % not all hexagons have non breeding birds
        if isempty(test_anything_there) == 0 % if info is available
            richness_nbb(M(i,1)) = NBB_richness(test_anything_there,2); % go ahead and pass tha info into the column
        end
        
        test_anything_there = find(M_richness(:,1)==polyfid_of_next_diet); % not all hexagons have mammal info
        if isempty(test_anything_there) == 0 % if info is available
            richness_ma(M(i,1)) = M_richness(test_anything_there,2); % go ahead and pass tha info into the column    
        end
        
        test_anything_there = find(strig_BB_richness(:,1)==polyfid_of_next_diet); % not all hexagons have mammal info
        if isempty(test_anything_there) == 0 % if info is available
            richness_strig_BB(M(i,1)) = strig_BB_richness(test_anything_there,2); % go ahead and pass tha info into the column    
        end
        
        test_anything_there = find(strig_NBB_richness(:,1)==polyfid_of_next_diet); % not all hexagons have mammal info
        if isempty(test_anything_there) == 0 % if info is available
            richness_strig_NBB(M(i,1)) = strig_NBB_richness(test_anything_there,2); % go ahead and pass tha info into the column    
        end
        
        test_anything_there = find(acc_BB_richness(:,1)==polyfid_of_next_diet); % not all hexagons have mammal info
        if isempty(test_anything_there) == 0 % if info is available
            richness_acc_BB(M(i,1)) = acc_BB_richness(test_anything_there,2); % go ahead and pass tha info into the column    
        end
        
        test_anything_there = find(acc_NBB_richness(:,1)==polyfid_of_next_diet); % not all hexagons have mammal info
        if isempty(test_anything_there) == 0 % if info is available
            richness_acc_NBB(M(i,1)) = acc_NBB_richness(test_anything_there,2); % go ahead and pass tha info into the column    
        end
        
        test_anything_there = find(rod_richness(:,1)==polyfid_of_next_diet); % not all hexagons have mammal info
        if isempty(test_anything_there) == 0 % if info is available
            richness_rodent(M(i,1)) = rod_richness(test_anything_there,2); % go ahead and pass tha info into the column    
        end
        
end

%%
% 
% figure(5);clf(5);
% % histogram(freqs_to_use);shg;
% % pd = fitdist(freqs_to_use,'Lognormal')
% 
% subplot(1,2,1);
% histfit(freqs_to_use,20,'Lognormal')
% title('Lognormal');
% xlabel('freq');
% xlim([0 1]);
% 
% subplot(1,2,2);
% histfit(log(freqs_to_use),20,'Normal')
% title('Normal');
% xlabel('log(freq)');
% %xlim([0 1]);



%% run through each data set and calculate specialization and other metrics
% plot curves along the way

figure(1);clf(1);
subplot(211);
    hold on; box on;

for i = 1:max(data_sets)
    % pull out data
    prey_to_use = prey_name(data_sets == i);
    counts_to_use = prey_counts(data_sets == i);
    freqs_to_use = counts_to_use./(sum(counts_to_use));
    class_to_use = prey_class(data_sets == i);
    mass_to_use = prey_mass_diet(data_sets == i);
    prey_id_level_to_use = prey_id_level(data_sets == i);
    most_freq(i) = max(freqs_to_use);
    mass_most_freq(i) = nanmean(mass_to_use(find(most_freq(i) == freqs_to_use)));
    %prey_most_freq(i) = prey_to_use(find(most_freq(i) == freqs_to_use));
        most_common = prey_id_level(find(most_freq(i) == freqs_to_use));
    id_most_freq(i) = most_common(1);
    freqs_mammals(i) = sum(freqs_to_use(ismember(class_to_use,'Mammalia')));
    freqs_birds(i) = sum(freqs_to_use(ismember(class_to_use,'Aves')));
    freqs_ins(i) = sum(freqs_to_use(ismember(class_to_use,'Insecta')));
    
    % rarefy datasets to 20 and estimate MF
    rarefy_to = 100;
    CS_vector = cumsum(counts_to_use);
    Slice_widths = CS_vector./CS_vector(end);
    for k = 1:100
        LI = rand(1,rarefy_to) < Slice_widths;
        for q = 1:rarefy_to
            Event_index(q) = find(LI(:,q),1,'first');
        end
        rarefied_loop_MF(k) = sum(Event_index==mode(Event_index))/rarefy_to;
        rarefied_loop_NT(k) = length(unique(Event_index));
        % get freqs of rarefied sample
        tbl = crosstab(Event_index);
        rare_freqs = tbl./sum(tbl);
        rarefied_loops_SE(k) = -sum(rare_freqs.*log(rare_freqs));
    end
    rarefied_MF(i) = max(rarefied_loop_MF);
    rarefied_NT(i) = mean(rarefied_loop_NT);
    rarefied_SE(i) = mean(rarefied_loops_SE);
    
    % generate a rank abundance distribution
    X = [counts_to_use freqs_to_use];
    X_sorted = sortrows(X,1,'descend');
    rank = 1:length(counts_to_use);
    sorted_freqs = X_sorted(:,2);    

    % set up and run fitting for net specialization
    ft = fittype( 'exp1' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.StartPoint = [0.2 -0.2];
    if length(counts_to_use) > 2
        [fitresult, gof] = fit( rank', sorted_freqs, ft, opts );
        coeffs = coeffvalues(fitresult);
        specialization(i) = coeffs(2);
        y_intercept(i) = coeffs(1);
        rsquares(i) = gof.rsquare;
    
%     h = plot(rank,sorted_freqs,'o');
%     color = get(h, 'Color');
%     g = plot(fitresult,'-');
%     set(g, 'Color',color);
    
        figure(1);
        x = 1:0.1:30;
        y = y_intercept(i).*exp(specialization(i).*x);
        plot(x,y,'-k');    
    
        % estimate sd of a lognormal distribution fit to the freqs
        pd = fitdist(freqs_to_use,'Lognormal');
        LN_sd(i) = pd.sigma;
        LN_mean(i) = pd.mu;
        
        % calculate sp50
        cdf_sorted = cumsum(sorted_freqs);
        if cdf_sorted(1) < 0.5
            cdf_steps = find(cdf_sorted > 0.5);    
            xvals = [cdf_steps(1)-1 cdf_steps(1)];
            yvals = cdf_sorted(cdf_steps(1)-1:cdf_steps(1));
            fitobject = fit(xvals',yvals,'poly1');
            sp50(i) = (0.5 - fitobject.p2)/fitobject.p1;
        else
            sp50(i) = 1;
        end
    
    else
        specialization(i) = NaN;
        y_intercept(i) = NaN;
        LN_sd(i) = NaN;
        LN_mean(i) = NaN;
        sp50(i) = 1;
    end
    
    % calculate gini coefficient
    [g(i),~,~] = gini(rank,sorted_freqs);
    
    % calculate Shannon entropy
    SE(i) = -sum(freqs_to_use.*log(freqs_to_use));
        
    % add up number of prey types in each data set
    numtypes(i) = length(sorted_freqs);
    
    % calculate range of prey from smallest to largest
    MassRange(i) = nanmean(max(mass_to_use))-nanmean(min(mass_to_use));
    MSP(i) = min(mass_to_use);
    MLP(i) = max(mass_to_use);
    
    
    % pull out the frequency of unidentified birds and mammals
    temp = freqs_to_use(ismember(prey_to_use,'Aves')); % check for unidentified birds
    if temp > 0
        freqs_unid_bird(i) = sum(temp);
    else
        freqs_unid_bird(i) = 0;
    end
    temp = freqs_to_use(ismember(prey_to_use,'Mammalia')); % check for unidentified mammals
    if temp > 0
        freqs_unid_mamm(i) = sum(temp);
    else
        freqs_unid_mamm(i) = 0;
    end

    
    % calculate the expected proportion for each prey type
    if sum(~isnan(mass_to_use)) > 5
    for j = 1:length(counts_to_use(~isnan(mass_to_use)))
        mass_to_use_not_nan = mass_to_use(~isnan(mass_to_use));
        counts_to_use_not_nan = counts_to_use(~isnan(mass_to_use));
        freq_to_use_not_nan = counts_to_use_not_nan./(sum(counts_to_use_not_nan));
        prop_expected(j) = (mass_to_use_not_nan(j)^-1)/(sum(mass_to_use_not_nan.^-1));
    end
        max_overselected_size(i) = mass_to_use_not_nan(find(max(prop_expected'-freq_to_use_not_nan)));
        summed_deviance(i) = sum(abs(prop_expected'-freq_to_use_not_nan));
        Z = [mass_to_use_not_nan, freq_to_use_not_nan, prop_expected'];
        Z_sorted = sortrows(Z,1,'ascend');
%         figure(7);
%         %subplot(10,1,i);
%         hold on; box on;
%         plot(Z_sorted(:,1),Z_sorted(:,2),'-k');
%         plot(Z_sorted(:,1),Z_sorted(:,3),'-b');
        clear prop_expected
    else
        max_overselected_size(i) = NaN;
        summed_deviance(i) = NaN;
    end
end

%%
figure(1);
xlabel('Rank of abundance','FontSize',12);
ylabel('Frequency of occurrence','FontSize',12);
xlim([1 20]);
ylim([0 1]);

subplot(212);
    hist(abs(specialization),8);
    xlabel('Net specialization','FontSize',12);
    ylabel('Frequency','FontSize',12);

%% read in climate data
    
[num words] = xlsread('bioclim-era5-variables.xlsx','bioclim-era5-variables'); % read in diet data bioclim-era5-variables
MeanTemp = num(:,2);
Seasonality = num(:,3);
AnnPrecip = num(:,4);
PrecipSeason = num(:,5);

%% read in biome data
    
[num words] = xlsread('biome.xlsx','biome'); % read in diet data bioclim-era5-variables
biome_index = num(:,2);

%% assemble and anaylize data table

raptor_ds = table(data_id,order,family,genus,species,latitude,longitude,num_obs,numtypes',rarefied_NT',-specialization',y_intercept',most_freq',rarefied_MF',sp50',g',SE',rarefied_SE',log(raptor_mass),id_most_freq',log(mass_most_freq)',log(MSP)',log(MLP)',MassRange',...
    richness_bb,richness_ma,richness_strig_BB,richness_strig_NBB,richness_acc_BB,richness_acc_NBB,...
    num_habitats,pred_range_size,freqs_birds',freqs_mammals',freqs_ins',freqs_unid_bird',freqs_unid_mamm',...
    max_overselected_size',summed_deviance',season,MeanTemp,Seasonality,AnnPrecip,PrecipSeason,biome_index,...
    pred_equal_split, pred_fair_prop,...
    'VariableNames',{'DataSet' 'Order' 'Family' 'Genus' 'Species' 'Lat' 'Long' 'NumObs' 'NumPreyTypes' 'rare_NT' 'NetSpec' 'yInt' 'MF' 'rare_MF' 'sp50' 'Gini' 'SE' 'rare_SE' 'Mass' 'ID_MF' 'MMF' 'MSP' 'MLP' 'MassRange',...
    'RichBirds' 'RichMamms' 'RichStrigB' 'RichStrigNB' 'RichAccB' 'RichAccNB',...
    'NumHabs' 'RangeSize' 'fBirds' 'fMamms' 'fIns' 'funidBird' 'funidMamm',...
    'MOS' 'SD' 'Season' 'Temp' 'Seasonality' 'AnnPrecip' 'PrecipSeason' 'Biome',...
    'EqSplit' 'FairProp'});

writetable(raptor_ds)
% T = readtable('raptor_ds.txt')
% raptor_ds = addvars(T,richness_rodent);
% writetable(raptor_ds)
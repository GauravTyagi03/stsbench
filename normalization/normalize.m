%%% normalize & average MUA - THINGS dataset
% 2022 P. Papale fecit

clear all
addpath(genpath('\_code\code_utils_v2\'));

% constants
monkey = 'monkeyN';
datadir_gen = '\';
% chns to region
if monkey == 'monkeyN'
    rois = ones([1 1024]); % V1
    rois(513:768) = 2; % V4
    rois(769:end) = 3; % IT
else
    rois = ones([1 1024]); % V1
    rois(513:832) = 3; % IT
    rois(833:end) = 2; % V4
end
% window of responses
time_int{1} = 25:125; %V1
time_int{2} = 50:150; %V4
time_int{3} = 75:175; %IT

%
filename = [datadir_gen,monkey,'\THINGS_MUA_trials.mat'];
load(filename)
load([datadir_gen,monkey,'logs_1024chns_mapping_20220105.mat'])

% ALLMAT: [#trial #train_pic #test_pic #pic_rep #ncount #day]
days = max(ALLMAT(:,end));
train_idx = ALLMAT(:,2)>0;
test_idx = ALLMAT(:,3)>0;
rois = rois(mapping);
test_trials = ALLMAT(test_idx,3);
test_trials_idx = unique(test_trials);

% get some basic SNR and latency estimates
baseT = tb <= 0;
for chn = 1:1024
    for day = 1:days
        clear day_norm_pool norm_pool base temp_mua noise signal maxsignal
        day_norm_pool = ALLMAT(:,end)==day & ALLMAT(:,5)==1;
        norm_pool = ALLMUA(:,day_norm_pool,:);
        base = nanmean(nanmean(norm_pool(chn,:,baseT),3),2);
        temp_mua = squeeze(norm_pool(chn,:,:))-base;
        noise = nanstd(nanmean(temp_mua(:,baseT),2));
        signal = smooth(squeeze(nanmean(temp_mua(:,tb>0))),25,'lowess');
        maxsignal = max(signal);
        SNR(chn,day) = maxsignal/noise;
        lats(chn,day) = latencyfit4AM(squeeze(nanmean(temp_mua)),tb/1000,1,0)*1000;
    end
    clear base test_MUA_trials temp_mua noise signal idx_ maxsignal
    base = nanmean(nanmean(ALLMUA(chn,:,baseT),3),2);
    test_MUA_trials = squeeze(ALLMUA(chn,test_idx,:));
    for i = 1:100
        f = find(test_trials == test_trials_idx(i));
        temp_reps = test_MUA_trials(f,:);
        test_MUA(i,:) = nanmean(temp_reps);
    end
    temp_mua = squeeze(ALLMUA(chn,:,:))-base;
    noise = nanstd(nanmean(temp_mua(:,baseT),2));
    signal = squeeze(nanmean(test_MUA(:,tb>0),2));
    idx_ = find(signal==max(signal));
    maxsignal = max(smooth(squeeze(test_MUA(idx_,tb>0)),25,'lowess'));
    SNR_max(chn) = maxsignal/noise;
    chn
end

SNR = SNR(mapping,:);
SNR_max = SNR_max(mapping);
lats = lats(mapping,:);

% normalize and average (in time) MUA data:
%   we normalize the data on a daily basis by taking the mean and std of
%   the test trials of each day
normMUA = nan([size(ALLMUA,1) size(ALLMUA,2)]);
for chn = 1:length(rois)
    temp_chn_all = [];
    for day = 1:days
        clear day_norm_pool norm_pool
        day_norm_pool = test_idx & ALLMAT(:,end)==day;
        norm_pool = ALLMUA(:,day_norm_pool,:);
        day_trials = ALLMAT(:,end)==day;
        
        clear  norm_mean norm_std temp_chn
        gt = tb > time_int{rois(chn)}(1) & tb <= time_int{rois(chn)}(end);
        norm_mean = nanmean(norm_pool(chn,:,gt),'all');
        norm_std = nanstd(norm_pool(chn,:,gt),[],'all');
        temp_chn = nanmean(squeeze(ALLMUA(chn,day_trials,gt)),2);
        temp_chn = (temp_chn-norm_mean)./norm_std;
        temp_chn_all = [temp_chn_all; temp_chn];
    end
    normMUA(chn,:) = temp_chn_all;
end

normMUA = normMUA(mapping,:);
rois = rois(mapping);

% get and sort train MUA data
train_trials = ALLMAT(train_idx,2);
[~,train_sorted] = sort(train_trials);
train_MUA = normMUA(:,train_idx);
train_MUA = train_MUA(:,train_sorted,:);

% get test MUA data, average repetitions and sort
test_trials = ALLMAT(test_idx,3);
test_trials_idx = unique(test_trials);
[~,test_sorted] = sort(test_trials_idx);

clear test_MUA
test_MUA_trials = normMUA(:,test_idx);
for i = 1:length(test_trials_idx)
    clear f temp_reps
    f = find(test_trials == test_trials_idx(i));
    temp_reps = test_MUA_trials(:,f);
    test_MUA(:,i) = nanmean(temp_reps,2);
    for j = 1:length(f)
        test_MUA_reps(:,i,j) = test_MUA_trials(:,f(j));
    end
end
test_MUA = test_MUA(:,test_sorted);

% compute reliability and oracle correlation of response across repetitions
for chn = 1:size(test_MUA_reps,1)
    clear temp_chn temp_oracle
    temp_chn = squeeze(test_MUA_reps(chn,:,:));
    reliab(chn,:) = (1-pdist(temp_chn','correlation'));
    for i = 1:size(test_MUA_reps,3)
        temp_oracle(i) = corr(temp_chn(:,i),nanmean(temp_chn(:,setdiff(1:size(test_MUA_reps,3),i)),2));
    end
    oracle(chn) = nanmean(temp_oracle);
end

filename = [datadir_gen,monkey,'THINGS_normMUA.mat'];
save(filename,'test_MUA','train_MUA','SNR','lats','reliab','oracle','test_MUA_reps','tb','SNR_max','-v7.3')
pause(5)

filename = [datadir_gen,monkey,'THINGS_normMUA_all.mat'];
save(filename,'normMUA','ALLMAT','tb','-v7.3')
pause(5)
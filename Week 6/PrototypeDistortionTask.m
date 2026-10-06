%% Prototype Distortion Task
%
% Human category learning experiment.
%
% The experiment consists of:
%   1. Training on examples of "leptons"
%   2. Training on examples of "non-leptons"
%   3. Classification of previously seen and unseen test stimuli
%   4. Plotting the proportion of "lepton" responses as a function
%      of distance from the prototype
%

clear;
close all;
clc;


%% Experiment parameters

nDots = 3;                         % Number of coloured dots
nTraining = 15;                    % Number of examples in each training set

leptonDistance    = 1.5;           % Distance of training leptons from prototype
nonLeptonDistance = 2.5;           % Distance of training non-leptons

testDistances = [1.0 1.5 2.0 2.5];
nTestPerDistance = 10;

nDimensions = 2 * nDots;           % x and y coordinate for each dot

dotColors = [1 0 0;                % red
             0 1 0;                % green
             0 0 1];               % blue


%% Generate prototype

% The prototype is a random point in the six-dimensional feature space.

prototype = 2 * (rand(1,nDimensions) - 0.5);


%% Generate training stimuli

% Training leptons all have distance 1.5 from the prototype.

trainingLeptons = generateStimuli( ...
    prototype, leptonDistance, nTraining);

% Training non-leptons all have distance 2.5 from the prototype.

trainingNonLeptons = generateStimuli( ...
    prototype, nonLeptonDistance, nTraining);


%% Generate unseen test stimuli

% For each test distance, generate a new set of stimuli.

testStimuli = zeros(length(testDistances), ...
                    nTestPerDistance, ...
                    nDimensions);

for i = 1:length(testDistances)

    testStimuli(i,:,:) = generateStimuli( ...
        prototype, testDistances(i), nTestPerDistance);

end


%% Determine common plot limits

% All stimuli should be shown using the same axis limits so that their
% absolute positions on the screen are directly comparable.

allStimuli = [trainingLeptons;
              trainingNonLeptons;
              reshape(testStimuli,[],nDimensions)];

xCoordinates = allStimuli(:,1:2:end);
yCoordinates = allStimuli(:,2:2:end);

xLimits = [min(xCoordinates(:)) max(xCoordinates(:))];
yLimits = [min(yCoordinates(:)) max(yCoordinates(:))];

margin = 0.2;

xLimits = xLimits + margin * [-1 1];
yLimits = yLimits + margin * [-1 1];


%% Training phase: leptons

showTrainingSet( ...
    trainingLeptons, ...
    dotColors, ...
    xLimits, ...
    yLimits, ...
    'Look at all these Leptons!');


%% Training phase: non-leptons

showTrainingSet( ...
    trainingNonLeptons, ...
    dotColors, ...
    xLimits, ...
    yLimits, ...
    'These are not Leptons!');


%% Construct test trials

% The test phase contains:
%
%   - all previously seen leptons
%   - all previously seen non-leptons
%   - new stimuli at each of the four test distances
%
% For every trial we store:
%   stimulus coordinates
%   distance from prototype
%   whether the stimulus was seen during training


% Previously seen stimuli

seenStimuli = [trainingLeptons;
               trainingNonLeptons];

seenDistances = [ ...
    repmat(leptonDistance,nTraining,1);
    repmat(nonLeptonDistance,nTraining,1)];

seenBefore = true(2*nTraining,1);


% Unseen stimuli

unseenStimuli = reshape( ...
    permute(testStimuli,[2 1 3]), ...
    [],nDimensions);

unseenDistances = repelem(testDistances',nTestPerDistance);

unseenBefore = false(size(unseenDistances));


% Combine all trials

trialStimuli = [seenStimuli;
                unseenStimuli];

trialDistances = [seenDistances;
                  unseenDistances];

trialSeen = [seenBefore;
             unseenBefore];

nTrials = size(trialStimuli,1);


%% Randomise trial order

trialOrder = randperm(nTrials);

trialStimuli   = trialStimuli(trialOrder,:);
trialDistances = trialDistances(trialOrder);
trialSeen      = trialSeen(trialOrder);


%% Test phase

responses = zeros(nTrials,1);

figure('WindowState','maximized');

for trial = 1:nTrials

    clf;

    % Use the same subplot size as during training
    subplot(3,5,8);

    plotStimulus( ...
        trialStimuli(trial,:), ...
        dotColors, ...
        xLimits, ...
        yLimits);

    title('Is this a Lepton? (Y/N)','FontSize',16);

    drawnow;

    responses(trial) = getYesNoResponse;

end

close;


%% Analyse unseen test stimuli

% The dependent variable is the proportion of trials for which the
% participant answered "yes".

pLeptonUnseen = zeros(size(testDistances));
seUnseen      = zeros(size(testDistances));

for i = 1:length(testDistances)

    idx = ~trialSeen & ...
          trialDistances == testDistances(i);

    pLeptonUnseen(i) = mean(responses(idx));

    n = sum(idx);

    seUnseen(i) = sqrt( ...
        pLeptonUnseen(i) * ...
        (1 - pLeptonUnseen(i)) / n);

end


%% Analyse previously seen stimuli

seenDistancesToPlot = [leptonDistance nonLeptonDistance];

pLeptonSeen = zeros(size(seenDistancesToPlot));
seSeen      = zeros(size(seenDistancesToPlot));

for i = 1:length(seenDistancesToPlot)

    idx = trialSeen & ...
          trialDistances == seenDistancesToPlot(i);

    pLeptonSeen(i) = mean(responses(idx));

    n = sum(idx);

    seSeen(i) = sqrt( ...
        pLeptonSeen(i) * ...
        (1 - pLeptonSeen(i)) / n);

end


%% Plot results

% Put seen and unseen data on the same x-axis
pSeenPlot  = [NaN pLeptonSeen(1) NaN pLeptonSeen(2)];
seSeenPlot = [NaN seSeen(1)      NaN seSeen(2)];

figure('Position',[200 100 700 700]);

% Unseen stimuli
subplot(2,1,1);
bar(testDistances,100*pLeptonUnseen);
hold on;
errorbar(testDistances,100*pLeptonUnseen,100*seUnseen,...
    'ko','LineStyle','none');

xlim([0.5 3]);
ylim([0 100]);
xticks(testDistances);

xlabel('Distance from prototype');
ylabel('Lepton responses (%)');
title('Previously unseen stimuli');


% Previously seen stimuli
subplot(2,1,2);
bar(testDistances,100*pSeenPlot);
hold on;
errorbar(testDistances,100*pSeenPlot,100*seSeenPlot,...
    'ko','LineStyle','none');

xlim([0.5 3]);
ylim([0 100]);
xticks(testDistances);

xlabel('Distance from prototype');
ylabel('Lepton responses (%)');
title('Previously seen training stimuli');

%% Save results

exportgraphics( ...
    gcf, ...
    'PrototypeDistortionResult.png', ...
    'Resolution',300);


% Save trial-by-trial data in a simple table.

results = table( ...
    trialDistances, ...
    trialSeen, ...
    responses, ...
    'VariableNames', ...
    {'Distance','SeenDuringTraining','LeptonResponse'});

writetable(results,'PrototypeDistortionResult.csv');


%% Local functions


function stimuli = generateStimuli(prototype,distance,nStimuli)
%GENERATESTIMULI Generate stimuli at a fixed Euclidean distance
%from the prototype.

    nDimensions = length(prototype);

    % Generate random directions.
    directions = randn(nStimuli,nDimensions);

    % Normalise every direction to unit length.
    lengths = sqrt(sum(directions.^2,2));
    directions = directions ./ lengths;

    % Scale to the requested distance.
    directions = distance * directions;

    % Shift so that the vectors surround the prototype.
    stimuli = directions + prototype;

end


function showTrainingSet(stimuli,dotColors,xLimits,yLimits,plotTitle)
%SHOWTRAININGSET Display all training examples in a grid.

    nStimuli = size(stimuli,1);

    nRows = 3;
    nColumns = 5;

    figure('WindowState','maximized');

    for i = 1:nStimuli

        subplot(nRows,nColumns,i);

        plotStimulus( ...
            stimuli(i,:), ...
            dotColors, ...
            xLimits, ...
            yLimits);

    end

    sgtitle(plotTitle, ...
            'Color','r', ...
            'FontSize',16);

    waitforbuttonpress;

    close;

end


function plotStimulus(stimulus,dotColors,xLimits,yLimits)
%PLOTSTIMULUS Plot one configuration of coloured dots.

    nDots = size(dotColors,1);

    hold on;

    for d = 1:nDots

        x = stimulus(2*d-1);
        y = stimulus(2*d);

        plot( ...
            x,y,'o', ...
            'MarkerFaceColor',dotColors(d,:), ...
            'MarkerEdgeColor',dotColors(d,:), ...
            'MarkerSize',15);

    end

    xlim(xLimits);
    ylim(yLimits);

    xticks([]);
    yticks([]);

    box on;

end


function response = getYesNoResponse
%GETYESNORESPONSE Wait for a Y or N key press.
%
% Returns:
%   1 = Y
%   0 = N

    while true

        % waitforbuttonpress returns:
        %   1 for a keyboard press
        %   0 for a mouse click
        keyPressed = waitforbuttonpress;

        % Ignore mouse clicks
        if ~keyPressed
            continue
        end

        key = get(gcf,'CurrentCharacter');

        if strcmpi(key,'Y')
            response = 1;
            return
        elseif strcmpi(key,'N')
            response = 0;
            return
        end

    end
end
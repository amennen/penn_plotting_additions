%function MakeExampleStimuli_replicateblock(categS,categF,imagesS,imagesF,imgPropS)
% function [] = MakeExampleStimuli()
%
%
% Written by: Megan deBettencourt - edited by ACM
% Version: 1.0
data_dir = pwd;
% first we have to load in block data
load('data/blockdata_7_20191028T185516.mat')
%%
blockNum=8; % change here if you want a different example block
SCENE = 1;
FACE = 2;
imgPropS = blockData(blockNum).smoothAttImgProp;
categS = blockData(blockNum).categs{SCENE};
categF = blockData(blockNum).categs{FACE};
imagesS = blockData(blockNum).images{SCENE};
imagesF = blockData(blockNum).images{FACE};
%%
% print category separation
categSep = blockData(blockNum).categsep;
categSep'

%% Boilerplate

Screen('Preference', 'SkipSyncTests', 1);

%initialize system time calls
GetSecs;

%% Experimental Parameters

%categories
SCENE = 1;
FACE = 2;
nSubCategs = 8;
INDOOR = 1;
OUTDOOR = 2;
MALE = 3;
FEMALE = 4;
MALESAD = 5;
FEMALESAD = 6;
MALEHAPPY = 7;
FEMALEHAPPY=8;
%the proportions of the two images must sum to 1
%imgPropS = 1-imgPropF;

% display parameters
textColor = 0;
textFont = 'Arial';
textSize = 18;
textSpacing = 25;
fixColor = 0;
respColor = 255;
backColor = 127;
imageSize = 256; % assumed square %MdB check image size
fixationSize = 4;% pixels

% function mapping classifier output to attended image proportion
gain = 3;
x_shift = .2;
y_shift = .15;
steepness = .9;

% timing
instructDur = 2; %[sec]
fixDur = 2;      %[sec]
stimDur = 2;     %[sec]

%instructions
sceneShorterInstruct = 'indoor scenes';
%faceShorterInstruct = subCategS;

%where to save the screenshots
%figDir = '/Users/megan/Documents/writing/papers/rtfmrisustainedattn/replicateblock_figs/';
%assert(logical(exist(figDir,'dir')),'check that you have the ntb volume mounted');

%% Initialize Screens

screenNumbers=Screen('Screens');
screenNum = screenNumbers(1);

%size of the display screen
screenX = 1280;
screenY = 720;

%create main window
%mainWindow = Screen('OpenWindow',screenNum,backColor);

mainWindow = Screen('OpenWindow',screenNum,backColor,[0 0 screenX screenY]);

% details of main window
centerX = screenX/2; centerY = screenY/2;
Screen(mainWindow,'TextFont',textFont);
Screen(mainWindow,'TextSize',textSize);

% placeholder for images
imageRect = [0,0,imageSize,imageSize];

% position of images
centerRect = [centerX-imageSize/2,centerY-imageSize/2,centerX+imageSize/2,centerY+imageSize/2];

% position of fixation dot
fixDotRect = [centerX-fixationSize,centerY-fixationSize,centerX+fixationSize,centerY+fixationSize];


%% Load Images
% changes from ACM work imac to laptop - 4/19/20
cd ~/rtAttenPenn/images/
for categ=1:nSubCategs
    
    % move into the right folder
     if (categ == INDOOR)
        cd indoor;
    elseif (categ == OUTDOOR)
        cd outdoor;
    elseif (categ == MALE)
        cd male_neut;
    elseif (categ == FEMALE)
        cd female_neut;
    elseif (categ == MALESAD)
        cd male_sad;
    elseif (categ == FEMALESAD)
        cd female_sad;
    elseif (categ == MALEHAPPY)
        cd male_happy;
    elseif (categ == FEMALEHAPPY)
        cd female_happy
    else
        error('Impossible category!');
    end
    
    
    % get filenames
    dirList{categ} = dir; %#ok<AGROW>
    dirList{categ} = dirList{categ}(3:end); %#ok<AGROW>  skip . & ..
    if (~isempty(dirList{categ}))
        if (strcmp(dirList{categ}(1).name,'.DS_Store')==1)
            dirList{categ} = dirList{categ}(2:end); %#ok<AGROW>
        end
        
        if (strcmp(dirList{categ}(end).name,'Thumbs.db')==1)
            dirList{categ} = dirList{categ}(1:(end-1)); %#ok<AGROW>
        end
        
        numImages(categ) = length(dirList{categ}); %#ok<AGROW>
        
        if (numImages(categ)>0)
            
            % get images
            for img=1:numImages(categ)
               
                % read images
                images{categ,img} = imread(dirList{categ}(img).name); %#ok<AGROW>
                tempFFT = fft2(images{categ,img});
                imagePower{categ,img} = abs(tempFFT); %#ok<NASGU,AGROW>
                imagePhase{categ,img} = angle(tempFFT); %#ok<NASGU,AGROW>
            end
            
            % randomize order of images in each run
            %imageShuffle{categ} = randperm(numImages(categ)); %#ok<AGROW>
            cd ..;
        end
    else
        error('Need at least one image per directory!');
    end
end
cd(data_dir)


%% Start Experiment

% wait for initial trigger
Priority(MaxPriority(screenNum));
Screen(mainWindow,'FillRect',backColor);
%Priority(0);


%% instructions

%show scene instructions
figname = ['indoor_Instruct.png'];
tempBounds = Screen('TextBounds',mainWindow,'indoor');
Screen('drawtext',mainWindow,sceneShorterInstruct,centerX-tempBounds(3)/2,centerY-tempBounds(4)/5,textColor);
clear tempBounds;
instructflip = Screen('Flip',mainWindow); % turn on
imageArray = Screen('GetImage', mainWindow);
fn = 'stimuli_examples/indoor_instruct.jpg';
imwrite(imageArray, fn)
% % show instructions
% tempBounds = Screen('TextBounds',mainWindow,'indoor');
% Screen('drawtext',mainWindow,'indoor',centerX-tempBounds(3)/2,centerY-tempBounds(4)/5,textColor);
% clear tempBounds;
% tic
% instructflip = Screen('Flip',mainWindow,blockData(iBlock).plannedinstructonset+instructOn); %#ok<AGROW> % turn on

% show fixation
Screen(mainWindow,'FillOval',fixColor,fixDotRect);
Screen('Flip',mainWindow,instructflip+1); %turn off
imageArray = Screen('GetImage', mainWindow);
fn = 'stimuli_examples/fixation.jpg';
imwrite(imageArray, fn)
%% stimuli

for iStim = 1:length(imagesS)
    % prep images
    %for half=[SCENE FACE]
        % get current images
        tempPower{SCENE} = imagePower{categS(iStim),imagesS(iStim)}; %#ok<AGROW>
        tempImagePhase{SCENE} = imagePhase{categS(iStim),imagesS(iStim)}; %#ok<AGROW>
        tempImage{SCENE} = real(ifft2(tempPower{SCENE}.*exp(sqrt(-1)*tempImagePhase{SCENE}))); %#ok<AGROW>
        
        tempPower{FACE} = imagePower{categF(iStim),imagesF(iStim)}; %#ok<AGROW>
        tempImagePhase{FACE} = imagePhase{categF(iStim),imagesF(iStim)}; %#ok<AGROW>
        tempImage{FACE} = real(ifft2(tempPower{FACE}.*exp(sqrt(-1)*tempImagePhase{FACE}))); %#ok<AGROW>
    %end
    
    % generate image
    fullImage = uint8(imgPropS(iStim)*tempImage{SCENE}+(1-imgPropS(iStim))*tempImage{FACE});
    fprintf('scene smooth proportion is %.2f\n\n', imgPropS(iStim));
    % make textures
    imageTex = Screen('MakeTexture',mainWindow,fullImage);
    Screen('PreloadTextures',mainWindow,imageTex);
    
    %for saving
    strImgPropF = num2str(1-imgPropS(iStim));
    strImgPropF = strImgPropF(3:end);
    strImgPropS = num2str(imgPropS(iStim));
    strImgPropS = strImgPropS(3:end);
    
    % show image with black fixation dot
    %figname = ['stim_' num2str(iStim) '_' dirList{categS(iStim)}(imagesS(iStim)).name(1:(end-4)) '_' dirList{categF(iStim)}(imagesS(iStim)).name(1:(end-4))  '_' strImgPropS 'imgPropS' '_' strImgPropF 'imgPropF' '_bFix.png'];
    FlushEvents('keyDown');
    Priority(MaxPriority(screenNum));
    Screen('FillRect',mainWindow,backColor);
    Screen('DrawTexture',mainWindow,imageTex,imageRect,centerRect);
    Screen(mainWindow,'FillOval',fixColor,fixDotRect);
    if iStim ~=1
        tempflip(iStim) = Screen('Flip',mainWindow,tempflip(iStim-1)+1);
    else
        tempflip(iStim) = Screen('Flip',mainWindow,instructflip+2);
    end
    %system(['screencapture ' figDir figname])
    
    % show image with white fixation dot
    %figname = ['stim_' dirList{1}(iStim).name(1:(end-4)) '_' dirList{2}(iStim).name(1:(end-4)) '_' strImgPropS 'imgPropS' '_' strImgPropF 'imgPropF' '_wFix.png'];
    %Screen('FillRect',mainWindow,backColor);
    %Screen('DrawTexture',mainWindow,imageTex,imageRect,centerRect);
    %Screen(mainWindow,'FillOval',respColor,fixDotRect);
    %Screen('Flip',mainWindow,stimOnset1+stimDur);
    %system(['screencapture ' figDir figname])
    %WaitSecs(stimDur);
    
    imageArray = Screen('GetImage', mainWindow);
    fn = sprintf('stimuli_examples/stim_%i.jpg', iStim);
    imwrite(imageArray, fn)
end

%% fixation

figname = 'fixation.png';
%Screen(mainWindow,'FillOval',fixColor,fixDotRect);
%Screen('Flip',mainWindow);
Screen(mainWindow,'FillOval',fixColor,fixDotRect);
fixOnset =Screen('Flip',mainWindow,tempflip(end)+1); %turn off
Screen(mainWindow,'FillOval',fixColor,fixDotRect);
Screen('Flip',mainWindow,fixOnset+4);
%WaitSecs(4);
%system(['screencapture ' figDir figname])

%% clean up and go home

sca;
ListenChar(1);
fclose('all');
%end

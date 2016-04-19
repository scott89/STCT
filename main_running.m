close all
clear
clc
warning off all;

addpath('obt');

% addpath(('D:\vlfeat-0.9.14\toolbox'));
% vl_setup

addpath(('./rstEval'));
addpath(['./trackers/VIVID_Tracker'])

seqs=configSeqs;

trackers=configTrackers;

shiftTypeSet = {'left','right','up','down','topLeft','topRight','bottomLeft','bottomRight','scale_8','scale_9','scale_11','scale_12'};

evalType='OPE'; %'OPE','SRE','TRE'

% diary(['./tmp/' evalType '.txt']);

numSeq=length(seqs);
numTrk=length(trackers);

finalPath = ['./results/results_' evalType '_CVPR13/'];
finalPath = 'benchmark50_res/';
finalPath = 'OBT50_SRE/';
if ~exist(finalPath,'dir')
    mkdir(finalPath);
end

tmpRes_path = ['./tmp/' evalType '/'];
bSaveImage=0;

if ~exist(tmpRes_path,'dir')
    mkdir(tmpRes_path);
end

% pathAnno = './anno/';

for idxSeq=length(seqs)-2
    s = seqs{idxSeq};
    
    %      if ~strcmp(s.name, 'coke')
    %         continue;
    %      end
    
    s.len = s.endFrame - s.startFrame + 1;
    s.s_frames = cell(s.len,1);
    nz	= strcat('%0',num2str(s.nz),'d'); %number of zeros in the name of image
    for i=1:s.len
        image_no = s.startFrame + (i-1);
        id = sprintf(nz,image_no);
        s.s_frames{i} = strcat(s.path,id,'.',s.ext);
        s.s_frames{i} = ['/home/lijun/Research/CVPR16/Data/data/' upper(s.name(1)) s.name(2:end) '/img/' id '.' s.ext];
    end
    %
        img = imread(s.s_frames{1});
        [imgH,imgW,ch]=size(img);
    pathAnno = ['/home/lijun/Research/CVPR16/Data/data/' upper(s.name(1)) s.name(2:end) '/'];
    try
        rect_anno = dlmread([pathAnno 'groundtruth_rect.txt']);
    catch
        pathAnno = [pathAnno upper(s.name(1)) s.name(2:end) '/'];
        rect_anno = dlmread([pathAnno 'groundtruth_rect.txt']);
    end
    numSeg = 20;
    
    [subSeqs, subAnno]=splitSeqTRE(s,numSeg,rect_anno);
    fprintf('%s: \t %4d - %4d - %4d\n', s.name, subSeqs{1}.startFrame, subSeqs{1}.endFrame, s.len);
%     continue;
    switch evalType
        case 'SRE'
            subS = subSeqs{1};
            subA = subAnno{1};
            subSeqs=[];
            subAnno=[];
            r=subS.init_rect;
            
            for i=1:length(shiftTypeSet)
                subSeqs{i} = subS;
                shiftType = shiftTypeSet{i};
                subSeqs{i}.init_rect=shiftInitBB(subS.init_rect,shiftType,imgH,imgW);
                subSeqs{i}.shiftType = shiftType;
                
                subAnno{i} = subA;
            end
            
        case 'OPE'
            subS = subSeqs{1};
            subSeqs=[];
            subSeqs{1} = subS;
            
            subA = subAnno{1};
            subAnno=[];
            subAnno{1} = subA;
        otherwise
    end
    
    
    for idxTrk=1:numTrk
        t = trackers{idxTrk};
        switch t.name
            case {'VTD','VTS'}
                continue;
        end
        
        results = [];
        for idx=1:length(subSeqs)
            disp([num2str(idxTrk) '_' t.name ', ' num2str(idxSeq) '_' s.name ': ' num2str(idx) '/' num2str(length(subSeqs))])
            
            rp = [tmpRes_path s.name '_' t.name '_' num2str(idx) '/'];
            if bSaveImage&~exist(rp,'dir')
                mkdir(rp);
            end
            
            subS = subSeqs{idx};
            
            subS.name = [subS.name '_' num2str(idx)];
            
            
            
            funcName = ['res=run_' t.name '(subS, [upper(s.name(1)) s.name(2:end)], rp, bSaveImage);'];
            eval(funcName);
            if isempty(res)
                results = [];
                break;
            end
            res.len = subS.len;
            res.annoBegin = subS.annoBegin;
            res.startFrame = subS.startFrame;
            
            switch evalType
                case 'SRE'
                    res.shiftType = shiftTypeSet{idx};
            end      
            results{idx} = res;    
        end
    save([finalPath s.name '_' t.name '.mat'], 'results');
end
end

figure
t=clock;
t=uint8(t(2:end));
disp([num2str(t(1)) '/' num2str(t(2)) ' ' num2str(t(3)) ':' num2str(t(4)) ':' num2str(t(5))]);


function adjFS = computeAdjacencyMatrixPatchWise(fs, patchIndex, linkType, linkOptions)

assert(length(patchIndex)==size(fs.data,2),'Invalid specification of patch membership.');

%{
funcHistDist = 
@kullback_leibler_divergence
@jensen_shannon_divergence
@chi_square_statistics_fast
@kolmogorov_smirnov_distance
@histogram_intersection
%}

switch upper(linkType)
    % % --------------  BINARY -------------- %
    case upper({ 'similarity', 'ThickNet' })
        linkFunc = @(x,y,opt) abs(median(x)-median(y)) < opt.threshold;
        
    case upper({ 'dissimilarity', 'GRADIENT', 'DECLINE' })
        linkFunc = @(x,y,opt) abs(median(x)-median(y)) > opt.threshold;
        
    case upper({ 'nonparam-ranksum-signif', 'NP-RS-p', 'RS-signif' })
        % whether the difference between the thickness distributions is
        % significant or not. Output binary adjacency matrix
        linkFunc = @ranksumSignif;
        
    case upper({ 'ttest2-signif', 'tstat-signif', 'T-signif' })
        % whether the difference between the thickness distributions is
        % significant or not. Output binary adjacency matrix
        linkFunc = @ttest2signif;

    % % --------------  weighted -------------- %
    case upper({ 'median-diff', 'patchwise-median-diff' })
        linkFunc = @(x,y,opt) abs(median(x)-median(y)) ;

    case upper({ 'Wee2012HBM', 'exp-neg-patchwise-mean-diff-norm-2-sum-std' })
        linkFunc = @(x,y,opt) exp( -( ( mean(x)-mean(y) )^2 ) / (2*(std(x)+std(y))) );
        
    case upper({ 'WilcoxonRS', 'nonparam-ranksum', 'NP-RS', 'RS' })
        % whether the difference between the thickness distributions is
        % significant or not. Output binary adjacency matrix
        linkFunc = @ranksumStatistic;
        
    case upper({ 't-statistic', 'tstat', 'T' })
        % whether the difference between the thickness distributions is
        % significant or not. Output binary adjacency matrix
        linkFunc = @ttest2statistic;
        
    case upper({...
            'histogram-correlation', 'histcorr', ...
            'histogram-np-ranksum-signif', 'hist-np-rs-signif', ...
            'kullback_leibler_divergence', 'KL-Div', ...
            'jensen_shannon_divergence', 'JS-Div', ...
            'chi_square_statistics_fast', 'ChiSq-Stat', ...
            'kolmogorov_smirnov_distance', 'KS-Dist', ...
            'histogram_intersection', 'HistogramIntersection', 'HistInt'})    
        pairwiseHistLink = @(x,y, opt) histDistLink(x,y,linkType, opt);
        linkFunc = pairwiseHistLink;

     % % --------------  NOT implemented properly yet --------------    
        %     case upper({ 'correlation-peasron', 'corr-rho', 'rho' })
%         linkFunc = @corrcoefDiffLen;
% 
%     case upper({ 'correlation-peasron-significance', 'corr-rho-signif', 'rho-signif' })
%         linkFunc = @corrcoefDiffLenSignif;        
    
        
    otherwise
        error('Invalid linking criterion.');
        
end

linkMat = computeAdjMat(fs.data, patchIndex, linkFunc, linkOptions, fs.subjectIds);

adjFS = fs;
adjFS.data = linkMat;

linkFuncDescr = func2str(linkFunc);

end


function linkMat = computeAdjMat(fsData, patchIndex, linkFunc, linkOptions, subjectIds)

numSub = size(fsData,1);
numDim = size(fsData,2);
uniqPatches = unique(patchIndex(~isnan(patchIndex)));
numPatches = length(uniqPatches);

idxTriU =  find( triu(ones(numPatches),1) );

linkMat = cast( nan(numSub,numPatches,numPatches), 'single');
for ss = 1 : numSub
    fprintf('\n Working on subject %d', ss);
    thisSubjData = fsData(ss,:);
    ewMat = pairwiseLink(thisSubjData,patchIndex,linkFunc, linkOptions);
    
    if any(isnan(ewMat(idxTriU)))       %#ok<FNDSB>
        warning('\n NaNs found in edge weight computations for subject %s ', ...
            subjectIds{ss});
    end
    
    linkMat(ss,:,:) = ewMat;
end

if isfield(linkOptions,'normalizeEdgesTo0to1') ...
        && linkOptions.normalizeEdgesTo0to1
    minWt = nanmin(linkMat(:));
    maxWt = nanmax(linkMat(:));
    linkMat = ( linkMat - minWt ) ./ ( maxWt - minWt);
end

end

function linkValueMat =  pairwiseLink(data,patchIndex,linkFunc, linkOptions)

PatchList = unique(patchIndex(~isnan(patchIndex)));
% patchWiseData = arrayfun( @(x) {data(patchIndex==x)}, uniqPatches);
getPatchData = @(x) data(patchIndex==x) ;

linkValueMat = cast(nan(length(PatchList),length(PatchList)), 'single');
for pp = 1 : length(PatchList)
    patchData_pp = getPatchData(pp);
    % computing this for only the upper triangular part of the matrix
    % notice the starting point from pp+1
    for rr = pp + 1 : length(PatchList)
        linkValueMat(pp,rr) = linkFunc(patchData_pp, getPatchData(rr), linkOptions);
    end
end

end
function linkValue = histDistLink(x, y, linkType, linkOptions)

numBins = 100;
if isfield(linkOptions,'numBins')
    numBins = linkOptions.numBins;
end

edgesOfEdges = quantile([x(:); y(:)],[0.05 0.95]);
edges = linspace(edgesOfEdges(1), edgesOfEdges(2), numBins);

% [ histOne ] = histcounts(x, edges, 'Normalization', 'probability'); 
% [ histTwo ] = histcounts(y, edges, 'Normalization', 'probability'); 
% % as histcounts requires version later than 2012b, switching to older
% versions

[ histOne ] = myHistCountProb(x, edges); 
[ histTwo ] = myHistCountProb(y, edges);

% one could optionally replace specifying edges with 'BinMethod' to be 'fd', 
% wherein the Freedman-Diaconis rule is less sensitive to outliers in the data, 
% and may be more suitable for data with heavy-tailed distributions. 
% It uses a bin width of 2*IQR(X(:))*numel(X)^(-1/3), 
% where IQR is the interquartile range of X.

switch upper(linkType)
    case  upper({'histogram-correlation', 'histcorr'})
        linkFunc = @corrcoefHist;
        
    case  upper({'histogram-np-ranksum-signif', 'hist-np-rs-signif'})
        linkFunc = @ranksumSignif;
        
    case upper({'kullback_leibler_divergence', 'KL-Div'})
        linkFunc = @kullback_leibler_divergence;
        
    case upper({'jensen_shannon_divergence', 'JS-Div'})
        linkFunc = @jensen_shannon_divergence;
        
    case upper({'chi_square_statistics', 'ChiSq-Stat'})
        linkFunc = @chi_square_statistics;
        
    case upper({'kolmogorov_smirnov_distance', 'KS-Dist'})
        linkFunc = @kolmogorov_smirnov_distance; %
        
    case upper({'histogram_intersection', 'HistogramIntersection', 'HistInt'})
        linkFunc = @histogram_intersection;
        
    otherwise
        error('Invalid selection of histogram based distance.');
end

linkValue = linkFunc(histOne, histTwo);

end

function hist = myHistCountProb(x, edges)
% remember edges only cover 5-95% range and strip out the 5% outliers

[ hist ] = histc(x, edges); 

% working with extremely skewed histograms
if nnz(hist)==0
    % all of them above upper bound
    if all(x>=edges(end))
        hist(end) = 1;
    % all of them below lower bound
    elseif all(x<=edges(1))
        hist(1) = 1;
    end
end

hist = hist / sum(hist);

end

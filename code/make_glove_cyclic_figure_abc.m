%% make_glove_cyclic_figure_abc.m
% Publication-style figure for 4 s cyclic finger-bending test.
%
% Uses:
%   - Representative trace: 4 s cycle run1
%   - Statistics: 4 s cycle run1, run3, run4
%   - Excludes run2 because it contains index high-resistance artifacts
%
% Put this script in the same folder as the CSV files and run:
%   close all force; clear; clc; make_glove_cyclic_figure_abc

clearvars;
close all force;
clc;

%% ---------------- User settings ----------------
dataFolder = fileparts(mfilename('fullpath'));
if isempty(dataFolder)
    dataFolder = pwd;
end

outFolder = fullfile(dataFolder, 'paper_cyclic_figures_4s');
if ~exist(outFolder, 'dir')
    mkdir(outFolder);
end

% Use newest matching run1 file for representative trace.
representativeRun = 'run1';

% Use these runs for cycle statistics.
statsRuns = {'run1', 'run3', 'run4'};

% Target fingers and sensor channels.
fingers = {'Thumb', 'Index', 'Middle', 'Ring', 'Pinky'};
sensors = {'Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'ReverseHorizontal', 'ReverseThumb'};
intendedSensorIndex = containers.Map(fingers, num2cell(1:5));

% Panel A style.
plotAll7Sensors = true;      % true = show all sensors; false = intended sensor only
panelAYLim = [-0.55 1.35];   % set [] for automatic y-axis
shadeBendRegions = true;

% Heatmap color limits. [] = automatic symmetric limit.
heatmapCLim = [-0.70 0.70];

% Colors. These are used consistently for the 7 sensors.
sensorColors = [
    0.0000 0.4470 0.7410;   % Thumb
    0.8500 0.3250 0.0980;   % Index
    0.9290 0.6940 0.1250;   % Middle
    0.4940 0.1840 0.5560;   % Ring
    0.4660 0.6740 0.1880;   % Pinky
    0.3010 0.7450 0.9330;   % ReverseHorizontal
    0.6350 0.0780 0.1840;   % ReverseThumb
];

%% ---------------- Locate files ----------------
fprintf('Data folder: %s\n', dataFolder);

repFullFile = findNewestFile(dataFolder, sprintf('glove_cyclic_finger_full_4s_cycle_%s_*.csv', representativeRun));
fprintf('Representative full file: %s\n', repFullFile);

cycleFiles = cell(numel(statsRuns), 1);
for i = 1:numel(statsRuns)
    pattern = sprintf('glove_cyclic_finger_cycle_summary_4s_cycle_%s_*.csv', statsRuns{i});
    cycleFiles{i} = findNewestFile(dataFolder, pattern);
    fprintf('Statistics cycle file %s: %s\n', statsRuns{i}, cycleFiles{i});
end

%% ---------------- Read data ----------------
Trep = readtable(repFullFile, 'VariableNamingRule', 'preserve');
Trep.target_finger = string(Trep.target_finger);
Trep.phase = string(Trep.phase);
Trep.label = string(Trep.label);

TallCycles = table();
for i = 1:numel(cycleFiles)
    T = readtable(cycleFiles{i}, 'VariableNamingRule', 'preserve');
    T.run_id = repmat(string(statsRuns{i}), height(T), 1);
    TallCycles = [TallCycles; T]; %#ok<AGROW>
end
TallCycles.target_finger = string(TallCycles.target_finger);
TallCycles.intended_sensor = string(TallCycles.intended_sensor);

%% ---------------- Compute statistics ----------------
% Matrix for heatmap: mean intended response across selected runs.
meanIntendedByFingerCycle = nan(numel(fingers), 5);
sdIntendedByFingerCycle = nan(numel(fingers), 5);

% Overall intended response per finger across all cycles and selected runs.
intendedMean = nan(numel(fingers), 1);
intendedSD = nan(numel(fingers), 1);
intendedN = nan(numel(fingers), 1);
allIndividualPoints = cell(numel(fingers), 1);

for f = 1:numel(fingers)
    finger = fingers{f};
    fingerMask = TallCycles.target_finger == string(finger);

    valuesAll = TallCycles.intended_local_response(fingerMask);
    allIndividualPoints{f} = valuesAll;
    intendedMean(f) = mean(valuesAll, 'omitnan');
    intendedSD(f) = std(valuesAll, 'omitnan');
    intendedN(f) = sum(isfinite(valuesAll));

    for c = 1:5
        mask = fingerMask & TallCycles.cycle == c;
        vals = TallCycles.intended_local_response(mask);
        meanIntendedByFingerCycle(f, c) = mean(vals, 'omitnan');
        sdIntendedByFingerCycle(f, c) = std(vals, 'omitnan');
    end
end

% Save processed stats.
processed = table(string(fingers(:)), intendedN, intendedMean, intendedSD, ...
    'VariableNames', {'Finger', 'N_cycles', 'Mean_intended_dR_R0', 'SD_intended_dR_R0'});
writetable(processed, fullfile(outFolder, 'cyclic_4s_run1_run3_run4_intended_stats.csv'));

cycleMeanTable = array2table(meanIntendedByFingerCycle, 'VariableNames', compose('Cycle_%d', 1:5));
cycleMeanTable.Finger = string(fingers(:));
cycleMeanTable = movevars(cycleMeanTable, 'Finger', 'Before', 1);
writetable(cycleMeanTable, fullfile(outFolder, 'cyclic_4s_run1_run3_run4_cycle_heatmap_values.csv'));

%% ---------------- Separate panel figures ----------------
figA = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 18 18]);
plotPanelA(figA, Trep, fingers, sensors, intendedSensorIndex, sensorColors, panelAYLim, plotAll7Sensors, shadeBendRegions);
set(figA, 'InvertHardcopy', 'off');
saveFigure(figA, fullfile(outFolder, 'panel_a_representative_run1_all7_cycles'));

figB = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 13 9]);
axB = axes(figB);
plotPanelB(axB, meanIntendedByFingerCycle, fingers, heatmapCLim);
set(figB, 'InvertHardcopy', 'off');
saveFigure(figB, fullfile(outFolder, 'panel_b_cycle_response_heatmap'));

figC = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 13 8]);
axC = axes(figC);
plotPanelC(axC, intendedMean, intendedSD, allIndividualPoints, fingers, sensorColors(1:5, :));
set(figC, 'InvertHardcopy', 'off');
saveFigure(figC, fullfile(outFolder, 'panel_c_intended_repeatability'));

%% ---------------- Combined figure ----------------
fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 24 18]);
set(fig, 'InvertHardcopy', 'off');

% Manual layout: panel A on left, panels B and C on right.
leftX = 0.065;
leftW = 0.58;
rightX = 0.705;
rightW = 0.255;

panelAHeight = 0.125;
panelAGap = 0.035;
panelAYTop = 0.835;

for f = 1:numel(fingers)
    y = panelAYTop - (f - 1) * (panelAHeight + panelAGap);
    ax = axes(fig, 'Position', [leftX, y, leftW, panelAHeight]); %#ok<LAXES>
    plotOneFingerBlock(ax, Trep, fingers{f}, sensors, intendedSensorIndex, sensorColors, panelAYLim, plotAll7Sensors, shadeBendRegions, f == 1);
    if f < numel(fingers)
        ax.XTickLabel = [];
        xlabel(ax, '');
    end
end
annotation(fig, 'textbox', [0.018 0.94 0.04 0.04], 'String', '(a)', 'LineStyle', 'none', ...
    'FontName', 'Arial', 'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');

axB2 = axes(fig, 'Position', [rightX, 0.57, rightW, 0.33]); %#ok<LAXES>
plotPanelB(axB2, meanIntendedByFingerCycle, fingers, heatmapCLim);
annotation(fig, 'textbox', [0.655 0.905 0.04 0.04], 'String', '(b)', 'LineStyle', 'none', ...
    'FontName', 'Arial', 'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');

axC2 = axes(fig, 'Position', [rightX, 0.14, rightW, 0.31]); %#ok<LAXES>
plotPanelC(axC2, intendedMean, intendedSD, allIndividualPoints, fingers, sensorColors(1:5, :));
annotation(fig, 'textbox', [0.655 0.465 0.04 0.04], 'String', '(c)', 'LineStyle', 'none', ...
    'FontName', 'Arial', 'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');

saveFigure(fig, fullfile(outFolder, 'figure_abc_cyclic_4s_combined'));

fprintf('\nDone. Figures saved in:\n%s\n', outFolder);

%% ========================================================================
% Local functions
% ========================================================================

function filename = findNewestFile(folder, pattern)
    files = dir(fullfile(folder, pattern));
    if isempty(files)
        error('No files found for pattern: %s', fullfile(folder, pattern));
    end
    names = {files.name};
    names = sort(names);
    filename = fullfile(folder, names{end});
end

function saveFigure(fig, baseName)
    % Save PNG and PDF. PDF is preferred for manuscripts.
    pngName = [baseName, '.png'];
    pdfName = [baseName, '.pdf'];

    try
        exportgraphics(fig, pngName, 'Resolution', 600, 'BackgroundColor', 'white');
        exportgraphics(fig, pdfName, 'ContentType', 'vector', 'BackgroundColor', 'white');
    catch
        print(fig, pngName, '-dpng', '-r600');
        print(fig, pdfName, '-dpdf', '-painters');
    end
end

function plotPanelA(figA, T, fingers, sensors, intendedSensorIndex, sensorColors, yLim, plotAll7, shadeBend)
    tiledlayout(figA, numel(fingers), 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    for f = 1:numel(fingers)
        ax = nexttile;
        plotOneFingerBlock(ax, T, fingers{f}, sensors, intendedSensorIndex, sensorColors, yLim, plotAll7, shadeBend, f == 1);
        if f < numel(fingers)
            ax.XTickLabel = [];
            xlabel(ax, '');
        end
    end
end

function plotOneFingerBlock(ax, T, finger, sensors, intendedSensorIndex, sensorColors, yLim, plotAll7, shadeBend, showLegend)
    target = string(T.target_finger);
    phase = string(T.phase);

    mask = target == string(finger) & (phase == "open" | phase == "bend");
    Tb = T(mask, :);

    if isempty(Tb)
        text(ax, 0.5, 0.5, sprintf('No data for %s', finger), 'Units', 'normalized', 'HorizontalAlignment', 'center');
        return;
    end

    t = Tb.protocol_elapsed_s - min(Tb.protocol_elapsed_s);

    intendedIdx = intendedSensorIndex(finger);
    if plotAll7
        channelsToPlot = 1:numel(sensors);
    else
        channelsToPlot = intendedIdx;
    end

    hold(ax, 'on');
    set(ax, 'Color', 'w', 'FontName', 'Arial', 'FontSize', 9, 'XColor', 'k', 'YColor', 'k', ...
        'Box', 'on', 'LineWidth', 0.8);

    if isempty(yLim)
        vals = [];
        for i = channelsToPlot
            col = sprintf('%s_dR_over_R0', sensors{i});
            vals = [vals; Tb.(col)]; %#ok<AGROW>
        end
        finiteVals = vals(isfinite(vals));
        if isempty(finiteVals)
            yLimUse = [-0.1 0.1];
        else
            yMin = prctile(finiteVals, 1);
            yMax = prctile(finiteVals, 99);
            pad = 0.1 * max(0.1, yMax - yMin);
            yLimUse = [yMin - pad, yMax + pad];
        end
    else
        yLimUse = yLim;
    end
    ylim(ax, yLimUse);

    if shadeBend
        cycles = unique(Tb.cycle(phase(mask) == "bend"));
        cycles = cycles(:)';
        for c = cycles
            m = Tb.cycle == c & string(Tb.phase) == "bend";
            if any(m)
                x1 = min(Tb.protocol_elapsed_s(m) - min(Tb.protocol_elapsed_s));
                x2 = max(Tb.protocol_elapsed_s(m) - min(Tb.protocol_elapsed_s));
                patch(ax, [x1 x2 x2 x1], [yLimUse(1) yLimUse(1) yLimUse(2) yLimUse(2)], ...
                    [0.92 0.96 1.00], 'EdgeColor', 'none', 'FaceAlpha', 0.75, 'HandleVisibility', 'off');
            end
        end
    end

    for i = channelsToPlot
        col = sprintf('%s_dR_over_R0', sensors{i});
        if i == intendedIdx
            lw = 2.3;
        else
            lw = 1.0;
        end
        plot(ax, t, Tb.(col), 'Color', sensorColors(i, :), 'LineWidth', lw, 'DisplayName', sensors{i});
    end

    yline(ax, 0, '-', 'Color', [0.25 0.25 0.25], 'LineWidth', 0.7, 'HandleVisibility', 'off');
    grid(ax, 'on');
    ax.GridColor = [0.85 0.85 0.85];
    ax.GridAlpha = 0.7;

    title(ax, sprintf('%s cyclic bending', finger), 'FontName', 'Arial', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    ylabel(ax, '\DeltaR/R_0', 'FontName', 'Arial', 'FontSize', 9, 'Color', 'k');
    xlabel(ax, 'Time within finger block (s)', 'FontName', 'Arial', 'FontSize', 9, 'Color', 'k');

    if showLegend && plotAll7
        lgd = legend(ax, 'Location', 'eastoutside');
        lgd.Box = 'off';
        lgd.FontSize = 7;
        lgd.TextColor = 'k';
    end
end

function plotPanelB(ax, heatData, fingers, cLim)
    imagesc(ax, heatData);
    axis(ax, 'tight');
    set(ax, 'Color', 'w', 'FontName', 'Arial', 'FontSize', 9, 'XColor', 'k', 'YColor', 'k', ...
        'Box', 'on', 'LineWidth', 0.8);

    colormap(ax, blueWhiteRed(256));
    if isempty(cLim)
        maxAbs = max(abs(heatData(:)), [], 'omitnan');
        if isempty(maxAbs) || maxAbs == 0 || ~isfinite(maxAbs)
            maxAbs = 0.5;
        end
        caxis(ax, [-maxAbs maxAbs]);
    else
        caxis(ax, cLim);
    end

    cb = colorbar(ax);
    cb.Label.String = 'Mean intended \DeltaR/R_0';
    cb.Label.FontName = 'Arial';
    cb.Label.Color = 'k';
    cb.Color = 'k';

    ax.XTick = 1:5;
    ax.XTickLabel = compose('C%d', 1:5);
    ax.YTick = 1:numel(fingers);
    ax.YTickLabel = fingers;
    xlabel(ax, 'Cycle number', 'FontName', 'Arial', 'Color', 'k');
    ylabel(ax, 'Bending gesture', 'FontName', 'Arial', 'Color', 'k');
    title(ax, 'Cycle-level intended response', 'FontName', 'Arial', 'FontWeight', 'bold', 'Color', 'k');

    % Add numeric labels.
    for r = 1:size(heatData, 1)
        for c = 1:size(heatData, 2)
            val = heatData(r, c);
            if isfinite(val)
                if abs(val) > 0.45
                    txtColor = 'w';
                else
                    txtColor = 'k';
                end
                text(ax, c, r, sprintf('%.2f', val), 'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', 'FontName', 'Arial', 'FontSize', 8, ...
                    'Color', txtColor);
            end
        end
    end
end

function plotPanelC(ax, means, sds, individualPoints, fingers, barColors)
    hold(ax, 'on');
    set(ax, 'Color', 'w', 'FontName', 'Arial', 'FontSize', 9, 'XColor', 'k', 'YColor', 'k', ...
        'Box', 'on', 'LineWidth', 0.8);

    b = bar(ax, 1:numel(fingers), means, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.6);
    for i = 1:numel(fingers)
        b.CData(i, :) = barColors(i, :);
    end

    errorbar(ax, 1:numel(fingers), means, sds, 'k', 'LineStyle', 'none', 'LineWidth', 1.0, 'CapSize', 7);

    % Overlay individual cycle/run values.
    rng(4);
    for i = 1:numel(fingers)
        vals = individualPoints{i};
        vals = vals(isfinite(vals));
        jitter = (rand(size(vals)) - 0.5) * 0.18;
        scatter(ax, i + jitter, vals, 18, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.6, 'MarkerFaceAlpha', 0.75, 'MarkerEdgeAlpha', 0.75);
    end

    yline(ax, 0, '-', 'Color', [0.25 0.25 0.25], 'LineWidth', 0.8);
    grid(ax, 'on');
    ax.GridColor = [0.85 0.85 0.85];
    ax.GridAlpha = 0.7;

    ax.XTick = 1:numel(fingers);
    ax.XTickLabel = fingers;
    xtickangle(ax, 30);
    ylabel(ax, 'Intended sensor \DeltaR/R_0', 'FontName', 'Arial', 'Color', 'k');
    title(ax, 'Repeatability across runs and cycles', 'FontName', 'Arial', 'FontWeight', 'bold', 'Color', 'k');
end

function cmap = blueWhiteRed(n)
    if nargin < 1
        n = 256;
    end
    half = floor(n / 2);
    blue = [0.10 0.25 0.75];
    white = [1.00 1.00 1.00];
    red = [0.75 0.10 0.10];

    c1 = [linspace(blue(1), white(1), half)', linspace(blue(2), white(2), half)', linspace(blue(3), white(3), half)'];
    c2 = [linspace(white(1), red(1), n - half)', linspace(white(2), red(2), n - half)', linspace(white(3), red(3), n - half)'];
    cmap = [c1; c2];
end

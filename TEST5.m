% Specify the file names
outputFilename = 'state_cur_log.csv';

% Read the input and output data from the CSV files
outputData = readmatrix(outputFilename);

% Check dimensions of inputData and outputData
[numOutputRows, numOutputCols] = size(outputData);

% Creating a figure for combined plots
figure;

sgtitle('Combined Output Data per Column');

for i = 1:numOutputCols
    subplot(11, 3, i);
    hold on;
    plot(1:200, outputData(1:200, i), 'r-');
    
    title(sprintf('Column %d: Output (red)', i));
    xlabel('Index');
    ylabel('Value');
end
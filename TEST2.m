% Specify the file names
outputFilename = 'policy_output_3.csv';
outputFilename2 = 'output_file_test_3.csv';


% Read the input and output data from the CSV files
outputData = readmatrix(outputFilename);
outputData2 = readmatrix(outputFilename2);

% Check dimensions of inputData and outputData
[numOutputRows, numOutputCols] = size(outputData);

% Creating a figure for combined plots
figure;

sgtitle('Combined Input and Output Data per Column');

for i = 1:numOutputCols
    subplot(3, 2, i);
    hold on;
    plot(1:736, outputData(:, i), 'r-');
    % plot(:, outputData2(:, i), 'b-');
    title(sprintf('Column %d: Output (red)', i));
    xlabel('Index');
    ylabel('Value');
    legend('Output from python', 'output from cpp');
end


figure;

for i = 1:numOutputCols
    subplot(3, 2, i);
    hold on;
    % plot(:, outputData(:, i), 'r-');
    plot(1:16000, outputData2(1:16000, i), 'b-');
    title(sprintf('Column %d: Output (red)', i));
    xlabel('Index');
    ylabel('Value');
    legend('Output from python', 'output from cpp');
end
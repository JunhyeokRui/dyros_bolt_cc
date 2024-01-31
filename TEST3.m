% Specify the file names
outputFilename = 'policy_output.csv';
outputFilename2 = 'output_file_test.csv';
outputFilename3 = 'policy_input.csv';
outputFilename4 = 'input_file_test.csv';

% Read the input and output data from the CSV files
outputData = readmatrix(outputFilename);
outputData2 = readmatrix(outputFilename2);
outputData3 = readmatrix(outputFilename3);
outputData4 = readmatrix(outputFilename4);

% Check dimensions of inputData and outputData
[numOutputRows, numOutputCols] = size(outputData);
[numOutputRows3, numOutputCols3] = size(outputData3);

% Creating a figure for combined plots
figure;

sgtitle('Combined Output Data per Column');

for i = 1:numOutputCols
    subplot(3, 2, i);
    hold on;
    plot(1:300, outputData(1:300, i), 'r-');
    plot(1:300, outputData2(1:300, i), 'b-');
    title(sprintf('Column %d: Output (red)', i));
    xlabel('Index');
    ylabel('Value');
    legend('Output from python', 'output from cpp');
end

figure;

sgtitle('Combined Input Data per Column');

for i = 1:numOutputCols3
    subplot(11, 3, i);
    hold on;
    plot(1:300, outputData3(1:300, i), 'r-');
    plot(1:300, outputData4(1:300, i), 'b-');
    title(sprintf('Column %d: INPUT (red)', i));
    xlabel('Index');
    ylabel('Value');
    legend('INPUT from python', 'INPUT from cpp');
end
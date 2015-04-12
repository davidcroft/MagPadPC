csvFile = csvread('trainingSetCol.csv', 1, 0);
[row, col] = size(csvFile);
csvFileNew = zeros(row, col+360);
csvFileNew(:, 1:96) = csvFile(:, 1:96);
csvFileNew(:, 457:col+360) = csvFile(:, 97:col);
for r = 1:row
    index = 97;
    for c = 6:20
        for i = c+1:21
            csvFileNew(r, index) = csvFile(r, c) / csvFile(r, i);
            index = index+1;
        end
    end
    for c = 38:52
        for i = c+1:53
            csvFileNew(r, index) = csvFile(r, c) / csvFile(r, i);
            index = index+1;
        end
    end
    for c = 70:84
        for i = c+1:85
            csvFileNew(r, index) = csvFile(r, c) / csvFile(r, i);
            index = index+1;
        end
    end
end
csvwrite('trainingSetColNew.csv',csvFileNew);

csvFile = csvread('trainingSetRow.csv', 1, 0);
[row, col] = size(csvFile);
csvFileNew = zeros(row, col+360);
csvFileNew(:, 1:96) = csvFile(:, 1:96);
csvFileNew(:, 457:col+360) = csvFile(:, 97:col);
for r = 1:row
    index = 97;
    for c = 6:20
        for i = c+1:21
            csvFileNew(r, index) = csvFile(r, c) / csvFile(r, i);
            index = index+1;
        end
    end
    for c = 38:52
        for i = c+1:53
            csvFileNew(r, index) = csvFile(r, c) / csvFile(r, i);
            index = index+1;
        end
    end
    for c = 70:84
        for i = c+1:85
            csvFileNew(r, index) = csvFile(r, c) / csvFile(r, i);
            index = index+1;
        end
    end
end
csvwrite('trainingSetRowNew.csv',csvFileNew);

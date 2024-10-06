#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>
#include<fstream>
#include<sstream>
#include<unordered_map>
#include<tuple>
#include<omp.h>
#include<chrono>

using namespace std;

unordered_map<string, int> label_encode(const vector<string>& labels) {
    unordered_map<string, int> label_map;
    vector<int> encoded_labels;
    int label_value = 0;

    for (const string& label : labels) {

        if (label_map.find(label) == label_map.end()) {
            label_map[label] = label_value;
            label_value++;
        }
        encoded_labels.push_back(label_map[label]);
    }

    return label_map;
}

tuple<vector<vector<float>>, vector<int>> read_data(const string& csv_filename, const unordered_map<string, int>& encoded_labels){
    vector<vector<float>> result;
    vector<int> label_string;
    ifstream file(csv_filename);
    string line;
    if (getline(file, line)) {
        cout << "Skipping header: " << line << endl;
    }
    while (getline(file, line)){
        stringstream ss(line);
        string value;
        vector<float> row_values;

        int current_column = 0;
        while (getline(ss, value, ',')){
            if (current_column== 4){
                auto it = encoded_labels.find(value);
                if (it != encoded_labels.end()) {
                    label_string.push_back(it->second);
                } else {
                    cerr << "Label not found in encoded_labels: " << value << endl;
                    label_string.push_back(-1);
                }
                break;

            }
            row_values.push_back(stof(value));
            
            current_column ++;

        }
        result.push_back(row_values);
    }
    
    return make_tuple(result, label_string);

    
}

float computeEucliedean(const vector<float>& X1, const vector<float>& X2){
    float distance_val = 0.0;
    for (int i=0; i<X1.size(); i++){
        distance_val += pow(X2[i]-X1[i], 2);

    }
    distance_val = sqrt(distance_val);

    return distance_val;
}

int kNNClassify(const vector<vector<float>>& trainData, const vector<int>& trainLabels, const vector<float> testData, int k){
    int labelMatched = -1;
    int maxLabelCount = 0;
    vector<pair<float, int>> distance_map;
    unordered_map<int, int> labelcount_map;
    vector<long long> runtimes;

    for (int i=0 ; i<trainData.size(); i++){
        float distance_comp = computeEucliedean(trainData[i], testData);
        auto start = chrono::steady_clock::now();

        distance_map.push_back(make_pair(distance_comp, trainLabels[i]));

        auto end = chrono::steady_clock::now();
        long long time_taken = chrono::duration_cast<chrono::microseconds>(end - start).count();

        runtimes.push_back(time_taken);
    }

    long long total_time = 0;
    for (long long runtime : runtimes) {
        total_time += runtime;
    }

    double average_time = static_cast<double>(total_time) / runtimes.size();

    cout << "Average time for Euclidean distance computation: " << average_time << " microseconds" << endl;

    sort(distance_map.begin(), distance_map.end());

    for (int i=0; i<k; i++){
        labelcount_map[distance_map[i].second]++;
    }

    for (const auto& label : labelcount_map){
        if (label.second > maxLabelCount){
            maxLabelCount = label.second;
            labelMatched = label.first;
        }
    }

    return labelMatched;
}

int main(){

    string in_filename = "data.csv";
    string test_filename = "test.csv";
    vector<string> labels = {"setosa", "versicolor", "virginica"};
    unordered_map<string, int> encoded_labels = label_encode(labels);

    tuple<vector<vector<float>>, vector<int>> processes_data = read_data(in_filename, encoded_labels);
    // tuple<vector<vector<float>>, vector<int>> process_test_data = read_data(test_filename, encoded_labels);
    vector<float> test_data_sample = {5.9,3.0,5.1,1.8};
    
    int predictedLabel = kNNClassify(get<0>(processes_data), get<1>(processes_data), test_data_sample, 2);
    
    cout<<"Predicted Label: " <<predictedLabel<<endl;

    
    return 0;
}
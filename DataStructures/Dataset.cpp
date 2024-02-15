#include <fstream>
#include <sstream>
#include <functional>
#include <limits>
#include <cmath>
#include <algorithm>

#include "Dataset.h"

/********************************************** CONSTRUCTORS ************************************************/

Dataset::Dataset(const std::string& path, char delimiter){

    // initialize variable
    std::string tmp_row, element;
    std::ifstream fin(path);

    // check existence
    if (!fin) {
        std::cerr << "File " << path << " does not exist" << std::endl;
        exit(1);
    }

    // read first line and save column names
    std::getline(fin, tmp_row);
    std::istringstream my_stream(tmp_row);
    bool is_first_column = true;
    while ( std::getline(my_stream, element, delimiter) ){
        // first column contains the indexes
        if (is_first_column){
            is_first_column = false;
        }
        // save the name of the other columns
        else{
            col_names.push_back( element );
            minmax_coefficients.emplace_back(0., 1.);
        }
    }

    //set number of columns
    n_cols = col_names.size();

    // read file and fill data structure
    while ( std::getline(fin, tmp_row) ){

        // create stream by the row that has just been read
        std::istringstream my_stream1(tmp_row);
        // and initialize a suitable vector
        std::vector<NN::type> a_row_of_data(n_cols);

        // saving elements of the row
        unsigned i = 0;
        while ( std::getline(my_stream1, element, delimiter) && i <= n_cols){
            if (i == 0){
                row_index.push_back(element);
                ++i;
            }
            else {
                a_row_of_data[i-1] = std::stod(element);
                ++i;
            }
        }

        // adding the row to the data structure
        data.push_back(a_row_of_data);
    }

}


/********************************************** METHODS *****************************************************/


unsigned Dataset::get_nrows(void) const{
    return data.size();
}


std::vector<std::string> Dataset::get_datetime(void) const{
    return row_index;
}


std::vector<std::pair<NN::type, NN::type>> Dataset::get_minmax(void) const{
    return minmax_coefficients;
}


std::pair<NN::type, NN::type> Dataset::get_minmax(std::string selected_column) const{
    const unsigned IDX = find_column(selected_column);
    return minmax_coefficients[IDX];
}


std::vector<unsigned> Dataset::reduce_dataset(std::string compare, std::string value){

    // select comparison method
    std::function<bool(std::string, std::string)> comp_method;

    // (switch does not work with std::strings)
    if (compare == "<")
        comp_method = std::less<>();
    else if (compare == "<=")
        comp_method = std::less_equal<>();
    else if (compare == ">")
        comp_method = std::greater<>();
    else if (compare == ">=")
        comp_method = std::greater_equal<>();
    else if (compare == "==")
        comp_method = std::equal_to<>();
    else if (compare == "!=")
        comp_method = std::not_equal_to<>();
    else{
        std::cerr << "Chosen compare method is not acceptable" << std::endl;
        exit(3);
    }

    // allocate data structure
    std::vector<std::vector<NN::type>> data_tmp;
    std::vector<std::string> row_index_tmp;
    std::vector<unsigned> accepted_indexes;

    // process the current data and select the columns that fulfill the condition,
    // using row_index as filter
    const unsigned LEN = data.size();
    for (unsigned i = 0; i < LEN; ++i)
        if (comp_method(row_index[i], value)){
            accepted_indexes.push_back(i);  // storing index for return
            data_tmp.push_back(data[i]);            // updating data
            row_index_tmp.push_back(row_index[i]);  // updating date
        }

    //swapping elements
    std::swap(data_tmp, data);
    std::swap(row_index_tmp, row_index);

    return accepted_indexes;

}


std::vector<unsigned> Dataset::reduce_dataset(std::string selected_column, std::string compare, NN::type value){

    // select comparison method
    std::function<bool(NN::type, NN::type)> comp_method;

    // (switch does not work with std::strings)
    if (compare == "<")
        comp_method = std::less<>();
    else if (compare == "<=")
        comp_method = std::less_equal<>();
    else if (compare == ">")
        comp_method = std::greater<>();
    else if (compare == ">=")
        comp_method = std::greater_equal<>();
    else if (compare == "==")
        comp_method = std::equal_to<>();
    else if (compare == "!=")
        comp_method = std::not_equal_to<>();
    else{
        std::cerr << "Chosen compare method is not acceptable" << std::endl;
        exit(3);
    }

    // allocate data structure
    std::vector<std::vector<NN::type>> data_tmp;
    std::vector<std::string> row_index_tmp;
    std::vector<unsigned> accepted_indexes;

    // process the current data and select the columns that fulfill the condition,
    // using the selected column as filter
    const unsigned LEN = data.size();
    const unsigned IDX = find_column(selected_column);

/*
    for (unsigned i = 0; i < LEN; ++i)
        std::cout << data[i][IDX] << std::endl;
    std::cout << "\n" << IDX << std::endl;
*/

    for (unsigned i = 0; i < LEN; ++i)
        if (comp_method(data[i][IDX], value)){
            accepted_indexes.push_back(i);    // storing index for return
            data_tmp.push_back(data[i]);              // updating data
            row_index_tmp.push_back(row_index[i]);    // updating date
        }

    //swapping elements
    std::swap(data_tmp, data);
    std::swap(row_index_tmp, row_index);

    return accepted_indexes;

}


void Dataset::apply_log(const std::string& selected_column){

    const unsigned IDX = find_column(selected_column);
    const unsigned LEN = data.size();
    for (unsigned i = 0; i < LEN; ++i)
        data[i][IDX] = log(data[i][IDX]);

}


void Dataset::normalize_dataframe(const std::vector<std::pair<NN::type, NN::type>>& ext_minmax){

    // store number of rows
    const unsigned LEN = data.size();
    for (unsigned cl = 0; cl < n_cols; ++cl){
        const NN::type mini = ext_minmax[cl].first;
        const NN::type maxi = ext_minmax[cl].second;
        for (unsigned rw = 0; rw < LEN; ++rw)
            data[rw][cl] = (data[rw][cl] - mini) / (maxi - mini);
    }

    // store used minmax_coefficients
    minmax_coefficients = ext_minmax;

}

void Dataset::normalize_dataframe(const std::vector<std::string>& except_columns){

    // store number of rows
    const unsigned LEN = data.size();

    // compute minimum and maximum
    for (unsigned cl = 0; cl < n_cols; ++cl){

        // if column is not exception
        if ( std::find(except_columns.cbegin(), except_columns.cend(), col_names[cl]) == except_columns.cend() ){

            // initialize with first value
            NN::type mini = data[0][cl], maxi = data[0][cl];

            // identify minimum and maximum
            for (unsigned rw = 1; rw < LEN; ++rw){
                if (data[rw][cl] < mini)
                    mini = data[rw][cl];
                else if (data[rw][cl] > maxi)
                    maxi = data[rw][cl];
            }

            // store min and max
            minmax_coefficients[cl].first = mini;
            minmax_coefficients[cl].second = maxi;
        }
    }

    // minmax normalize using the member minmax_coefficients
    normalize_dataframe(minmax_coefficients);

}


std::vector<NN::type> Dataset::extract_vector(const std::string& selected_column) const{

    const unsigned IDX = find_column(selected_column);

    const unsigned LEN = data.size();
    std::vector<NN::type> out_vec( LEN );

    for (unsigned i = 0; i < LEN; ++i)
        out_vec[i] = data[i][IDX];

    return out_vec;

}


std::vector<VectorClass> Dataset::extract_matrix(const std::vector<std::string>& selected_columns) const{

    const unsigned LEN = data.size();
    const unsigned N_SELCOLS = selected_columns.size();

    std::vector<unsigned> IDX(N_SELCOLS);

    unsigned ii = 0;
    for (const auto& col : selected_columns)
        IDX[ii++] = find_column(col);

    std::vector<VectorClass> out_mtrx( LEN );
    for (unsigned i = 0; i < LEN; ++i) {
        VectorClass tmp(N_SELCOLS);
        for (unsigned j = 0; j < N_SELCOLS; ++j)
            tmp(j) = data[i][IDX[j]];
        out_mtrx[i] = tmp;
    }

    return out_mtrx;

}


unsigned Dataset::find_column(const std::string& col) const{

    for (std::size_t i = 0; i < col_names.size(); ++i)
        if(col == col_names[i])
            return i;

    // exit
    std::cerr << "Column name does not exist" << std::endl;
    exit(4);
}

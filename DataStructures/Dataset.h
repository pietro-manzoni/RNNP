/// Import and store an external dataset
/**
 * Class that imports a dataset from a given .csv file. \n
 * Provides methods for reducing the size of the dataset, for extracting columns
 * and for MinMax normalize columns.
 * \note It is thus fundamental for the .csv to include a header row with the name of the columns.
 */

#ifndef DNAX_LIBRARY_DATASET_H
#define DNAX_LIBRARY_DATASET_H

#include <vector>
#include <list>

#include "Matrix.h"
#include "VectorClass.h"

class Dataset {

    /********************************************** ATTRIBUTES **************************************************/

private:

    /// Data structure to collect the imported dataset.
    /**
     * The content of the csv is imported and stored here. The \a STL \a vector interface provides
     * a more usable and versatile solution for this kind of operation, with respect to the implemented
     * Matrix and VectorRn class, and is therefore chosen for the purpose.
     */
    std::vector< std::vector<NN::type> > data;

    /// Vector containing the name of the columns.
    /**
     * The names of the columns in .csv file are here stored during the construction of the object. \n
     * \note It is thus fundamental for the .csv to include a header row with the name of the columns.
     */
    std::vector<std::string> col_names;

    /// Number of columns of the dataset.
    /**
     * Stored for convenience and checks. \n
     * Both #col_names and #col_names are vector with length #n_cols.
     */
    unsigned n_cols;

    /// Vector containing the date
    /**
     * The names of the columns in .csv file are here stored during the construction of the object. \n
     * \note It is thus fundamental for the .csv to include a header row with the name of the columns.
     */
    std::vector<std::string> row_index;

    /// Vector indicating the current normalization coefficients
    /**
     * Each column is supposed to be normalized as:
     * normalized_value = (value - min) / (max - min).
     * This vector contains a pair (min, max) for each column.
     * Default values are (0,1), i.e. the identity transformation.
     */
    std::vector<std::pair<NN::type, NN::type>> minmax_coefficients;

    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor of the object.
    /**
     * Read all the elements contained in a given .csv file and store them in the #data structure.
     * @param path: path to the .csv file
     * @param delimiter: .csv file delimiter for the elements in the same row. Default is comma.
     */
    // WARNING: csv file MUST contain the HEADER
    explicit Dataset(const std::string& path, char delimiter = ',');


    /********************************************** METHODS *****************************************************/

public:

    /// Get number of rows
    /**
     * Return the number of rows
     */
    unsigned get_nrows(void) const;

    /// Get dateteime (i.e. row_index)
    /**
     * Return row_index.
     */
    std::vector<std::string> get_datetime(void) const;

    /// Return minmax vector
    /**
     * Return a copy of the minmax vector.
     */
    std::vector<std::pair<NN::type, NN::type>> get_minmax(void) const;

    /// Return minmax pair for a specified column
    /**
     * Return the copy minmax pair for a selected column
     */
    std::pair<NN::type, NN::type> get_minmax(std::string selected_column) const;

    /// Reduce number of the rows using the index of the dataframe.
    /**
    * Keep just the relevant rows (the ones that fulfill the indicated relationship) and discard the others.
    * Selection is performed using the row_index vector.
    * @param compare: std::string representing the comparison relationship. Can be
    * - ">"   (greater)
    * - ">="  (greater or equal)
    * - "<"   (less)
    * - "<="  (less or equal)
    * - "=="  (equal)
    * - "!="  (unequal)
    * @param value: comparison date. Should be in the format "YYYY-MM-DD hh:mm:dd"
    */
    std::vector<unsigned> reduce_dataset(std::string compare, std::string value);

    /// Reduce number of the rows using a specified column of the dataset.
    /**
    * Keep just the relevant rows (the ones that fulfill the indicated relationship) and discard the others.
    * Selection is performed using the row_index vector.
    * @param selected_column: reference column.
    * @param compare: std::string representing the comparison relationship. Can be
    * - ">"   (greater)
    * - ">="  (greater or equal)
    * - "<"   (less)
    * - "<="  (less or equal)
    * - "=="  (equal)
    * - "!="  (unequal)
    * @param value: comparison value.
    */
    std::vector<unsigned> reduce_dataset(std::string selected_column, std::string compare, NN::type value);

    /// Apply logarithm to a given column
    /**
     * It is assumed that all the values are positive, no checks are performed.
     * @param selected_column: string containing the name of the needed column
     */
    void apply_log(const std::string& selected_column);

    /// Minmax normalize dataframe
    /**
     * Minmax normalize each column of a dataframe
     */
    void normalize_dataframe(const std::vector<std::string>& except_columns);

    /// Minmax normalize dataframe using external minmax vector
    /**
     * Minmax normalize each column of a dataframe using external minmax vector
     * @param ext_minmax: external minmax vector
     */
    void normalize_dataframe(const std::vector<std::pair<NN::type, NN::type>>& ext_minmax);

    /// Extract the column that corresponds to the provided string.
    /**
     * @param selected_column: string containing the name of the needed column
     * @return vector of suitable size containing the value
     */
    std::vector<NN::type> extract_vector(const std::string& selected_column) const;

    /// Extract the columns that correspond to the provided strings.
    /**
     * @param selected_columns: vector of strings containing the names of the needed columns
     * @return matrix of suitable size containing the value
     */
    std::vector<VectorClass> extract_matrix(const std::vector<std::string>& selected_columns) const;

private:

    /// Find the index of the column with a given name
    /**
     * Utility method. Given a certain string containing name of a column, returns the position index of the
     * corresponding column in #col_names. \n
     * Since the columns are supposed to be in a reasonably small number, a naive vector structure is chosen,
     * and binary search is not involved.
     * @param selected_column: string containing the name of the column
     * @return position index
     */
    unsigned find_column(const std::string& selected_column) const;

};


#endif //DNAX_LIBRARY_DATASET_H

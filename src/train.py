
import os
import numpy as np
import pandas as pd
import yaml
from functools import partial



current_path = os.getcwd()
PROJECT_ROOT = "/".join(current_path.split("/")[:-1]) +"/"


def read_yaml(path: str) -> dict:
    """Returns YAML file as dict"""
    with open(path, 'r') as file_in:
        config = yaml.safe_load(file_in)
    return config

def _rename_columns(name, cohort):
    new_name = name + '@' + cohort
    return new_name

class DataSet:
    """Gathers data from RNA-seq or microarrays.

    Attributes
    ----------
    dataset_name_list: list of str
        A list of subfolder names in the data folder corresponding to the datasets to be used.
    features: pandas.DataFrame
        A dataframe gathering RNA expression data for every patient. Rows correspond to patients and columns to genes or probes.
    metadata: pandas.DataFrame
        A dataframe gathering information about every patient in the dataset. Rows correspond to patients and columns to clinical features.
    common_predictors: list of str
        The list of common predictors (genes or probes) for which expression data is available in every dataset.
    
    Methods
    -------
    load_data()
        Loads data from all datasets in dataset_name_list.
    get_smaller_subset(restrictions)
        Returns smaller subsets with restrictions on patients and predictors.
    """

    def __init__(self, dataset_name_list, dictionary_list=None):
        """
        Parameters:
        -----------
        dataset_name_list: list of str
            A list of subfolder names in the data folder corresponding to the datasets to be used.
        dictionary_list: list of str
            A list of dictionaries to use to convert predictor names for each dataset.
        features: pandas.DataFrame
            A dataframe gathering RNA expression data for every patient.
        metadata: pandas.DataFrame
            A dataframe gathering information about every patient in the dataset. 
        """

        self.dataset_name_list = dataset_name_list
        if dictionary_list:
            self.dictionary_list = dictionary_list
        else:
            self.dictionary_list = [None for _ in dataset_name_list]
        self.features = None
        self.metadata = None
        self.common_predictors = None

    def _find_common_predictors(self): 
        """Computes the list of all predictors that are common to all datasets in dataset_name_list.    

        Returns:
        --------
        common_predictors: list of str
            The list of all predictors that are common to all datasets in dataset_name_list. 
        """

        common_predictors = self._load_predictors(self.dataset_name_list[0])
        common_predictors = self._translate(common_predictors, self.dictionary_list[0])
        for dataset_name, dictionary_name in zip(self.dataset_name_list, self.dictionary_list):
            predictors = self._load_predictors(dataset_name)
            translated_predictors = self._translate(predictors, dictionary_name)
            common_predictors = self._intersection(common_predictors, translated_predictors)
        common_predictors.sort()

        return common_predictors
    
    def _translate(self, predictors, dictionary_name):
        """"Translates predictor names using prescribed dictionary.
        
        Parameters:
        -----------
        predictors: list of str
            A list of predictor names to be translated.
        dictionary_name: str or None
            The name of the file containing a mapping table from old to new predictor names.

        Returns:
        --------
        translated_predictors: list of str
            The list of translated predictor names.
        """

        if dictionary_name:
            dictionary = self._load_dictionary(dictionary_name)
            translated_predictors = [dictionary[predictor_name] for predictor_name in predictors]
            return translated_predictors
        else:
            return predictors

    def _load_dictionary(self, dictionary_name):
        """Loads required dictionary.

        Parameters:
        -----------
        dictionary_name: str
            The name of the file containing the required dictionary in PROJECT_ROOT/src/gene_translation/translators/.

        Returns:
        --------
        dictionary: pd.DataFrame
            A dictionary whose keys are the old predictors, and whose values are the new predictors.
        """

        path_to_dictionary = PROJECT_ROOT + "src/gene_translation/translators/" + dictionary_name
        dictionary_as_dataframe = pd.read_csv(filepath_or_buffer=path_to_dictionary,
                                 sep = ' ',
                                 header = None,
                                 names = ['old_predictors', 'new_predictors'])
        dictionary = {}
        for index, row in dictionary_as_dataframe.iterrows():
            old_predictor = row['old_predictors']
            new_predictor = row['new_predictors']
            dictionary[old_predictor] = new_predictor

        return dictionary

    def load_data(self):
        """Loads data from all datasets in dataset_name_list.
        """

        self.common_predictors = self._find_common_predictors()
        self.features = self._load_features()
        self.metadata = self._load_metadata()

    def _load_predictors(self, dataset_name):
        """Loads list of predictors corresponding to a particular dataset.

        Parameters:
        -----------
        dataset_name: str
            The name of the subfolder in the data folder corresponding to the considered dataset.

        Returns:
        --------
        predictors: list of str
            The list of predictors corresponding to the specified dataset.
        """

        if dataset_name.endswith('/'):
            dataset_name = dataset_name[:-1]
        if self.dataset_type == 'micro_array':
            path_to_data = PROJECT_ROOT + 'data/' + dataset_name + '/treated_files/probes.txt' 
        elif self.dataset_type.startswith('RNA-seq'):
            path_to_data = PROJECT_ROOT + 'data/' + dataset_name + '/treated_files/genes.txt'
        predictors = pd.read_csv(path_to_data, sep = '\t', header=None)[0].tolist()
        predictors.sort()

        return predictors

    def _load_features(self):
        """Loads features corresponding to all considered datasets.
        
        Returns:
        --------
        total_features: pandas.DataFrame
            A dataframe containing features corresponding to all considered datasets.
        """
        
        total_features = pd.DataFrame([])
        for dataset_name, dictionary_name in zip(self.dataset_name_list, self.dictionary_list):
            if dataset_name.endswith('/'):
                dataset_name = dataset_name[:-1]
            features = self._load_features_from_dataset(dataset_name, dictionary_name)
            if total_features.shape == (0, 0):
                total_features = features
            else:
                total_features = pd.concat([total_features, features], axis=0)
        total_features.sort_index(axis=0, inplace=True)
        total_features.sort_index(axis=1, inplace=True)

        return total_features
    
    def _load_features_from_dataset(self, dataset_name, dictionary_name):
        """Loads features corresponding to a particular dataset.

        Returns:
        --------
        features: pandas.DataFrame
            A dataframe containing features corresponding to a particular dataset.
        """
        
        if dataset_name.endswith('/'):
            dataset_name = dataset_name[:-1]
        path_to_data = PROJECT_ROOT + 'data/' + dataset_name + '/treated_files/counts.txt'
        features = pd.read_csv(path_to_data, sep = '\t')
        if dictionary_name:
            dictionary = self._load_dictionary(dictionary_name)
            features = features.rename(index=dictionary)
            features = features.loc[~features.index.duplicated(keep='first'), ]
        features = features.transpose()
        features = features[self.common_predictors]
        features.sort_index(axis=0, inplace=True)
        features.sort_index(axis=1, inplace=True)

        return features
    
    def _load_metadata(self):
        """Loads additional information on every patient in the considered datasets.

        Returns:
        --------
        total_metadata: pandas.DataFrame
            A dataframe containing additional information on every patient in the considered datasets.
        """

        total_metadata = pd.DataFrame([])
        for dataset_name in self.dataset_name_list:
            metadata = self._load_metadata_from_dataset(dataset_name)
            if total_metadata.shape == (0, 0):
                total_metadata = metadata
            else:
                total_metadata = pd.concat([total_metadata, metadata], axis=0)
        total_metadata.sort_index(inplace=True)
        total_metadata = total_metadata.fillna('NA')

        return total_metadata

    def _load_metadata_from_dataset(self, dataset_name):
        """Loads additional information on every patient in a particular dataset.

        The names of the features in the dataset will be changed according to the config file included in the dataset_name subfolder.

        Parameters:
        -----------
        dataset_name: str
            The name of the subfolder in the data folder containing the dataset of interest.

        Returns:
        --------
        total_metadata: pandas.DataFrame
            A dataframe containing additional information on every patient in a particular dataset.
        """

        path_to_data = PROJECT_ROOT + 'data/' + dataset_name + '/treated_files/metadata.txt'
        metadata = pd.read_csv(path_to_data, sep = '\t')
        metadata = metadata.transpose()
        config = self._load_config_from_dataset(dataset_name)
        metadata = self._rename_metadata(metadata, config)
        metadata.sort_index(inplace=True)

        return metadata

    def _load_config_from_dataset(self, dataset_name):
        """Loads config file giving information on a particular dataset.

        Parameters:
        -----------
        dataset_name: str
            The name of the subfolder in the data folder containing the dataset of interest.

        Returns:
        --------
        total_metadata: pandas.DataFrame
            A dataframe containing additional information on every patient in a particular dataset.
        """
        
        path_to_data = PROJECT_ROOT + 'data/' + dataset_name + '/treated_files/config.yml'
        with open(path_to_data, 'r') as file_in:
            config = yaml.safe_load(file_in)
        return config

    def _rename_metadata(self, metadata, config):
        """Changes names of the features and of the values of these features in the dataframe of metadata according to the information in the config file.

        Parameters:
        -----------
        metadata: pandas.DataFrame
            A dataframe containing additional information on every patient in a particular dataset. 
        config: dict
            A dictionary containing information about the way feature and value names should be modified.

        Returns:
        --------
        metadata: pandas.DataFrame
            The modified metadata dataframe.
        """

        for predictor_name in list(config.keys()):
            old_predictor_name = config[predictor_name]['name']
            metadata = metadata.rename(columns = {old_predictor_name: predictor_name})
            if config[predictor_name]['type'] == 'categorical':
                dictionary = config[predictor_name]['values']
                metadata[predictor_name] = [dictionary[item] for item in metadata[predictor_name]]

        return metadata

    def _intersection(self, list1, list2):
        """Computes the intersection between two lists.
        
        Parameters:
        -----------
        list1, list2: list
        """

        return [elt for elt in list1 if elt in list2]

    def get_smaller_subset(self, restrictions):
        """Returns a smaller subset with restrictions on patients and predictors.

        Parameters:
        -----------
        restrictions: dict
            A dictionary with two keys: 'probes'/'genes' (depending on the type of data) and 'patients'. 
            The value for 'probes'/'genes' is a list of probes/genes.
            The value for 'patients' is a dictionary whose keys are feature names corresponding to the metadata, and whose values are the values of these features to select.
        
        Returns:
        --------
        new_dataset: pandas.DataFrame
            Object DataSet with restrictions on patients and predictors.
        """

        if self.dataset_type == 'micro_array':
            new_dataset = MicroArray_DataSet(self.dataset_name_list)
            predictor_restrictions = restrictions['probes']
        patient_restrictions = restrictions['patients']
        
        new_dataset.common_predictors = self.common_predictors
        relevant_predictors = self._get_relevant_predictors(predictor_restrictions) 
        relevant_patients = self._get_relevant_patients(patient_restrictions)        
        new_dataset.common_predictors = relevant_predictors
        new_dataset.features = self._apply_restrictions_for_features(relevant_patients, relevant_predictors)
        new_dataset.metadata = self._apply_restrictions_for_metadata(relevant_patients)

        return new_dataset
    
    def _apply_restrictions_for_features(self, relevant_patients, relevant_predictors):
        """Transforms feature dataframe to take restrictions into account.

        Parameters:
        -----------
        relevant_patients: list
            List of patients to select.
        relevant_predictors: list
            List of predictors to select.

        Returns:
        --------
        new_features: pandas.DataFrame
            A dataframe of features restricted to prescribed patients and predictors.
        """
        
        new_features = self.features.loc[relevant_patients]
        new_features = new_features[relevant_predictors]
        new_features.sort_index(inplace=True)

        return new_features

    def _apply_restrictions_for_metadata(self, relevant_patients):
        """Transforms metadata dataframe to take restrictions into account.

        Parameters:
        -----------
        relevant_patients: list
            List of patients to select.
        
        Returns:
        --------
        new_features: pandas.DataFrame
            A dataframe of metadata restricted to prescribed patients.
        """

        new_metadata = self.metadata.loc[relevant_patients]
        new_metadata.sort_index(inplace=True)

        return new_metadata

    def _get_relevant_patients(self, patient_restrictions):
        """Returns the list of patients corresponding to given restrictions.
        
        Parameters:
        -----------
        patient_restrictions: dict
            A dictionary whose keys are feature names corresponding to the metadata, and whose values are the values of these features to select.
            For example, {'FEATURE': FEATURE_VALUE} corresponds to selecting patients for which FEATURE takes the value FEATURE_VALUE.

        Returns:
        --------
        relevant_patients: list
            The list of patients corresponding to prescribed restrictions on patients.
        """

        relevant_patients = self.metadata.index
        if patient_restrictions != None:
            for predictor_name in patient_restrictions:
                authorized_values = patient_restrictions[predictor_name]
                relevant_patients = [patient for patient in relevant_patients if self.metadata[predictor_name][patient] in authorized_values]

        return relevant_patients
    
    def _get_relevant_predictors(self, predictor_restrictions):
        """Returns the list of predictors corresponding to given restrictions.
        
        Parameters:
        -----------
        predictor_restrictions: list
            A list of predictors (probes or genes) to select, if they exist in the dataset.

        Returns:
        --------
        relevant_predictors: list
            The list of predictors to select.
        """

        relevant_predictors = self.common_predictors
        if predictor_restrictions != None:
            relevant_predictors = self._intersection(relevant_predictors, predictor_restrictions) 

        return relevant_predictors
    
    def _get_group_indices(self):
        """Returns a list with the number of the corresponding group for each predictor.

        A group is made of the indices corresponding to the same biological predictor accross different features. 
        
        Returns:
        --------
        group_indices: list

        """
    
        number_predictors = len(self.common_predictors)
        group_indices = [i for i in range(1, number_predictors+1) for _ in self.dataset_name_list]

        return group_indices
    
    def _get_features_in_blocks_by_cohort(self):
        """Transforms features to be handled by logistic regression with group sparsity.
        
        Predictors in the new feature matrix are predictors associated with a particular cohort. 
        For example, new predictors could be MTUS1@GSE123845 and MTUS1@GSE122639.

        Returns:
        --------
        features_by_cohort: pandas.DataFrame
            A dataframe of features.
        """

        features_by_cohort = pd.DataFrame([])
        for dataset_name in self.dataset_name_list:
            if dataset_name.endswith('/'):
                dataset_name = dataset_name[:-1]
            if self.dataset_type == 'micro_array':
                cohort_restrictions = {'patients': {'COHORT': dataset_name}, 'probes': None}
            elif self.dataset_type.startswith('RNA-seq'):
                cohort_restrictions = {'patients': {'COHORT': dataset_name}, 'genes': None}
            features = self.get_smaller_subset(cohort_restrictions).features
            _rename_columns_with_dataset_name = partial(_rename_columns, cohort = dataset_name)
            features.rename(_rename_columns_with_dataset_name, axis='columns', inplace=True) 
            if features_by_cohort.shape == (0, 0):
                features_by_cohort = features
            else:
                features_by_cohort = pd.concat([features_by_cohort, features], axis=0)
        features_by_cohort.sort_index(axis=0, inplace=True)
        features_by_cohort.sort_index(axis=1, inplace=True)

        return features_by_cohort.fillna(0)

    def _convert_to_(self, predictors_to_convert, predictor_type):
        """Converts a list of genes to probes or probes to genes.
        
        Parameters:
        -----------
        predictors_to_convert: list
            A list of probes or genes to convert.
        predictor_type: str
            A string 'probes' or 'genes' giving the type of the elements in the list predictors_to_convert.

        Returns:
        --------
        converted_predictors: list
            A list of probes or genes.
        """

        mapping_table = self._load_mapping_table()
        if predictor_type == 'genes':
            origin_predictor_type = 'probes'
        elif predictor_type == 'probes':
            origin_predictor_type = 'genes'
        converted_predictors = mapping_table.loc[mapping_table[origin_predictor_type].isin(predictors_to_convert), predictor_type]
        converted_predictors = converted_predictors.unique().tolist()
        converted_predictors.sort()

        return converted_predictors
            
    def _load_mapping_table(self):
        """Loads table mapping probes to genes.
        
        Returns:
        --------
        mapping_table: pandas.DataFrame
            A dataframe with columns 'probes' and 'genes', where most rows match a probe to a gene.
        """

        file_name = '../data/probes_to_genes.txt'
        mapping_table = pd.read_csv(file_name, sep='\t', header=None, names = ['probes', 'genes'])

        return mapping_table

class MicroArray_DataSet(DataSet):
    """Gathers data from RNA-seq.

    Child of the DataSet class.
    """

    def __init__(self, dataset_name_list, dictionary_list=None):
        self.dataset_type = 'micro_array'
        DataSet.__init__(self, dataset_name_list, dictionary_list)

    


#if __name__ == "__main__":


import pandas as pd
import sys
import os

from ReadAndWrite.Name_Reader import ReadNames
from ReadAndWrite.Write import WriteResult
from Extraction.ExtractFluorescence import FluorescenceProcessor
from Extraction.AbsorptionCoordinator import AbsorptionCoordinator



class Coordinator:

    def __init__(self, file_path, Version):
        self.Version = Version
        self.file_path = file_path
        self.result = {}
        self.metadata = []


    def get_result(self):
        df = self._read_file() # 1: Read the file
        sample_names, sample_methods, sample_positions, sample_dilutions = self._read_names(df) # 2: Read the names, methods, and positions from the file
        write = WriteResult(self.file_path)
        self.result = write.final_result_dict(sample_names)  # 3: Create a result dict with the sample names as keys, here the values are all none and will be updated if evaluated
        self.coordinate_extraction(sample_names, sample_methods, sample_positions, sample_dilutions, df)
        return self.result

    def coordinate_extraction(self, sample_names, sample_methods, sample_positions, sample_dilutions, df):
        
        if "f" in sample_methods:
            fprocessor = FluorescenceProcessor(df, sample_methods, sample_names, sample_dilutions)
            values_fluorescence = fprocessor._extract_setup()
            for sample in self.result:
                try:
                    self.result[sample].update(values_fluorescence[sample])  # update the result dict with the fluorescence values
                except:
                    print(f"⚠️ Sample '{sample}' not found in the fluorescence data. Skipping merge.")
                    

        if "b" or "r" in sample_methods:
            Extract = AbsorptionCoordinator(df, sample_methods, sample_names, sample_positions, self.Version, sample_dilutions)
            values_absorption = Extract._Version_Coordination()
            for sample in self.result:
                try:
                    self.result[sample].update(values_absorption[sample]) # update the result dict with the absorption values
                except:
                    print(f"⚠️ Sample '{sample}' not found in the absorption data. Skipping merge.")



    def _read_file(self):
        # Open the file
        try:
            df = pd.read_excel(self.file_path, header=None)
        except:
            print(f"❌ Failed to read the file: {self.file_path}. Please check if the file exists and is a valid Excel file.")
            sys.exit()
        return df

    # Read names from file with LS's Name_Reader
    def _read_names(self, df):
        try:
            name_reader = ReadNames(df, self.file_path)
            name_reader.read_setup()  
        except Exception as e:
            print(f"❌ Failed to read names from the file: {self.file_path}. Please check the file format: {e}")
            sys.exit()
        sample_names = name_reader.samples
        sample_methods = name_reader.methods
        sample_positions = name_reader.positions
        sample_dilutions = name_reader.dilutions
        return sample_names, sample_methods, sample_positions, sample_dilutions


    
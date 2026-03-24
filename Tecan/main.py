import os
import sys
from pathlib import Path
from ReadAndWrite.Write import WriteResult
from ReadAndWrite.Sort import SortResult
from Coordinator import Coordinator

"""
-------------------This is the main script--------------------
This script is designed to extract data from multiple Excel files in a specified folder 
and save the results into a single output file.

In the following part the script will cycle through all the files in the reading-folder, 
will give the file to the Coordinator (another script) to evaluate the data,
and then saves the data given by the Coordinator in the output file.
"""

current_dir = Path(__file__).parent

"""---Personalize the paths below---"""
# Extract the data from the Excel files in this folder
input_folder = current_dir / "Evaluation Folder" / "Input"
#input_folder = r'C:\Users\m4ng0\OneDrive\Desktop\Morph\K\DOE\Tecan\Evaluate'  # Folder containing your Excel files
# Collected data will be saved in this folder
output_file = current_dir / "Evaluation Folder" / "Output" / "Data collection.xlsx"
#output_file = r'C:\Users\m4ng0\OneDrive\Desktop\Morph\Data collection.xlsx'  # Path to save the output file



# Version of the script
Version = "LS"
"""
The script needs to know what the samples are, what the blanks are and what samples should be evaluated in what way.
The names are written in the Excel files itself. In what style they are written in the Excel file can be found in the README.md file.
Furthermore, the script can evaluate the data further based on the Version selected.

Known Versions:
LS: LS's Rheomorph project
Raw: Raw data. Takes the average and standard deviation of the samples, considers the dilution, but no further evaluation, same writing style as LS
"""




class setup:
    # This is needed to start the script and initialises the lists in which all the data is collected
    def __init__(self, input_folder, output_file, Version):
        self.input_folder = input_folder
        self.output_path = output_file
        self.Version = Version
        self.all_results = []
    
    # This function will try to run the Coordinator on all files in the input folder
    def run(self):
        files = self._find_files()
        for file_name in files:
            full_path = os.path.join(self.input_folder, file_name)
            print(f"\n📂 Processing: {file_name}")
            try:
                processor = Coordinator(full_path, self.Version)  # Create an instance of the Coordinator class
                self.all_results = processor.get_result()
                #print(f"✅ Processed {file_name} successfully: {self.all_results} samples extracted.")
            except Exception as e:
                print(f"❌ Failed to process {file_name}: {e}")
                sys.exit()
            

            # Collect results
            try:
                self._write_output()  # Write the results to the output file
            except Exception as e:
                print(f"❌ Failed to write output: {e}")
                sys.exit()
            
            # Sort the output file
            try:
                self._sort_output()  # Sort the output file 
            except Exception as e:
                print(f"❌ Failed to sort output: {e}")
                sys.exit()

        
    # This function finds all the files in the input folder that end with .xlsx
    def _find_files(self):
        return [f for f in os.listdir(self.input_folder) if f.endswith(".xlsx")]


    # This function saves the data collected in the all_results and all_metadata lists to the output file
    def _write_output(self):
        writer = WriteResult(self.output_path)
        writer.write(self.all_results)  # Write the results to the output file
    
    # This function sorts the output file after writing it
    def _sort_output(self):
        sorter = SortResult(self.output_path)
        sorter.sort_file()

    

# This is the main entry point of the script
if __name__ == "__main__":
    app = setup(input_folder, output_file, Version) # Calling the setup class (the part above)
    app.run() # Running the setup class


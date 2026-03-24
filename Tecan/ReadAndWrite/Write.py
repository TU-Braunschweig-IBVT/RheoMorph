import pandas as pd
from pathlib import Path


class WriteResult:

    def __init__(self, output_file_path):
        self.output_file_path = Path(output_file_path)
        self.columns = ["Sample", "fluorescence_avg", "fluorescence_std", "blue_avg", "blue_std", "red_avg", "red_std"]

    def write(self, result_dict):
        """
        Main interface for writing or updating the Excel file.
        """
        if not self.output_file_path.exists():
            print(f"📄 Output file not found. Creating new file at {self.output_file_path}")
            self._create_new_file(result_dict)
        else:
            print(f"📄 Output file found. Updating existing file at {self.output_file_path}")
            self._update_existing_file(result_dict)

    def _create_new_file(self, result_dict):
        """
        Creates a new Excel file from scratch with the given results if the file does not exist.
        """
        data_rows = []
        for name, values in sorted(result_dict.items(), key=lambda x: x[0]):
            row = [name] + [values.get(col, None) for col in self.columns[1:]]
            data_rows.append(row)

        df = pd.DataFrame(data_rows, columns=self.columns)
        df.to_excel(self.output_file_path, index=False)
        print("✅ File created and data written.")

    def _update_existing_file(self, result_dict):
        """
        Updates an existing Excel file by adding or updating rows.
        """
        df = pd.read_excel(self.output_file_path)

        # Ensure all expected columns are present
        for col in self.columns:
            if col not in df.columns:
                df[col] = None

        df.set_index("Sample", inplace=True)
        df = df.astype(object)


        already_present = []
        newly_added = []

        for name, values in sorted(result_dict.items(), key=lambda x: x[0]):
            if name in df.index:
                already_present.append(name)
            else:
                newly_added.append(name)
                df.loc[name] = [None] * (len(self.columns) - 1)

            for col, val in values.items():
                if col in self.columns:
                    df.at[name, col] = val

        df.reset_index(inplace=True)  
        df = df.sort_values(by="Sample") # Basic alphabetical sort
        df.to_excel(self.output_file_path, index=False)
        print("✅ Existing file updated with new data.")

        # 🔔 Notify user
        if already_present:
            print(f"ℹ️ These samples already existed and were updated: {', '.join(map(str, already_present))}")
        if newly_added:
            print(f"🆕 These new samples were added: {', '.join(map(str, newly_added))}")

    def final_result_dict(self, names):
        """
        Initializes a result dictionary from sample names with default values (None).
        """
        excluded_names = {"Blank", "BlankB", "BlankR"}
        unique_names = sorted({name for name in names if name not in excluded_names})

        # Exclude "Sample" from the columns list
        data_columns = [col for col in self.columns if col != "Sample"]

        # Build the result structure
        full_data = {
            name: {col: None for col in data_columns}
            for name in unique_names
        }

        return full_data
    
    

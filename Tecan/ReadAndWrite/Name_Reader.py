import pandas as pd
import re

class ReadNames:
    
    def __init__(self, dataframe, filename):
        self.filename = filename
        self.df = dataframe
        self.positions = []

    def read_setup(self):
        first_ex_row = None
        second_ex_row = None
        third_ex_row = None
        plate_area_row = None

        # Step 1: Locate "Ex" and "Plate area". 
        # The "Ex" is where one should have the sample names and methods
        # The "Plate area" is where the used wells are and is part of the regular documentation in the Excel file.
        try:
            for idx, row in self.df.iterrows():
                if 'Ex' in row.values.tolist():
                    if first_ex_row is None:
                        first_ex_row = idx
                    elif first_ex_row is not None and second_ex_row is None:
                        second_ex_row = idx
                    elif first_ex_row is not None and second_ex_row is not None and third_ex_row is None:
                        third_ex_row = idx
                if 'Plate area' in row.values.tolist():
                    plate_area_row = idx

            if first_ex_row is None or second_ex_row is None or third_ex_row is None:
                raise ValueError("Could not find three rows containing 'Ex'")
            if plate_area_row is None:
                raise ValueError("Could not find row containing 'Plate area'")
        except Exception as e:
            print(f"❌ Failed to locate labels in {self.filename}. Reason: {e}")
            return

        # Step 2: Get sample names -> putting the names after the first "Ex" in a list
        try:
            first_ex_col = self.df.loc[first_ex_row].tolist().index('Ex')
            self.samples = []
            for cell in self.df.loc[first_ex_row, first_ex_col + 1:]:
                if pd.isna(cell):
                    break
                self.samples.append(str(cell))
        except Exception as e:
            print(f"❌ Failed to read sample names in {self.filename}. Reason: {e}")
            return

        # Step 3: Get method codes -> putting the methoods after the second "Ex" in a list
        try:
            second_ex_col = self.df.loc[second_ex_row].tolist().index('Ex')
            self.methods = []
            for cell in self.df.loc[second_ex_row, second_ex_col + 1:]:
                if pd.isna(cell):
                    break
                self.methods.append(str(cell))
        except Exception as e:
            print(f"❌ Failed to read method codes in {self.filename}. Reason: {e}")
            return

        # Step 4: Get dilutions -> putting the dilutions after the third "Ex" in a list
        try:
            third_ex_col = self.df.loc[third_ex_row].tolist().index('Ex')
            self.dilutions = []
            for cell in self.df.loc[third_ex_row, third_ex_col + 1:]:
                if pd.isna(cell):
                    break
                self.dilutions.append(str(cell))
        except Exception as e:
            print(f"❌ Failed to read method codes in {self.filename}. Reason: {e}")
            return
        
        # Step 5: Get position string -> finding and saving the string after "Plate area"
        try:
            row_values = self.df.loc[plate_area_row].tolist()
            plate_area_col = row_values.index('Plate area')
            pos_string = None
            for val in row_values[plate_area_col + 1:]:
                if pd.notna(val):
                    pos_string = str(val)
                    break
            if pos_string is None:
                raise ValueError("No non-empty position string found after 'Plate area'")
        except Exception as e:
            print(f"❌ Failed to read plate area string in {self.filename}. Reason: {e}")
            return

        # Step 6: Parse position string -> converting the string into a list of positions (e.g. from 'A1-A12;B1-B11' to [['A', 1], ..., ['B', 11]])
        try:
            print(f"Parsing plate positions from string: {pos_string}")
            self.positions = self._parse_position_string(pos_string)
        except Exception as e:
            print(f"❌ Failed to parse plate positions in {self.filename}. Reason: {e}")
            return

        # Step 7: Check length consistency -> ensuring that the number of samples, methods, and positions match
        try:
            n_samples = len(self.samples)
            n_methods = len(self.methods)
            n_positions = len(self.positions)
            n_dilutions = len(self.dilutions)

            if n_samples != n_methods or n_samples != n_positions or n_samples != n_dilutions:
                print("❗ Sample / method / position count mismatch!")
                print(f"Samples ({n_samples}): {self.samples}")
                print(f"Methods ({n_methods}): {self.methods}")
                print(f"Positions ({n_positions}): {self.positions}")
                print(f"Dilutions ({n_dilutions}): {self.dilutions}")
                print(f"→ Please check your input in {self.filename}")
            else:
                print(f"✅ Sample info loaded successfully from {self.filename}.")
        except Exception as e:
            print(f"❌ Failed during consistency check in {self.filename}. Reason: {e}")

    # This function parses a position string like 'A1-A12;B1-B11' into a list of positions that will look like [['A', 1], ['A', 2], ..., ['B', 11]]
    def _parse_position_string(self, pos_string):
        positions = []
        ranges = pos_string.split(';')
        for r in ranges:
            r = r.strip()

            # Check for a range like A1-A12 or B3-B9
            range_match = re.match(r'^([A-Z])(\d+)-([A-Z])(\d+)$', r)
            if range_match:
                start_row, start_col, end_row, end_col = range_match.groups()
                start_row, end_row = ord(start_row), ord(end_row)
                start_col, end_col = int(start_col), int(end_col)

                for row in range(start_row, end_row + 1):
                    for col in range(start_col, end_col + 1):
                        positions.append([chr(row), col])
                continue  # Skip to next entry

            # Check for a single position like G1
            single_match = re.match(r'^([A-Z])(\d+)$', r)
            if single_match:
                row, col = single_match.groups()
                positions.append([row, int(col)])
                continue

            # If neither match, warn the user
            print(f"⚠️ Invalid format: '{r}' → Skipping.")
        return positions



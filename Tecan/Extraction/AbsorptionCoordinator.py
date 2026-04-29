from Extraction.Reader.BlockReading import BlockReader
from Extraction.Transformation.TransformationBlue import TransformationBlue
from Extraction.Transformation.TransformationRed import TransformationRed
from Extraction.Transformation.avg_std import compute_averages_and_deviations
import sys
import traceback

class AbsorptionCoordinator:
    def __init__(self, dataframe, sample_methods, sample_names, sample_positions, Version, sample_dilutions, return_individual):
        self.Version = Version
        self.df = dataframe
        self.sample_dilutions = sample_dilutions
        self.sample_methods = sample_methods
        self.sample_names = sample_names
        self.sample_positions = sample_positions
        self.all_sample_data = {}
        self.return_individual = return_individual

    def _Version_Coordination(self):
        try:
            if self.Version == "LS":
                print("🔹 Starting absorption extraction with LS version...")
                return self.LS()
            elif self.Version == "Raw":
                print("🔹 Starting absorption extraction with Raw version...")
                return self.Raw()
            else:
                print(f"Unknown Version: {self.Version}. Please use 'LS' or 'Raw'. This can be set in the main.py file.")
                sys.exit()
        except Exception as e:
            print(f"❌ Error during absorption extraction with version '{self.Version}': {e}")
            traceback.print_exc()
            sys.exit()


    # Apply dilution to replicate values
    def apply_dilution_replicates(self, name, values, dilution_map):
        dilutions = dilution_map.get(name, [1.0] * len(values))
        if len(dilutions) != len(values):
            print(f"⚠️ Mismatch for '{name}': {len(values)} values, {len(dilutions)} dilutions. Skipping dilution.")
            return values

        return [v * d for v, d in zip(values, dilutions)]

    
    def get_dilution_map(self, method):
        """
        Creates a dict mapping sample names to a list of dilution factors
        filtered by method ('b' or 'r'), preserving order.
        """
        dilution_map = {}
        for name, m, d in zip(self.sample_names, self.sample_methods, self.sample_dilutions):
            if m == method:
                dilution_map.setdefault(name, []).append(float(d) if d not in [None, ""] else 1.0)
        return dilution_map



    # This function is used to extract the data from the file with the LS's method
    def LS(self):
        try:
            marker_indices = self.df[self.df.iloc[:, 0] == "<>"].index.tolist()
            if len(marker_indices) < 4:
                raise ValueError(f"❌ Expected at least 4 '<>' markers, found {len(marker_indices)}.")
        except Exception as e:
            print(f"❌ Error finding '<>' markers: {e}")
            sys.exit()

        blankblue = "BlankB"
        methodBlue = "b"
        blankred = "BlankR"
        methodRed = "r"

        excluded_names = {"Blank", "BlankB", "BlankR"}
        unique_names = sorted({name for name in self.sample_names if name not in excluded_names})
        full_data = {name: {"blue_avg": None, "blue_std": None, "red_avg": None, "red_std": None} for name in unique_names}

        # --- BLUE ---
        blue_dilution_map = self.get_dilution_map("b")
        blue_data = {}
        for i in range(0, 2):
            try:
                reader = BlockReader(
                    self.df, marker_indices[i],
                    self.sample_methods,
                    self.sample_names,
                    self.sample_positions,
                    blank=blankblue,
                    method=methodBlue
                )
                block_data = reader.process_block()
                for name, values in block_data.items():
                    values = self.apply_dilution_replicates(name, values, blue_dilution_map)
                    blue_data.setdefault(name, []).extend(values)
            except Exception as e:
                print(f"❌ Error in blue block {i+1} at marker index {marker_indices[i]}: {e}")

        # ⬇️ Apply transformation to blue data
        for name in unique_names:
            if name in blue_data:
                raw_values = blue_data[name]

                if self.return_individual:
                    for idx, val in enumerate(raw_values, start=1):
                        new_name = f"{name}_{idx}"
                        if new_name not in full_data:
                            full_data[new_name] = {"blue_avg": None, "blue_std": None, "red_avg": None, "red_std": None}
                        full_data[new_name]["blue_avg"] = val
                        full_data[new_name]["blue_std"] = 0
                else:
                    transformerBlue = TransformationBlue(raw_values)
                    avg, std = transformerBlue.transformation()
                    full_data[name]["blue_avg"] = avg
                    full_data[name]["blue_std"] = std


        # --- RED ---
        red_data = {}
        red_dilution_map = self.get_dilution_map("r")
        for i in range(2, 4):
            try:
                reader = BlockReader(
                    self.df, marker_indices[i],
                    self.sample_methods,
                    self.sample_names,
                    self.sample_positions,
                    blank=blankred,
                    method=methodRed
                )
                block_data = reader.process_block()
                for name, values in block_data.items():
                    values = self.apply_dilution_replicates(name, values, red_dilution_map)
                    red_data.setdefault(name, []).extend(values)
            except Exception as e:
                print(f"❌ Error in red block {i-1} at marker index {marker_indices[i]}: {e}")

        for name in unique_names:
            if name in red_data:
                raw_values = red_data[name]

                if self.return_individual:
                    for idx, val in enumerate(raw_values, start=1):
                        new_name = f"{name}_{idx}"
                        if new_name not in full_data:
                            full_data[new_name] = {"blue_avg": None, "blue_std": None, "red_avg": None, "red_std": None}
                        full_data[new_name]["red_avg"] = val
                        full_data[new_name]["red_std"] = 0
                else:
                    transformerRed = TransformationRed(raw_values)
                    avg, std = transformerRed.transformation()
                    full_data[name]["red_avg"] = avg
                    full_data[name]["red_std"] = std

        return full_data

    # This function is used to extract the data from the file with the Raw's method
    def Raw(self):
            try:
                marker_indices = self.df[self.df.iloc[:, 0] == "<>"].index.tolist()
                if len(marker_indices) < 4:
                    raise ValueError(f"❌ Expected at least 4 '<>' markers, found {len(marker_indices)}.")
            except Exception as e:
                print(f"❌ Error finding '<>' markers: {e}")
                sys.exit()

            blankblue = "BlankB"
            methodBlue = "b"
            blankred = "BlankR"
            methodRed = "r"

            excluded_names = {"Blank", "BlankB", "BlankR"}
            unique_names = sorted({name for name in self.sample_names if name not in excluded_names})
            full_data = {name: {"blue_values": None, "red_values": None} for name in unique_names}
    
            # --- BLUE ---
            blue_dilution_map = self.get_dilution_map("b")
            blue_data = {}
            for i in range(0, 2):
                try:
                    reader = BlockReader(
                        self.df, marker_indices[i],
                        self.sample_methods,
                        self.sample_names,
                        self.sample_positions,
                        blank=blankblue,
                        method=methodBlue
                    )
                    block_data = reader.process_block()
                    for name, values in block_data.items():
                        values = self.apply_dilution_replicates(name, values, blue_dilution_map)
                        blue_data.setdefault(name, []).extend(values)
                except Exception as e:
                    print(f"❌ Error in blue block {i+1} at marker index {marker_indices[i]}: {e}")

            # ⬇️ Apply transformation to blue data
            for name in unique_names:
                if name in red_data:
                    if self.return_individual:
                        for idx, val in enumerate(blue_data[name], start=1):
                            new_name = f"{name}_{idx}"
                            if new_name not in full_data:
                                full_data[new_name] = {"blue_avg": None, "blue_std": None, "red_avg": None, "red_std": None}
                            full_data[new_name]["blue_avg"] = val
                            full_data[new_name]["blue_std"] = 0
                    else:
                        avg, std = compute_averages_and_deviations(blue_data[name])
                        full_data[name]["blue_avg"] = avg
                        full_data[name]["blue_std"] = std

            # --- RED ---
            red_dilution_map = self.get_dilution_map("r")
            red_data = {}
            for i in range(2, 4):
                try:
                    reader = BlockReader(
                        self.df, marker_indices[i],
                        self.sample_methods,
                        self.sample_names,
                        self.sample_positions,
                        blank=blankred,
                        method=methodRed
                    )
                    block_data = reader.process_block()
                    for name, values in block_data.items():
                        values = self.apply_dilution_replicates(name, values, red_dilution_map)
                        red_data.setdefault(name, []).extend(values)
                except Exception as e:
                    print(f"❌ Error in red block {i-1} at marker index {marker_indices[i]}: {e}")

            for name in unique_names:
                if name in red_data:
                    if self.return_individual:
                        for idx, val in enumerate(blue_data[name], start=1):
                            new_name = f"{name}_{idx}"
                            if new_name not in full_data:
                                full_data[new_name] = {"red_avg": None, "red_std": None, "blue_avg": None, "blue_std": None}
                            full_data[new_name]["red_avg"] = val
                            full_data[new_name]["red_std"] = 0
                    else:
                        avg, std = compute_averages_and_deviations(blue_data[name])
                        full_data[name]["red_avg"] = avg
                        full_data[name]["red_std"] = std

            return full_data

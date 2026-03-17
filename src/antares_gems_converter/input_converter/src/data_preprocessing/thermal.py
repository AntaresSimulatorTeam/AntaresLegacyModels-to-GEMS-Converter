import os
from pathlib import Path
from typing import Callable


import pandas as pd
from antares.craft.model.thermal import ThermalCluster

from gems.study.parsing import InputComponentParameter


class ThermalDataPreprocessing:
    DEFAULT_PERIOD: int = 168

    def __init__(self, thermal: ThermalCluster, study_path: Path, suffix: str = ".tsv"):
        self.thermal = thermal
        self.study_path = study_path
        self.suffix = suffix
        self.output_series_dir = self.study_path / "input" / "data-series"
        self._prepro_parameter_functions: dict[str, Callable[[int], pd.DataFrame]] = {
            "minimum_generation_modulation": lambda _: self.thermal.get_prepro_modulation_matrix().iloc[:, 3],
            "p_max_cluster": lambda _: self._compute_p_max_cluster(),
        }

    def _compute_p_max_cluster(self) -> pd.DataFrame:
        return self.thermal.get_series_matrix()

    def _build_csv_path_and_name(self, param_id: str) -> tuple[Path, str]:
        name = f"{self.thermal.area_id}_{self.thermal.id}_{param_id}"
        return self.output_series_dir / str(name + self.suffix), name

    def generate_component_parameter(
        self, parameter_id: str, period: int = 0
    ) -> InputComponentParameter:
        if parameter_id not in self._prepro_parameter_functions:
            raise ValueError(f"Unsupported parameter_id: {parameter_id}")

        df = self._prepro_parameter_functions[parameter_id](period)
        csv_path, value_name = self._build_csv_path_and_name(parameter_id)

        output_dir = os.path.dirname(csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # This separator is chosen to comply with the antares_craft timeseries creation
        df.to_csv(csv_path, sep="\t", index=False, header=False)

        return InputComponentParameter(
            id=parameter_id,
            time_dependent=True,
            scenario_dependent=True,
            value=value_name,
        )

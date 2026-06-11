import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from antares.craft import *

from antares_runner.antares_runner import AntaresHybridStudyBenchmarker
from antares_gems_converter.input_converter.src.converter import AntaresStudyConverter
from antares_gems_converter.input_converter.src.data_preprocessing.data_classes import (
    ConversionMode,
)
from antares_gems_converter.input_converter.src.logger import Logger

ANTARES_VERSION_CREATED_STUDIES = "9.3"
ANTARES_LEGACY_MODELS_PATH = [
    Path("src/antares_gems_converter/libs/antares_historic/antares_legacy_models.yml")
]
ACCURATE_TEMPLATE_PATH = Path(
    "tests/antares_historic/antares-resources/hybrid_mode_addon/uc_accurate"
)


def convert_study(
    study_dir: Path, study_name: str, model_list: list[str]
) -> tuple[Path, Path]:
    """Take the study study_dir / study_name and generate a hybrid study where all the components of the listed models are converted in GEMS format, according library ANTARES_LEGACY_MODELS_PATH."""
    log_path = ""
    logger: logging.Logger = Logger(__name__, log_path)
    study_path = study_dir / study_name
    converter_output_folder = study_dir.parent / "antares-studies-converted/"
    params = {
        "study_input": study_path,
        "logger": logger,
        "mode": ConversionMode.HYBRID.value,
        "output_folder": converter_output_folder,
        "lib_paths": ANTARES_LEGACY_MODELS_PATH,
        "models_to_convert": model_list,
    }
    converter = AntaresStudyConverter(**params)  # type: ignore
    converter.process_all()
    return study_path, converter.output_folder


def first_optim_relgap(
    exec_folder: Path,
    study_path_1: Path,
    study_path_2: Path,
    solver: Optional[str] = None,
) -> float:
    benchmarker = AntaresHybridStudyBenchmarker(
        exec_folder, study_path_1, study_path_2, solver
    )
    benchmarker.run()
    rel_gaps = benchmarker.weekly_rel_gaps()
    return rel_gaps[0].max()


def addHybridBehavior(study_path: Path) -> None:
    """Function to add some files to a Legacy Antares Study, so as to generate an hybrid behaviour (generation of the simulation table) with no impact on the simulation."""
    shutil.copytree(
        ACCURATE_TEMPLATE_PATH / "input", study_path / "input", dirs_exist_ok=True
    )
    shutil.copy2(
        ACCURATE_TEMPLATE_PATH / "generaldata.ini",
        study_path / "settings" / "generaldata.ini",
    )


def _create_area_with_base_clusters(
    study,
    load_timeserie: pd.DataFrame,
    area_name: str = "unique",
    cluster1_capacity: int = 150,
    cluster1_cost: int = 10,
    cluster2_capacity: int = 200,
    cluster2_cost: int = 20,
):
    area = study.create_area(
        area_name=area_name,
        properties=AreaProperties(energy_cost_unsupplied=20000, energy_cost_spilled=1),
    )
    area.set_load(load_timeserie)
    cluster1 = area.create_thermal_cluster(
        "prod",
        ThermalClusterProperties(
            unit_count=2,
            nominal_capacity=cluster1_capacity,
            marginal_cost=cluster1_cost,
            market_bid_cost=cluster1_cost,
            group=ThermalClusterGroup.NUCLEAR,
        ),
    )
    cluster1.set_series(pd.DataFrame(data=cluster1_capacity * np.ones((8760, 1))))
    cluster2 = area.create_thermal_cluster(
        "prod2",
        ThermalClusterProperties(
            unit_count=1,
            nominal_capacity=cluster2_capacity,
            marginal_cost=cluster2_cost,
            market_bid_cost=cluster2_cost,
            group=ThermalClusterGroup.NUCLEAR,
        ),
    )
    cluster2.set_series(pd.DataFrame(data=cluster2_capacity * np.ones((8760, 1))))
    return area


def createThermalTestAntaresStudy(
    study_name: str,
    parent_dir_path: Path,
    load_time_serie_file: Path,
    marg_cluster_properties: ThermalClusterProperties,
    marg_cluster_data_frame: pd.DataFrame,
    modulation_data_frame: Optional[pd.DataFrame] = None,
) -> None:
    study = create_study_local(
        study_name=study_name,
        version=ANTARES_VERSION_CREATED_STUDIES,
        parent_directory=parent_dir_path,
    )
    load_timeserie = pd.read_csv(load_time_serie_file, header=None)
    area = _create_area_with_base_clusters(study, load_timeserie)
    cluster3 = area.create_thermal_cluster("prod3", marg_cluster_properties)
    cluster3.set_series(marg_cluster_data_frame)
    if modulation_data_frame is not None:
        cluster3.set_prepro_modulation(modulation_data_frame)
    addHybridBehavior(parent_dir_path / study_name)


def createLinkTestAntaresStudy(
    study_name: str,
    parent_dir_path: Path,
    load1_time_serie_file: Path,
    load2_time_serie_file: Path,
    direct_capacity: np.ndarray,
    indirect_capacity: np.ndarray,
    hurdle_cost_direct: Optional[np.ndarray] = None,
    hurdle_cost_indirect: Optional[np.ndarray] = None,
    loop_flow: Optional[np.ndarray] = None,
) -> None:
    study = create_study_local(
        study_name=study_name,
        version=ANTARES_VERSION_CREATED_STUDIES,
        parent_directory=parent_dir_path,
    )
    load_timeseries = [
        pd.read_csv(load1_time_serie_file, header=None),
        pd.read_csv(load2_time_serie_file, header=None),
    ]
    area_list = [
        _create_area_with_base_clusters(study, load_timeseries[i], area_name=area_name)
        for i, area_name in enumerate(["unique", "area2"])
    ]
    link = study.create_link(
        area_from=area_list[0].name,
        area_to=area_list[1].name,
    )
    link.set_capacity_direct(pd.DataFrame(direct_capacity))
    link.set_capacity_indirect(pd.DataFrame(indirect_capacity))
    if (
        hurdle_cost_direct is not None
        or hurdle_cost_indirect is not None
        or loop_flow is not None
    ):
        parameters = np.zeros((8760, 6))
        if hurdle_cost_direct is not None:
            parameters[:, 0] = hurdle_cost_direct.flatten()
        if hurdle_cost_indirect is not None:
            parameters[:, 1] = hurdle_cost_indirect.flatten()
        if loop_flow is not None:
            parameters[:, 3] = loop_flow.flatten()
        link.set_parameters(
            pd.DataFrame(parameters),
        )
    link.update_properties(LinkPropertiesUpdate(hurdles_cost=True, loop_flow=True))
    opt_upd = OptimizationParametersUpdate(include_hurdlecosts=True)
    settings_upd = StudySettingsUpdate(optimization_parameters=opt_upd)

    study.update_settings(settings_upd)
    addHybridBehavior(parent_dir_path / study_name)


STS_TIMESERIES_SETTER_MAP = {
    "cost_injection": "set_cost_injection",
    "cost_withdrawal": "set_cost_withdrawal",
    "cost_level": "set_cost_level",
    "pmax_injection": "set_pmax_injection",
    "pmax_withdrawal": "set_pmax_withdrawal",
    "lower_rule_curve": "set_lower_rule_curve",
    "upper_rule_curve": "set_upper_rule_curve",
    "storage_inflows": "set_storage_inflows",
    "cost_variation_injection": "set_cost_variation_injection",
    "cost_variation_withdrawal": "set_cost_variation_withdrawal",
}


def createSTSTestAntaresStudy(
    study_name: str,
    parent_dir_path: Path,
    load_time_serie_file: Path,
    sts_properties: STStorageProperties,
    sts_timeseries: Optional[dict[str, pd.DataFrame]] = None,
) -> None:
    study = create_study_local(
        study_name=study_name,
        version=ANTARES_VERSION_CREATED_STUDIES,
        parent_directory=parent_dir_path,
    )
    load_timeserie = pd.read_csv(load_time_serie_file, header=None)
    area = _create_area_with_base_clusters(
        study,
        load_timeserie,
        cluster1_capacity=200,
        cluster2_capacity=400,
        cluster2_cost=100,
    )
    area.hydro.update_properties(
        HydroPropertiesUpdate(
            overflow_spilled_cost_difference=-1,
        )
    )
    cluster3 = area.create_st_storage("sts", sts_properties)
    if sts_timeseries:
        for key, df in sts_timeseries.items():
            getattr(cluster3, STS_TIMESERIES_SETTER_MAP[key])(df)
    addHybridBehavior(parent_dir_path / study_name)


def createMiscGenTestAntaresStudy(
    study_name: str,
    parent_dir_path: Path,
    load_time_serie_file: Path,
    misc_gen: pd.DataFrame,
) -> None:
    study = create_study_local(
        study_name=study_name,
        version=ANTARES_VERSION_CREATED_STUDIES,
        parent_directory=parent_dir_path,
    )
    load_timeserie = pd.read_csv(load_time_serie_file, header=None)
    area = _create_area_with_base_clusters(study, load_timeserie)
    area.set_misc_gen(misc_gen)
    addHybridBehavior(parent_dir_path / study_name)


def createHydroTestAntaresStudy(
    study_name: str,
    parent_dir_path: Path,
    load_time_serie_file: Path,
    ror_timeseries: Optional[pd.DataFrame] = None,
    maxpower: Optional[pd.DataFrame] = None,
    minimum_generation: Optional[pd.DataFrame] = None,
    inflows: Optional[pd.DataFrame] = None,
    hydro_properties: Optional[HydroPropertiesUpdate] = None,
) -> None:
    study = create_study_local(
        study_name=study_name,
        version=ANTARES_VERSION_CREATED_STUDIES,
        parent_directory=parent_dir_path,
    )
    load_timeserie = pd.read_csv(load_time_serie_file, header=None)
    area = _create_area_with_base_clusters(study, load_timeserie)

    if ror_timeseries is not None:
        area.hydro.set_ror_series(ror_timeseries)

    if maxpower is None:
        maxpower = pd.DataFrame(np.zeros((365, 4)))
        maxpower.loc[:, 0] = 100
        maxpower.loc[:, 1] = 24
        maxpower.loc[:, 2] = 100
        maxpower.loc[:, 3] = 24
    area.hydro.set_maxpower(maxpower)

    if minimum_generation is not None:
        area.hydro.set_mingen(minimum_generation)

    reservoir = pd.DataFrame(np.ones((365, 3)) * 0.5)
    area.hydro.set_reservoir(reservoir)

    if inflows is not None:
        area.hydro.set_mod_series(inflows)

    area.hydro.update_properties(
        HydroPropertiesUpdate(
            reservoir=True,
            reservoir_capacity=5000,
            pumping_efficiency=0.75,
            overflow_spilled_cost_difference=0,
        )
    )
    if hydro_properties is not None:
        area.hydro.update_properties(hydro_properties)

    addHybridBehavior(parent_dir_path / study_name)


def random_availability_ratio(seed: int = 1000) -> np.ndarray:
    np.random.seed(seed)  # for reproducibility
    raw = np.random.random((8760, 1))
    return 0.2 + 0.8 * raw

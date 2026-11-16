from pathlib import Path
from time import time

import pytest
from antares.craft import STStorageProperties

from tests.antares_historic.utils import (
    AreaTestSetup,
    convert_study,
    createAreaConnectionTestAntaresStudy,
    first_optim_relgap,
)

## TESTING PROCEDURE FOR THE AREA-CONNECTION OF THE "flow" PORT-TYPE (HYBRID MODE) : CONSTANT DATA ##

# In hybrid mode, the port fields of a converted component replace its contribution to three
# constraints of the legacy area: the balance (injection-to-balance), the bound on spillage, also
# called "fictitious load" (spillage-bound), and the bound on unsupplied energy
# (unsupplied-energy-bound). Legacy Antares bounds the spilled energy by the dispatchable thermal,
# hydro and short-term storage generation plus the must-run generation, which forbids an area to
# import energy only to spill it. It bounds the unsupplied energy by the load.

AREA_CONNECTION_TEST_REL_ACCURACY = 1e-6
HYDRO_TEST_REL_ACCURACY = 5e-6  # Antares adds noise to the hydro generation and pumping costs, which we can't reproduce
AREA_CONNECTION_TEST_SOLVER = "highs"
LINK_CAPACITY = 1000
STS_PROPERTIES = STStorageProperties(
    group="battery",
    injection_nominal_capacity=100,
    withdrawal_nominal_capacity=100,
    reservoir_capacity=300,
    efficiency=1,
    efficiency_withdrawal=1,
    initial_level=0.5,
    initial_level_optim=False,
)
# Area with a misc gen surplus (800 MW for a 300 MW load) that is expensive to spill
AREA_WITH_SURPLUS = AreaTestSetup(
    name="unique", load=300, misc_gen=800, energy_cost_spilled=100
)

## TESTING PROCEDURE FOR THE AREA-CONNECTION OF THE "flow" PORT-TYPE (HYBRID MODE) : TESTS ##

# unsupplied-energy-bound [OK : test_unsupplied_energy_with_converted_load]
# spillage-bound
## load [OK : test_imports_of_area_with_converted_load]
## link [OK : test_spillage_of_imports_with_converted_link]
## short-term storage [OK : test_spillage_of_imports_charging_converted_storage]
## hydro [OK : test_spillage_of_imports_pumped_by_converted_hydro]
## generation [OK : test_spillage_of_converted_generation]


def area_connection_test_procedure(
    study_path: Path,
    areas: list[AreaTestSetup],
    models: list[str],
    exec_folder: Path,
    link_capacity: float | None = None,
    rel_accuracy: float = AREA_CONNECTION_TEST_REL_ACCURACY,
) -> None:
    study_name = f"area_connection_test_study_{str(int(100*time()))}"
    createAreaConnectionTestAntaresStudy(study_name, study_path, areas, link_capacity)
    original_study_path, converted_study_path = convert_study(
        study_path, study_name, models
    )
    rel_gap = first_optim_relgap(
        exec_folder,
        original_study_path,
        converted_study_path,
        AREA_CONNECTION_TEST_SOLVER,
    )
    assert rel_gap < rel_accuracy


@pytest.mark.parametrize("models", [["load"], ["load", "thermal"]])
def test_unsupplied_energy_with_converted_load(
    models: list[str], auto_generated_studies_path: Path, antares_exec_folder: Path
) -> None:
    # 600 MW of load for 350 MW of thermal generation: 250 MW of unsupplied energy every hour.
    # The converted load must still allow it, as the legacy load did.
    area_connection_test_procedure(
        auto_generated_studies_path,
        [AreaTestSetup(name="unique", load=600)],
        models,
        antares_exec_folder,
    )


# Thermal clusters are not converted here: the objective holds only thermal costs, so the noise
# that legacy Antares adds to them would exceed the test accuracy.
@pytest.mark.parametrize("models", [["load"], ["load", "link"]])
def test_imports_of_area_with_converted_load(
    models: list[str], auto_generated_studies_path: Path, antares_exec_folder: Path
) -> None:
    # area2 imports its load from the cheap generation of unique rather than generating it
    # locally at a high cost: a converted load must not change the spillage bound of its area.
    area_connection_test_procedure(
        auto_generated_studies_path,
        [
            AreaTestSetup(name="unique", load=300, cluster1_capacity=1000),
            AreaTestSetup(name="area2", load=300, cluster1_cost=100, cluster2_cost=200),
        ],
        models,
        antares_exec_folder,
        LINK_CAPACITY,
    )


@pytest.mark.parametrize("models", [["link"], ["link", "load", "misc_gen"]])
def test_spillage_of_imports_with_converted_link(
    models: list[str], auto_generated_studies_path: Path, antares_exec_folder: Path
) -> None:
    # Spilling is cheaper in area2 than in unique, but area2 cannot import more than its load
    # to spill it: unique exports 200 MW and spills the remaining 300 MW of its surplus.
    area_connection_test_procedure(
        auto_generated_studies_path,
        [AREA_WITH_SURPLUS, AreaTestSetup(name="area2", load=200)],
        models,
        antares_exec_folder,
        LINK_CAPACITY,
    )


@pytest.mark.parametrize(
    "models",
    [
        ["short-term-storage"],
        ["short-term-storage", "link", "load", "misc_gen", "thermal"],
    ],
)
def test_spillage_of_imports_charging_converted_storage(
    models: list[str], auto_generated_studies_path: Path, antares_exec_folder: Path
) -> None:
    # area2 may import beyond its load what its storage injects, and spill what its storage
    # withdraws: unique exports 300 MW, area2 cycles 100 MW through its storage and spills it.
    area_connection_test_procedure(
        auto_generated_studies_path,
        [
            AREA_WITH_SURPLUS,
            AreaTestSetup(name="area2", load=200, sts_properties=STS_PROPERTIES),
        ],
        models,
        antares_exec_folder,
        LINK_CAPACITY,
    )


def test_spillage_of_imports_pumped_by_converted_hydro(
    auto_generated_studies_path: Path, antares_exec_folder: Path
) -> None:
    # Same as the short-term storage case with a hydro reservoir: area2 may import beyond its
    # load what it pumps, and spill what it generates. The hydro of every area is converted, so
    # unique gets a reservoir without generation nor pumping. A high cost of spillage in unique
    # keeps the hydro noise of Antares below the test accuracy.
    area_connection_test_procedure(
        auto_generated_studies_path,
        [
            AreaTestSetup(
                name="unique",
                load=300,
                misc_gen=800,
                energy_cost_spilled=1000,
                hydro_max_power=0,
            ),
            AreaTestSetup(name="area2", load=200, hydro_max_power=100),
        ],
        ["hydro"],
        antares_exec_folder,
        LINK_CAPACITY,
        HYDRO_TEST_REL_ACCURACY,
    )


def test_spillage_of_converted_generation(
    auto_generated_studies_path: Path, antares_exec_folder: Path
) -> None:
    # The 500 MW misc gen surplus of a single area is spilled every hour.
    area_connection_test_procedure(
        auto_generated_studies_path,
        [AREA_WITH_SURPLUS],
        ["misc_gen"],
        antares_exec_folder,
    )

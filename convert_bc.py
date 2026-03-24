import os
import sys
import yaml
from pathlib import Path
from typing import List, Dict
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


from antares.craft.model.study import read_study_local, Study
from antares.craft.model.binding_constraint import BindingConstraintOperator, BindingConstraintFrequency, BindingConstraint, ConstraintTerm
from antares.craft.model.thermal import ThermalCluster
from antares.craft import ThermalClusterGroup

from gems.model.parsing import InputLibrary, InputPortType, InputField, InputModel, InputParameter, InputModelPort, InputConstraint, parse_yaml_library
from gems.study.parsing import InputComponent, InputComponentParameter, InputPortConnections, InputSystem, parse_yaml_components

from antares_gems_converter.input_converter.src.utils import dump_to_yaml


def convert_bc_nuc(study_path : Path, study_name : str) -> None:
    study = read_study_local(study_path / study_name)

    bc_nuc = get_list_bc_nuc(study)

    clusters_nuc = get_list_cluster_nuc(study)

    with (study_path / study_name / "input" / "model-libraries" / "antares_legacy_models.yml").open() as c:
        lib = yaml.safe_load(c)
    
    # bc_port = initial_lib.port_types[0]
    # lib = yaml.load(study_path / study_name / "input" / "model-libraries" / "antares_legacy_models.yml")
    # print(lib)

    lib_id = lib["library"]["id"]


    bc_port = InputPortType(id="flow", fields=[InputField(id = "flow")])

    list_models :List[InputModel] = []
    list_components : List[InputComponent] = []
    list_connections : List[InputPortConnections] = []

    for id_cst, constraint in bc_nuc.items():
        print(f" Converting constraint {id_cst}")
        if constraint.properties.enabled:
        # print(constraint.properties.group) ### TODO
            model_id = f"bc_nuc_{constraint.properties.time_step.value}_{constraint.properties.operator.value}"

            component = get_bc_component(study_path, study_name, clusters_nuc, list_connections, id_cst, constraint, model_id,lib_id)
            list_components.append(component)

            if model_id not in [m.id for m in list_models]:
                model_bc = get_bc_model(clusters_nuc, bc_port, constraint, model_id)
                list_models.append(model_bc)
        
        study.delete_binding_constraint(constraint)

    # save_library(study_path, study_name, bc_port, list_models)
    lib["library"]["models"] += [model.model_dump(by_alias=True, exclude_unset=True) for model in list_models]
    with open(study_path / study_name / "input" / "model-libraries" / "antares_legacy_models.yml", "w", encoding="utf-8") as yaml_file:
        yaml.dump(
        lib,
        yaml_file,
        allow_unicode=True,
        sort_keys=False
    )

    save_system(study_path, study_name,list_components, list_connections)

    areas = study.get_areas()
    for cluster in clusters_nuc:
        print(f" Deleting cluster {cluster.id}")
        areas[cluster.area_id].delete_thermal_cluster(cluster)

def save_system(study_path : Path, study_name : str,list_components : List[InputComponent], list_connections : List[InputPortConnections]) -> None:
    with (study_path / study_name / "input" / "system.yml").open() as c:
        system = parse_yaml_components(c)
    if system.components is not None :
        system.components += list_components
    else :
        system.components = list_components
    if system.connections is not None :
        system.connections += list_connections
    else :
        system.connections = list_connections
    dump_to_yaml(system, study_path / study_name / "input" / "system.yml")

def save_library(study_path : Path, study_name : str, bc_port : InputPortType, list_models : List[InputModel]) -> None:
    lib = InputLibrary(id = "bc_nuc", port_types=[bc_port], models=list_models)

    with open(study_path / study_name / "input" / "model-libraries" / "lib_bc_nuc.yml", "w", encoding="utf-8") as yaml_file:
        yaml.dump(
        {
            "library": lib.model_dump(by_alias=True, exclude_unset=True),
        },
        yaml_file,
        allow_unicode=True,
        sort_keys=False
    )

def get_list_bc_nuc(study : Study) -> Dict[str, BindingConstraint]:
    bc_nuc = {n : cst for (n,cst) in study.get_binding_constraints().items() if "nuc" in n}
    return bc_nuc

def get_list_cluster_nuc(study : Study) -> List[ThermalCluster]:
    clusters_nuc : List[ThermalCluster] = []
    for area in study.get_areas().values():
        if area.name in ["fr","y_nuc_modulation"]:
            for cluster in area.get_thermals().values():
                if cluster.properties.group == ThermalClusterGroup.NUCLEAR and cluster.properties.enabled:
                    clusters_nuc.append(cluster)
    return clusters_nuc

def get_bc_model(clusters_nuc:List[ThermalCluster], bc_port:InputPortType, constraint:BindingConstraint, model_id:str) -> InputModel:
    left_terms = [f"alpha_{cluster.id} * (sum_connections(flow_{cluster.id}.flow))" for cluster in clusters_nuc]
    operator = get_operator(constraint)
    right_terms = ["rhs"]
    bc = build_constraint(constraint, left_terms, operator, right_terms)
    model_bc = InputModel(
                id=model_id,
                parameters=[InputParameter(id=f"alpha_{cluster.id}") for cluster in clusters_nuc] + [InputParameter(id="rhs", time_dependent=True, scenario_dependent=True)],
                ports=[
                    InputModelPort(type=bc_port.id, id=f"flow_{cluster.id}") for cluster in clusters_nuc
                ],
                binding_constraints=bc,
            )

    return model_bc

def build_constraint(constraint:BindingConstraint, left_terms:List[str], operator:str, right_terms:List[str])->List[InputConstraint]:
    if constraint.properties.time_step == BindingConstraintFrequency.HOURLY:
        bc=[
                    InputConstraint(
                        id="bc_hourly",
                        expression=" + ".join(left_terms)+operator+" + ".join(right_terms),
                    )
                ]
    elif constraint.properties.time_step == BindingConstraintFrequency.WEEKLY:
        bc = [
                    InputConstraint(
                        id="bc_weekly",
                        expression="sum(" + " + ".join(left_terms) + ")" + operator + "sum(" + " + ".join(right_terms) + ")",
                    )
                ]
    elif constraint.properties.time_step == BindingConstraintFrequency.DAILY:
        bc=[
                    InputConstraint(
                        id=f"bc_daily_{day}",
                        expression=" + ".join([t + f"[{hour}]" for t in left_terms for hour in range(24*day,24*(day+1))])+operator+" + ".join([t + f"[{hour}]" for t in right_terms for hour in range(24*day,24*(day+1))]),
                    )
                for day in range(7)]
    else:
        raise TypeError(f"Frequency {constraint.properties.time_step} not supported")
    return bc

def get_operator(constraint:BindingConstraint)->str:
    if constraint.properties.operator == BindingConstraintOperator.LESS:
        operator = " <= "
    elif constraint.properties.operator == BindingConstraintOperator.GREATER:
        operator = " >= "
    elif constraint.properties.operator == BindingConstraintOperator.EQUAL:
        operator = " = "
    else :
        raise TypeError(f"Operator {constraint.properties.operator} not supported")
    return operator

def get_bc_component(study_path:Path, study_name:str, clusters_nuc:List[ThermalCluster], list_connections:List[InputPortConnections], id_cst:str, constraint:BindingConstraint, model_id:str, lib_id:str)->InputComponent:
    constraint_terms = constraint.get_terms()
    parameters : List[InputComponentParameter] = []
    for cluster in clusters_nuc:
        list_connections.append(InputPortConnections(component1=cluster.area_id+"_"+cluster.id,
                                                         port1="balance_port",
                                                         component2=id_cst,
                                                         port2=f"flow_{cluster.id}"))
        alpha = get_alpha_value(constraint_terms, cluster)
        parameters.append(InputComponentParameter(id = f"alpha_{cluster.id}", value = alpha,
                                                  time_dependent=False,
                                              scenario_dependent=False))
        

    name = save_rhs(study_path, study_name, id_cst, constraint)

    parameters.append(InputComponentParameter(id = f"rhs",
                                              time_dependent=True,
                                              scenario_dependent=True,
                                              value = name))
        
    component = InputComponent(id = id_cst,
                                   model = lib_id+"."+model_id,
                                #    scenario_group="", # TODO 
                                   parameters=parameters)
                           
    return component

def save_rhs(study_path:Path, study_name:str, id_cst:str, constraint:BindingConstraint)->str:
    if constraint.properties.operator == BindingConstraintOperator.LESS:
        rhs = constraint.get_less_term_matrix()
    if constraint.properties.operator == BindingConstraintOperator.GREATER:
        rhs = constraint.get_greater_term_matrix()
    if constraint.properties.operator == BindingConstraintOperator.EQUAL:
        rhs = constraint.get_equal_term_matrix()

    if constraint.properties.time_step == BindingConstraintFrequency.HOURLY:
        rhs = np.array(rhs)[:8760]
    elif constraint.properties.time_step == BindingConstraintFrequency.WEEKLY or constraint.properties.time_step == BindingConstraintFrequency.DAILY:
        rhs = np.array(rhs)[:364]
        rhs_week = rhs.reshape(52, 7, 1).sum(axis=1)
        rhs_hourly = rhs_week / 168.0          # (52, 1)
        rhs = np.repeat(rhs_hourly, 168, axis=0)  # (8736, 1)

    name = f"{id_cst}_rhs"
    csv_path = study_path / study_name / "input" / "data-series" / str(name + ".tsv")

    output_dir = os.path.dirname(csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # This separator is chosen to comply with the antares_craft timeseries creation
    np.savetxt(csv_path, rhs, delimiter="\t")
    return name

def get_alpha_value(constraint_terms:Dict[str,ConstraintTerm], cluster:ThermalCluster)->float:
    if cluster.area_id+"."+cluster.id in constraint_terms.keys():
        term = constraint_terms[cluster.area_id+"."+cluster.id]
        assert term.offset == 0
        alpha = term.weight
    else :
        alpha = 0
    return alpha
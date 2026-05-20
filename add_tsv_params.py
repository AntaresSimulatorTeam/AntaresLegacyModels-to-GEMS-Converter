from pathlib import Path
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

# Paths
data_series_dir = r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\antares-studies-converted\BP25\input\data-series"
system_yml_path = r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\antares-studies-converted\BP25\input\system.yml"

# Possible NOM_TS
possible_nom_ts = [
    "ve_fr_max_mobilite_lourde_24h",
    "ve_fr_min_24h_charging_at_night",
    "ve_fr_min_24h_charging",
    "ve_fr_max_v2g_24h",
    "ve_eu_agregee_min_24h_charging_at_day",
    "stock_minimal",
    "stock_mobile"
]

def parse_filename(filename, area_names):
    """
    Parse the filename to extract NOM_TS, AREA, NOM_MODELE.
    Returns (NOM_TS, AREA, NOM_MODELE) or None if not matching.
    """
    if not filename.endswith('.tsv'):
        return None
    for nom_ts in possible_nom_ts:
        if filename.startswith(nom_ts + '_'):
            remaining = filename[len(nom_ts) + 1:-4]  # remove nom_ts + '_' and .tsv
            for area in sorted(area_names, key=len, reverse=True):
                prefix = area + '_'
                if remaining.startswith(prefix):
                    nom_modele = remaining[len(prefix):]
                    return nom_ts, area, nom_modele
    return None

def main():
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    # Load system.yml
    with open(system_yml_path, 'r', encoding='utf-8') as f:
        system_data = yaml.load(f)

    # Known areas and component ids from system.yml
    area_names = {
        entry.get('area')
        for entry in system_data.get('system', {}).get('area-connections', [])
        if entry.get('area')
    }
    area_to_component = {
        entry.get('area'): entry.get('component')
        for entry in system_data.get('system', {}).get('area-connections', [])
        if entry.get('area') and entry.get('component')
    }
    component_model = {
        comp.get('id'): comp.get('model')
        for comp in system_data.get('system', {}).get('components', [])
        if comp.get('id')
    }

    # Fallback area mapping from component ids when area-connections are missing
    if not area_names:
        for comp_id, model in component_model.items():
            if not model:
                continue
            model_name = model.split('.')[-1]
            prefix = f"{model_name}_"
            if comp_id.startswith(prefix):
                area = comp_id[len(prefix):]
                if area:
                    area_names.add(area)

    # Collect all .tsv files
    tsv_files = list(Path(data_series_dir).glob("*.tsv"))
    parsed_tsv = []
    for tsv_file in tsv_files:
        parsed = parse_filename(tsv_file.name, area_names)
        if parsed:
            parsed_tsv.append((tsv_file.name, parsed))

    # Process each parsed TSV
    added_values = set()
    for filename, (nom_ts, area, nom_modele) in parsed_tsv:
        component_id = area_to_component.get(area)
        if component_id is None:
            component_id = f"{nom_modele}_{area}"

        model_name = component_model.get(component_id, "")
        if model_name and not model_name.endswith(nom_modele):
            print(f"Warning: Component {component_id} model '{model_name}' does not match expected '{nom_modele}' for {filename}")
            continue

        value = f"{nom_ts}_{area}_{nom_modele}"
        component = next((comp for comp in system_data.get('system', {}).get('components', []) if comp.get('id') == component_id), None)

        if component is None:
            print(f"Warning: Component {component_id} not found for {filename}")
            continue

        # Check if parameter already exists
        parameters = component.get('parameters', [])
        param_exists = any(p.get('id') == nom_ts for p in parameters)

        if not param_exists:
            # Add the parameter
            new_param = CommentedMap()
            new_param['id'] = nom_ts
            new_param['scenario-dependent'] = False
            new_param['time-dependent'] = True
            new_param['value'] = value
            new_param.yaml_add_eol_comment('Added by TSV script', key='id')
            parameters.append(new_param)
            component['parameters'] = parameters
            added_values.add(value + '.tsv')
            print(f"Added parameter {nom_ts} to component {component_id}")
        else:
            print(f"Parameter {nom_ts} already exists in component {component_id}")

    # Add ve_dispo_charging parameter to electric_vehicle_fr component if it exists
    electric_vehicle_comp = next((comp for comp in system_data.get('system', {}).get('components', []) if comp.get('id') == 'electric_vehicle_fr'), None)
    if electric_vehicle_comp:
        parameters = electric_vehicle_comp.get('parameters', [])
        param_exists = any(p.get('id') == 've_dispo_charging' for p in parameters)
        if not param_exists:
            # Add the parameter
            new_param = CommentedMap()
            new_param['id'] = 've_dispo_charging'
            new_param['scenario-dependent'] = False
            new_param['time-dependent'] = True
            new_param['value'] = 've_dispo_charging_fr_electric_vehicle_fr'
            new_param.yaml_add_eol_comment('Added for electric_vehicle_fr', key='id')
            parameters.append(new_param)
            electric_vehicle_comp['parameters'] = parameters
            print(f"Added parameter ve_dispo_charging to component electric_vehicle_fr")
        else:
            print(f"Parameter ve_dispo_charging already exists in component electric_vehicle_fr")
    else:
        print("Component electric_vehicle_fr not found, skipping ve_dispo_charging parameter")

    # Save the modified system.yml
    with open(system_yml_path, 'w', encoding='utf-8') as f:
        yaml.dump(system_data, f)

    # Verification: Check that each TSV has a corresponding component
    for filename, (nom_ts, area, nom_modele) in parsed_tsv:
        component_id = area_to_component.get(area) or f"{nom_modele}_{area}"
        if not any(comp.get('id') == component_id for comp in system_data.get('system', {}).get('components', [])):
            print(f"Error: No component found for {filename} (expected {component_id})")
        else:
            print(f"Verified: {filename} has corresponding component {component_id}")

    # Additional verification: Check that all added values have corresponding TSV files
    for value_file in added_values:
        if not (Path(data_series_dir) / value_file).exists():
            print(f"Warning: Added value {value_file} does not have a corresponding TSV file")
        else:
            print(f"Verified: {value_file} exists")

    print(f"Script completed.")

if __name__ == "__main__":
    main()
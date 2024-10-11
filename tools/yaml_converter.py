"""YAML conversion module"""


from __future__ import annotations

import json
import os

import yaml


def toJson(yaml_config: str) -> dict:
    """
    Method converts YAML configuration to JSON object.
    Args:
        yaml_config   - YAML configuration file
    Returns:
        <dict>        - JSON string representation of the input YAML
    """
    content = ""

    # Open file content
    with open(yaml_config, "r") as file:
        content = file.read()

    # Filter tabs before loading yaml format
    content = content.replace("\t", "")

    return json.loads(
        json.dumps(yaml.load(content, Loader=yaml.FullLoader), sort_keys=True, indent=2)
    )  # , ensure_ascii=False))


def toYaml(json_content: dict, yaml_config: str) -> bool:
    """
    Method converts JSON object to YAML and write it to a file.
    Args:
        json_content    - JSON data that needs to be converted into YAML format
        yaml_config     - YAML configuration file name
    Returns:
        <bool>   - True if conversion was successful, False otherwise
    """
    if not os.path.exists(yaml_config):
        return False

    with open(yaml_config, "w") as f:
        yaml.dump(json_content, f)
    return True

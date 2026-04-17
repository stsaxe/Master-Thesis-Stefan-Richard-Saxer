import copy
from typing import List, override

import yaml

from ble.pseudomization.epoch_rotation_enum import EpochRotation
from ble.generation.GenerationStrategies import DefaultGenStratField, GEN_STRAT_REGISTRY
from ble.pseudomization.PseudomizerSupportInterface import PseudomizerSupportInterface
from ble.pseudomization.pseudomizer_config import PseudomizerConfig
from ble.walking.PathSegment import PathSegment
from ble.generation.GenerationRule import GenerationRule
from ble.walking.Path import Path
from ble.fields.AbstractField import AbstractField
from ble.interfaces.NodeInterface import NodeInterface
from ble.interfaces.RuntimeInterface import RuntimeInterface


class GenConfig(RuntimeInterface, PseudomizerSupportInterface):
    def __init__(self) -> None:
        self.current_path: Path = Path()
        self.field_gen_rules: List[GenerationRule] = []
        self.node_gen_rules: List[GenerationRule] = []
        self.ctx: NodeInterface = None
        self.global_pseudo_config: PseudomizerConfig = PseudomizerConfig()

    def set_ctx(self, ctx: NodeInterface) -> None:
        assert isinstance(ctx, NodeInterface), "ctx is not of type NodeInterface"
        self.ctx = ctx

    def from_yaml(self, path: str) -> None:
        with open(path, "r", encoding='utf-8') as f:
            file = yaml.safe_load(f)

        assert isinstance(file, dict), " loaded yaml is not a dictionary"

        self.from_dict(file)


    def from_dict(self, data: dict):
        assert "seed" in data.keys(), "seed not found in config"

        seed = data["seed"]
        self.global_pseudo_config.seed = seed
        self.global_pseudo_config.epoch = 0

        if "node rules" in data.keys():
            for rule in data["node rules"]:
                new_rule = GenerationRule()
                new_rule.from_dict(rule)
                self.node_gen_rules.append(new_rule)

        if "field rules" in data.keys():
            for rule in data["field rules"]:
                new_rule = GenerationRule()
                new_rule.from_dict(rule)
                self.field_gen_rules.append(new_rule)


        self.configure_pseudomizer(self.global_pseudo_config)


    @override
    def enter_node(self, node: NodeInterface) -> None:
        self.current_path.append(node.get_path_segment())

        number_of_rules_applied = 0

        for rule in self.node_gen_rules:
            if rule.get_path().matches(self.current_path):
                _ = rule.execute(None, node)
                rule.rotate_epoch(rotation_type="call")

                number_of_rules_applied += 1

        if number_of_rules_applied == 0:
            try:
                strat = GEN_STRAT_REGISTRY.create(self.current_path)

                strat.configure_pseudomizer(self.global_pseudo_config)

                _ = strat.execute(None, node)

                self.global_pseudo_config.epoch += 1

            except KeyError:
                pass



    @override
    def process_fields(self, fields: list[AbstractField]) -> None:
        for field in fields:
            number_of_rules_applied = 0

            for rule in self.field_gen_rules:

                field_path = copy.deepcopy(self.current_path)
                field_path.append(PathSegment(field.get_name()))

                if rule.get_path().matches(field_path):
                    _ = rule.execute(field, self.ctx)
                    rule.rotate_epoch(rotation_type="call")

                    number_of_rules_applied += 1

            if number_of_rules_applied == 0:
                strat = DefaultGenStratField()
                strat.configure_pseudomizer(self.global_pseudo_config)

                _ = strat.execute(field, self.ctx)
                self.global_pseudo_config.epoch += 1

    @override
    def leave_node(self) -> None:
        self.current_path.pop()

    @override
    def configure_pseudomizer(self, config: PseudomizerConfig) -> None:
        for rule in self.node_gen_rules:
            rule.configure_pseudomizer(config)
        for rule in self.field_gen_rules:
            rule.configure_pseudomizer(config)

        self.global_pseudo_config = config

    @override
    def rotate_epoch(self, rotation_type: EpochRotation | str):
        for rule in self.node_gen_rules:
            rule.rotate_epoch(rotation_type)
        for rule in self.field_gen_rules:
            rule.rotate_epoch(rotation_type)

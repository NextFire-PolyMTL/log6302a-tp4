from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable

from code_analysis import CFG


def is_definition(cfg: CFG, nid: int) -> bool:
    return any(
        [
            cfg.get_type(cfg.get_children(nid)[0]) == "ValueParameter",
            (
                cfg.get_type(cfg.get_children(cfg.get_children(nid)[0])[0])
                == "OptValueParameter"
            ),
            (
                cfg.get_type(cfg.get_children(nid)[0]) == "BinOP"
                and cfg.get_image(cfg.get_children(nid)[0]) == "="
            ),
            "MemberDeclaration" in cfg.get_type(cfg.get_parents(nid)[0]),
            cfg.get_type(cfg.get_parents(nid)[0]) == "Global",
        ]
    )


def yield_all_vars(cfg: CFG) -> Iterable[int]:
    for nid in cfg.get_node_ids():
        if cfg.get_type(nid) == "Variable":
            yield nid


def yield_all_defs(cfg: CFG) -> Iterable[int]:
    for nid in yield_all_vars(cfg):
        if is_definition(cfg, nid):
            yield nid


def yield_all_refs(cfg: CFG) -> Iterable[int]:
    for nid in yield_all_vars(cfg):
        if not is_definition(cfg, nid):
            yield nid


def get_key(cfg: CFG, nid: int) -> tuple[str, str]:
    return (cfg.get_var_scope(nid), cfg.get_var_id(nid))


class DataFlowAlgorithm(ABC):
    def __init__(self):
        self.cfg: CFG

        self.in_dict: dict[int, set[int]]
        self.out_dict: dict[int, set[int]]

        self.all_defs: dict[tuple[str, str], set[int]]
        self.gen_dict: dict[int, set[int]]
        self.kill_dict: dict[int, set[int]]

        self.visited: set[int]
        self.worklist: list[int]

    def build_defs(self) -> dict[tuple[str, str], set[int]]:
        all_defs = defaultdict[tuple[str, str], set[int]](set)
        for nid in yield_all_defs(self.cfg):
            key = get_key(self.cfg, nid)
            all_defs[key].add(nid)
        return all_defs

    @abstractmethod
    def build_gen(self) -> dict[int, set[int]]:
        ...

    def build_kill(self) -> dict[int, set[int]]:
        kill_dict = defaultdict[int, set[int]](set)
        for nid, var_nids in self.gen_dict.items():
            for var_nid in var_nids:
                key = get_key(self.cfg, var_nid)
                kill_dict[nid] |= self.all_defs[key] - var_nids
        return kill_dict

    @abstractmethod
    def pre_loop_init(self) -> Iterable[None]:
        ...

    @abstractmethod
    def apply_flow_eq(self, nid: int) -> None:
        ...

    @abstractmethod
    def next_nodes(self, nid: int) -> Iterable[int]:
        ...

    @abstractmethod
    def can_propagate(self, nid: int, next_nid: int) -> bool:
        ...

    @abstractmethod
    def propagate(self, nid: int, next_nid: int):
        ...

    def __call__(self, cfg: CFG) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
        self.cfg = cfg

        self.all_defs = self.build_defs()
        self.gen_dict = self.build_gen()
        self.kill_dict = self.build_kill()

        self.in_dict = defaultdict(set)
        self.out_dict = defaultdict(set)

        self.visited = set()
        self.worklist = []
        for _ in self.pre_loop_init():
            while self.worklist:
                nid = self.worklist.pop()
                self.apply_flow_eq(nid)
                for next_nid in self.next_nodes(nid):
                    if next_nid not in self.visited or self.can_propagate(
                        nid, next_nid
                    ):
                        self.propagate(nid, next_nid)
                        self.worklist.append(next_nid)
                        self.visited.add(next_nid)

        return self.in_dict, self.out_dict


class PossiblyReachingDefinitions(DataFlowAlgorithm):
    def build_gen(self) -> dict[int, set[int]]:
        gen_dict = defaultdict[int, set[int]](set)
        for nid in yield_all_defs(self.cfg):
            gen_dict[nid].add(nid)
        return gen_dict

    def pre_loop_init(self) -> Iterable[None]:
        for entry_nid in self.get_entry_node():
            self.in_dict[entry_nid] = set()
            self.visited.add(entry_nid)
            self.worklist.append(entry_nid)
            yield

    def get_entry_node(self) -> Iterable[int]:
        node_ids = self.cfg.get_node_ids()
        for nid in node_ids:
            if self.cfg.get_type(nid) == "Entry":
                yield nid

    def apply_flow_eq(self, nid: int) -> None:
        self.out_dict[nid] = self.gen_dict[nid] | (
            self.in_dict[nid] - self.kill_dict[nid]
        )

    def next_nodes(self, nid: int) -> Iterable[int]:
        return self.cfg.get_any_children(nid)

    def can_propagate(self, nid: int, next_nid: int) -> bool:
        return (self.out_dict[nid] - self.in_dict[next_nid]) != set()

    def propagate(self, nid: int, next_nid: int) -> None:
        self.in_dict[next_nid] |= self.out_dict[nid]


class PossibleReachableReferences(DataFlowAlgorithm):
    def build_gen(self) -> dict[int, set[int]]:
        gen_dict = defaultdict[int, set[int]](set)
        for nid in yield_all_refs(self.cfg):
            gen_dict[nid].add(nid)
        return gen_dict

    def pre_loop_init(self) -> Iterable[None]:
        for exit_nid in self.get_exit_node():
            self.out_dict[exit_nid] = set()
            self.visited.add(exit_nid)
            self.worklist.append(exit_nid)
            yield

    def get_exit_node(self) -> Iterable[int]:
        node_ids = self.cfg.get_node_ids()
        for nid in node_ids:
            if self.cfg.get_type(nid) == "Exit":
                yield nid

    def apply_flow_eq(self, nid: int) -> None:
        self.in_dict[nid] = self.gen_dict[nid] | (
            self.out_dict[nid] - self.kill_dict[nid]
        )

    def next_nodes(self, nid: int) -> Iterable[int]:
        return self.cfg.get_any_parents(nid)

    def can_propagate(self, nid: int, next_nid: int) -> bool:
        return (self.in_dict[nid] - self.out_dict[next_nid]) != set()

    def propagate(self, nid: int, next_nid: int) -> None:
        self.out_dict[next_nid] |= self.in_dict[nid]

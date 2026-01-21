from .graph import GraphData


class GraphListDataset:
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> GraphData:
        return self.data_list[idx]


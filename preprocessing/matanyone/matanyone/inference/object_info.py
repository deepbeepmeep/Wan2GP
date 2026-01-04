class ObjectInfo:
    """
    Store meta information for an object
    """
    def __init__(self, id: int):
        self.id = id
        self.poke_count = 0  # count number of detections missed

    def poke(self) -> None:
        self.poke_count += 1

    def unpoke(self) -> None:
        self.poke_count = 0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if type(other) == int:
            return self.id == other
        return self.id == other.id

    def __repr__(self):
        return f'(ID: {self.id})'

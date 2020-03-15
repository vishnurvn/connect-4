from .exceptions import ColumnFilledException


class ConnectBoard:
    def __init__(self, height=6, width=7):
        self._height = height
        self._width = width
        self._board = [[0 for _ in range(self._width)] for _ in range(self._height)]

    def __getitem__(self, item):
        row, col = item
        return self._board[row][col]

    def __setitem__(self, key, value):
        row, col = key
        self._board[row][col] = value

    def board_input(self, col, value):
        for row in range(self._height - 1, -1, -1):
            if self._board[row][col] == 0:
                self._board[row][col] = value
                return None
        raise ColumnFilledException

    def check_board(self) -> int:
        pass

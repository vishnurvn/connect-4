from unittest import TestCase

from core.board import ConnectBoard
from core.exceptions import ColumnFilledException
from .data import connect_data


class TestBoard(TestCase):

    def setUp(self) -> None:
        self.connect_board = ConnectBoard()

    def test_count_slots(self):
        self.assertEqual(self.connect_board._height, 6)
        self.assertEqual(self.connect_board._width, 7)

    def test_all_zero(self):
        for row in range(self.connect_board._height):
            for col in range(self.connect_board._width):
                self.assertEqual(self.connect_board[row, col], 0)

    def test_board_get_item(self):
        self.assertEqual(self.connect_board[0, 0], 0)

    def test_board_get_item_invalid_index(self):
        try:
            self.connect_board[8, 9]
        except IndexError:
            pass
        else:
            self.fail('Some other exception raised')

    def test_board_set_item(self):
        self.connect_board[1, 1] = 1
        self.assertEqual(self.connect_board[1, 1], 1)

    def test_board_set_item_invalid_index(self):
        try:
            self.connect_board[8, 9] = 1
        except IndexError:
            pass
        else:
            self.fail('Some other exception raised')

    def test_board_input_one(self):
        self.connect_board.board_input(1, 1)
        self.assertEqual(self.connect_board[5, 1], 1)

    def test_board_input_three(self):
        for _ in range(3):
            self.connect_board.board_input(1, 1)
        self.assertEqual(self.connect_board[3, 1], 1)
        self.assertEqual(self.connect_board[2, 1], 0)

    def test_board_input_full(self):
        for _ in range(self.connect_board._height):
            self.connect_board.board_input(1, 1)
        try:
            self.connect_board.board_input(1, 1)
        except ColumnFilledException:
            pass
        else:
            self.fail('Some other exceptions raised')


class TestBoardCheck(TestCase):

    def test_check_board(self):
        connect_board = ConnectBoard()
        for data in connect_data:
            connect_board._board = data
            status, winner = connect_board.check_board()
            self.assertTrue(status)
            self.assertEqual(winner, 1)

import pytest

from synthetic.positions import assign_split, parse_position, square_name

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_parses_full_fen_bare_placement_and_chesscog_prefix():
    assert parse_position(f"{START} b KQkq - 0 1").white_to_move is False
    assert parse_position(START).white_to_move is None
    assert parse_position(f"W{START}").white_to_move is True
    assert parse_position(f"B{START}").placement == START


def test_leading_white_bishop_is_not_a_side_to_move_prefix():
    position = parse_position("B7/8/8/8/8/8/8/k6K")
    assert position.placement == "B7/8/8/8/8/8/8/k6K"
    assert position.white_to_move is None


@pytest.mark.parametrize("line", ["", "8/8/8", "9/8/8/8/8/8/8/8", f"{START} x"])
def test_rejects_malformed_lines(line):
    with pytest.raises(ValueError):
        parse_position(line)


def test_pieces_are_indexed_from_a1():
    pieces = parse_position("8/8/8/8/8/8/8/R6k").pieces()
    assert pieces == [("R", 0, 0), ("k", 7, 0)]
    assert square_name(0, 0) == "a1" and square_name(7, 7) == "h8"


def test_split_depends_only_on_placement():
    fractions = {"train": 0.9, "val": 0.03, "test": 0.07}
    assert assign_split(START, fractions) == assign_split(START, fractions)
    splits = {assign_split(f"{START}{i}", fractions) for i in range(200)}
    assert splits == {"train", "val", "test"}

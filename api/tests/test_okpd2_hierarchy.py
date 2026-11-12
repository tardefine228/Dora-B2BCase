from ml.okpd2 import is_code_ancestor, unique_detailed_code


def test_okpd2_hierarchy_supports_dot_and_one_digit_expansion() -> None:
    assert is_code_ancestor("27.20.22", "27.20.22.000")
    assert is_code_ancestor("26.12.2", "26.12.20")
    assert is_code_ancestor("10.51.4", "10.51.40.120")
    assert is_code_ancestor("28.13.2", "28.13.28")
    assert not is_code_ancestor("26.60.12.119", "26.70.22.150")
    assert not is_code_ancestor("27.20.22", "27.20.220000")
    assert not is_code_ancestor("27.20.22", "27.20.22")


def test_only_one_complete_branch_can_be_collapsed() -> None:
    assert unique_detailed_code({"27.20.22", "27.20.22.000"}) == "27.20.22.000"
    assert unique_detailed_code({"26.12.2", "26.12.20"}) == "26.12.20"
    assert unique_detailed_code({"26.60.12.119", "26.70.22.150"}) is None
    assert unique_detailed_code({"26.11.3", "26.11.30", "26.11.31"}) is None
    assert unique_detailed_code({"27.20.22"}) is None

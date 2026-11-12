from app.services.okpd_reference import code_description, reference


def test_reference_has_full_dictionary_and_distinct_butter_names():
    assert len(reference()["codes"]) > 20000
    assert "пищевкусовыми" in code_description("10.51.30.100")
    assert code_description("10.51.30.110") == "Масло сливочное"
    assert code_description("21.20.24.120") == "Шовные материалы"


def test_missing_code_is_not_replaced_with_ancestor_or_guessed():
    assert code_description("22.22.14.000") is None
    assert code_description("not-a-code") is None


def test_identical_alto_names_have_different_instrument_groups():
    from app.services.okpd_reference import code_context

    assert code_description("32.20.12.112") == code_description("32.20.13.163") == "Альты"
    assert "струн" in code_context("32.20.12.112").lower()
    assert "духов" in code_context("32.20.13.163").lower()

from sdialog.audio.whisper_normalizer import EnglishNumberNormalizer, EnglishSpellingNormalizer, EnglishTextNormalizer


def test_english_number_normalizer():
    normalizer = EnglishNumberNormalizer()

    # Test basic numbers
    assert normalizer("one") == "one"
    assert normalizer("two") == "2"
    assert normalizer("ten") == "10"
    assert normalizer("one two three") == "123"
    assert normalizer("twenty five") == "25"
    assert normalizer("one hundred") == "100"
    assert normalizer("one thousand five hundred") == "1500"

    # Test ordinals
    assert normalizer("first") == "1st"
    assert normalizer("twenty third") == "23rd"
    assert normalizer("one hundredth") == "100th"

    # Test cardinals/plurals
    assert normalizer("tens") == "10s"
    assert normalizer("millions") == "1000000s"

    # Test currency
    assert normalizer("ten dollars") == "$10"
    assert normalizer("twenty five euros") == "€25"

    # Test decimals and points
    assert normalizer("two point five") == "2.5"
    assert normalizer("o point two") == "0.2"
    assert normalizer("one oh one") == "101"

    # Test with prefixes
    assert normalizer("minus twenty") == "-20"
    assert normalizer("positive five") == "+5"

    # Test complex cases
    assert normalizer("one hundred and twenty three") == "123"
    assert normalizer("two thousand twenty three") == "2023"


def test_english_spelling_normalizer():
    normalizer = EnglishSpellingNormalizer()
    assert normalizer("accessorise") == "accessorize"
    assert normalizer("colour") == "color"
    assert normalizer("haematology") == "hematology"
    # A word not in the mapping
    assert normalizer("sdialog") == "sdialog"


def test_english_text_normalizer():
    normalizer = EnglishTextNormalizer()

    # Test contractions
    assert normalizer("I won't go.") == "i will not go"
    assert normalizer("She's happy.") == "she is happy"
    assert normalizer("We're leaving.") == "we are leaving"

    # Test special ignores
    assert normalizer("Uhm, I think so.") == "uhm i think so"
    assert normalizer("Hmm... maybe.") == " maybe"

    # Test symbols and punctuation
    assert normalizer("This is a test.") == "this is a test"
    assert normalizer("[bracketed text] is removed") == "is removed"
    assert normalizer("So (this) is also gone") == "so is also gone"

    # Test full normalization
    assert normalizer(
        "I've got 2 apples and she's got 3. That's five apples in total, colour-wise."
    ) == "i have got two apples and she has got three that is five apples in total color wise"

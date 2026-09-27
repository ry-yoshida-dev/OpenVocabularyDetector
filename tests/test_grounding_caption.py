from open_vocabulary_detector.backends.grounding_dino import CharacterSpan, GroundingCaption


def test_caption_spans_point_to_text_queries() -> None:
    caption: GroundingCaption = GroundingCaption.from_queries(("Cat", "traffic cone"))
    assert caption.text == "cat . traffic cone ."
    assert [caption.text[span.start : span.end] for span in caption.query_spans] == ["cat", "traffic cone"]


def test_character_span_contains() -> None:
    span: CharacterSpan = CharacterSpan(start=6, end=18)
    assert span.contains(6, 13)
    assert not span.contains(0, 0)
    assert not span.contains(17, 19)

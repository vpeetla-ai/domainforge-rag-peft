from domainforge.prep.bitext import stratified_sample_records


def test_stratified_sample_covers_multiple_intents():
    records = []
    for intent in ("track_order", "get_refund", "cancel_order"):
        for i in range(50):
            records.append({"instruction": f"{intent} msg {i}", "intent": intent, "category": "ORDER"})
    sampled = stratified_sample_records(records, max_rows=30, seed=42)
    intents = {r["intent"] for r in sampled}
    assert len(intents) >= 3
    assert len(sampled) == 30

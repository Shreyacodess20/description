def one_hot_encode(categories: list[str]) -> tuple[list[str], list[list[int]]]:
    distinct = sorted(set(categories))
    index_map = {cat: i for i, cat in enumerate(distinct)}
    rows = [[1 if i == index_map[c] else 0 for i in range(len(distinct))] for c in categories]
    return distinct, rows

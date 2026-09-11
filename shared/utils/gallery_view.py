"""A bounded view of a gallery; indices in application state remain absolute."""


def gallery_offset(count, limit):
    return max(0, count - limit) if limit > 0 else 0


def gallery_window(paths, selected, limit):
    offset = gallery_offset(len(paths), limit)
    return paths[offset:], selected - offset if offset <= selected < len(paths) else None, offset

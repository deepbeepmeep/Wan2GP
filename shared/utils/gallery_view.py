"""A bounded view of a gallery; indices in application state remain absolute."""


def gallery_offset(count, limit):
    return max(0, count - limit) if limit > 0 else 0


def gallery_window(paths, selected, limit, *, keep_selected=False):
    offset = gallery_offset(len(paths), limit)
    if keep_selected and 0 <= selected < len(paths):
        offset = min(offset, selected)
    visible = paths[offset:offset + limit] if limit > 0 else paths[offset:]
    return visible, selected - offset if offset <= selected < offset + len(visible) else None, offset

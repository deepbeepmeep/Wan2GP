"""Export the composition used by YuE2, without naming or saving files."""


def score_side_files(abc, midi=None):
    if midi is None:
        from symusic import Score
        midi = Score.from_abc(abc).dumps_midi()
    return {".abc": abc.encode("utf-8"), ".mid": midi}

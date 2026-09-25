


STRUCTURE_LABELS = ['silence', 'intro', 'outro', 'verse', 'chorus', 'bridge', 'pre-chorus', 'post-chorus',
                                'interlude', 'fade-out', 'loop', 'rap', 'preshot', 'irregular']


STRUCTURE_LABELS = ['silence', 'intro', 'outro', 'verse', 'chorus', 'bridge', 'pre-chorus', 'post-chorus',
                               'interlude', 'fade-out', 'loop', 'rap', 'preshot', 'irregular', 'instrumental',
                               'intro and verse', 'pre-chorus and chorus', 'verse and pre-chorus', 'solo', 'theme', 'development', 'variation', 'pre-outro']


MIREX_STRUCTURE_LABEL_REDUCTION = {
    'silence': 'silence',
    'intro': 'intro',
    'outro': 'outro',
    'verse': 'verse',
    'chorus': 'chorus',
    'bridge': 'bridge',
    'pre-chorus': 'verse',
    'post-chorus': 'verse',
    'interlude': 'inst',
    'inst': 'inst',
    'fade-out': 'outro',
    'loop': 'chorus',
    'rap': 'verse',
    'preshot': 'inst',
    'irregular': 'verse',
}
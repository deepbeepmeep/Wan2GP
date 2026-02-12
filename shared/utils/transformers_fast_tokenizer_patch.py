import json
import os
import pickle
import sys


_PATCH_ALLOWED_PATHS = None
_ORIG_FAST_INIT = None
_DISABLE_FULL_TOKENIZER_PICKLE_CACHE = True  # Temporary: disable Python pickle tokenizer cache load/dump.


def _normalize_path(path):
    if not path:
        return None
    try:
        return os.path.normcase(os.path.abspath(path))
    except Exception:
        return None


def _path_allowed(path):
    if not _PATCH_ALLOWED_PATHS:
        return False
    norm = _normalize_path(path)
    if norm is None:
        return False
    for allowed in _PATCH_ALLOWED_PATHS:
        if allowed is None:
            continue
        try:
            if os.path.commonpath([norm, allowed]) == allowed:
                return True
        except Exception:
            if norm.startswith(allowed):
                return True
    return False


def _load_cached_tokenizer(tokenizer_file, TokenizerFast):
    if not tokenizer_file:
        return None
    return TokenizerFast.from_file(tokenizer_file)


def patch_pretrained_tokenizer_fast(allow_paths=None):
    global _PATCH_ALLOWED_PATHS
    global _ORIG_FAST_INIT
    if allow_paths is not None:
        _PATCH_ALLOWED_PATHS = [_normalize_path(p) for p in allow_paths if p]

    try:
        import transformers.tokenization_utils_fast as tuf
    except Exception:
        return

    cls = tuf.PreTrainedTokenizerFast
    if getattr(cls, "_wan2gp_fast_init_patched", False):
        return

    if _ORIG_FAST_INIT is None:
        _ORIG_FAST_INIT = cls.__init__

    def _patched_init(self, *args, **kwargs):
        fast_tokenizer_file = kwargs.get("tokenizer_file")
        from_slow = kwargs.get("from_slow", False)
        if not fast_tokenizer_file or from_slow or not _path_allowed(fast_tokenizer_file):
            return _ORIG_FAST_INIT(self, *args, **kwargs)

        try:
            fast_tokenizer = _load_cached_tokenizer(fast_tokenizer_file, tuf.TokenizerFast)
            if fast_tokenizer is None:
                return _ORIG_FAST_INIT(self, *args, **kwargs)
            kwargs["tokenizer_object"] = fast_tokenizer
        except Exception:
            return _ORIG_FAST_INIT(self, *args, **kwargs)

        tokenizer_object = kwargs.pop("tokenizer_object", None)
        slow_tokenizer = kwargs.pop("__slow_tokenizer", None)
        fast_tokenizer_file = kwargs.pop("tokenizer_file", None)
        from_slow = kwargs.pop("from_slow", False)
        added_tokens_decoder = kwargs.pop("added_tokens_decoder", {})
        self.add_prefix_space = kwargs.get("add_prefix_space", False)

        if from_slow and slow_tokenizer is None and self.slow_tokenizer_class is None:
            raise ValueError(
                "Cannot instantiate this tokenizer from a slow version. If it's based on sentencepiece, make sure you "
                "have sentencepiece installed."
            )

        if tokenizer_object is not None:
            fast_tokenizer = tokenizer_object
        else:
            fast_tokenizer = tuf.TokenizerFast.from_file(fast_tokenizer_file)

        self._tokenizer = fast_tokenizer

        if slow_tokenizer is not None:
            kwargs.update(slow_tokenizer.init_kwargs)

        self._decode_use_source_tokenizer = False

        _truncation = self._tokenizer.truncation

        if _truncation is not None:
            self._tokenizer.enable_truncation(**_truncation)
            kwargs.setdefault("max_length", _truncation["max_length"])
            kwargs.setdefault("truncation_side", _truncation["direction"])
            kwargs.setdefault("stride", _truncation["stride"])
            kwargs.setdefault("truncation_strategy", _truncation["strategy"])
        else:
            self._tokenizer.no_truncation()

        _padding = self._tokenizer.padding
        if _padding is not None:
            self._tokenizer.enable_padding(**_padding)
            kwargs.setdefault("pad_token", _padding["pad_token"])
            kwargs.setdefault("pad_token_type_id", _padding["pad_type_id"])
            kwargs.setdefault("padding_side", _padding["direction"])
            kwargs.setdefault("max_length", _padding["length"])
            kwargs.setdefault("pad_to_multiple_of", _padding["pad_to_multiple_of"])

        tuf.PreTrainedTokenizerBase.__init__(self, **kwargs)
        self._tokenizer.encode_special_tokens = self.split_special_tokens

        added_tokens_decoder_hash = {hash(repr(token)) for token in self.added_tokens_decoder}
        tokens_to_add = [
            token
            for index, token in sorted(added_tokens_decoder.items(), key=lambda x: x[0])
            if hash(repr(token)) not in added_tokens_decoder_hash
        ]
        encoder_set = set(self.added_tokens_encoder.keys())
        for token in tokens_to_add:
            if isinstance(token, tuf.AddedToken):
                encoder_set.add(token.content)
            else:
                encoder_set.add(str(token))
        tokens_to_add_set = set(tokens_to_add)
        tokens_to_add += [
            token
            for token in self.all_special_tokens_extended
            if token not in encoder_set and token not in tokens_to_add_set
        ]

        if len(tokens_to_add) > 0:
            special_tokens = set(self.all_special_tokens)
            tokens = []
            append = tokens.append
            for token in tokens_to_add:
                if isinstance(token, tuf.AddedToken):
                    content = token.content
                    if (not token.special) and (content in special_tokens):
                        token.special = True
                    append(token)
                else:
                    append(tuf.AddedToken(token, special=(token in special_tokens)))
            if tokens:
                self.add_tokens(tokens)

        try:
            pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
            if pre_tok_state.get("add_prefix_space", self.add_prefix_space) != self.add_prefix_space:
                pre_tok_class = getattr(tuf.pre_tokenizers_fast, pre_tok_state.pop("type"))
                pre_tok_state["add_prefix_space"] = self.add_prefix_space
                self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)
        except Exception:
            pass

    cls.__init__ = _patched_init
    cls._wan2gp_fast_init_patched = True


def unpatch_pretrained_tokenizer_fast():
    global _ORIG_FAST_INIT
    if _ORIG_FAST_INIT is None:
        return
    try:
        import transformers.tokenization_utils_fast as tuf
    except Exception:
        return
    cls = tuf.PreTrainedTokenizerFast
    if not getattr(cls, "_wan2gp_fast_init_patched", False):
        return
    cls.__init__ = _ORIG_FAST_INIT
    cls._wan2gp_fast_init_patched = False


def _get_transformers_version():
    try:
        import transformers as _transformers
        return getattr(_transformers, "__version__", None)
    except Exception:
        return None


def _get_tokenizers_version():
    try:
        import tokenizers as _tokenizers
        return getattr(_tokenizers, "__version__", None)
    except Exception:
        return None


def _collect_tokenizer_files(tokenizer_dir):
    candidates = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "config.json",
        "sentencepiece.bpe.model",
        "tokenizer.model",
    ]
    files = []
    for name in candidates:
        path = os.path.join(tokenizer_dir, name)
        if os.path.isfile(path):
            try:
                stat = os.stat(path)
                files.append({"path": name, "mtime": stat.st_mtime, "size": stat.st_size})
            except OSError:
                files.append({"path": name, "mtime": None, "size": None})
    return files


def _sanitize_cache_tag(tag):
    if not tag:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(tag))
    return safe.strip("._-")


def _cache_paths(tokenizer_dir, cache_tag=None):
    suffix = _sanitize_cache_tag(cache_tag)
    if suffix:
        cache_file = os.path.join(tokenizer_dir, f"tokenizer.wgp.full.{suffix}.pkl")
        meta_file = os.path.join(tokenizer_dir, f"tokenizer.wgp.full.{suffix}.meta.json")
    else:
        cache_file = os.path.join(tokenizer_dir, "tokenizer.wgp.full.pkl")
        meta_file = os.path.join(tokenizer_dir, "tokenizer.wgp.full.meta.json")
    return cache_file, meta_file


def _read_cache_meta(meta_file):
    try:
        with open(meta_file, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _meta_matches(meta, tokenizer_dir):
    if not meta:
        return False
    if tuple(meta.get("py_version", [])) != tuple(sys.version_info[:3]):
        return False
    if meta.get("transformers_version") != _get_transformers_version():
        return False
    if meta.get("tokenizers_version") != _get_tokenizers_version():
        return False
    expected_files = meta.get("files", [])
    current_files = _collect_tokenizer_files(tokenizer_dir)
    if len(expected_files) != len(current_files):
        return False
    current_map = {f.get("path"): f for f in current_files}
    for entry in expected_files:
        cur = current_map.get(entry.get("path"))
        if cur is None:
            return False
        if entry.get("mtime") != cur.get("mtime") or entry.get("size") != cur.get("size"):
            return False
    return True


def _load_full_tokenizer_cache(tokenizer_dir, cache_tag=None):
    if _DISABLE_FULL_TOKENIZER_PICKLE_CACHE:
        return None
    cache_file, meta_file = _cache_paths(tokenizer_dir, cache_tag=cache_tag)
    if not os.path.isfile(cache_file) or not os.path.isfile(meta_file):
        return None
    meta = _read_cache_meta(meta_file)
    if not _meta_matches(meta, tokenizer_dir):
        return None
    try:
        with open(cache_file, "rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _save_full_tokenizer_cache(tokenizer_dir, tokenizer, cache_tag=None):
    if _DISABLE_FULL_TOKENIZER_PICKLE_CACHE:
        return
    cache_file, meta_file = _cache_paths(tokenizer_dir, cache_tag=cache_tag)
    meta = {
        "py_version": list(sys.version_info[:3]),
        "transformers_version": _get_transformers_version(),
        "tokenizers_version": _get_tokenizers_version(),
        "files": _collect_tokenizer_files(tokenizer_dir),
    }
    try:
        with open(cache_file, "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(meta_file, "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
    except Exception:
        pass


def load_cached_lm_tokenizer(tokenizer_dir, loader_fn, cache_tag=None):
    if not tokenizer_dir:
        return loader_fn()
    cached = _load_full_tokenizer_cache(tokenizer_dir, cache_tag=cache_tag)
    if cached is not None:
        return cached
    patch_pretrained_tokenizer_fast(allow_paths=[tokenizer_dir])
    try:
        tokenizer = loader_fn()
    finally:
        unpatch_pretrained_tokenizer_fast()
    _save_full_tokenizer_cache(tokenizer_dir, tokenizer, cache_tag=cache_tag)
    return tokenizer

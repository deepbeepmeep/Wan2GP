from pydub import AudioSegment

class AudioData:
    def __init__(
        self,
        samples,
        mel_chunks,
        duration,
        samples_count,
        sample_rate,
        metadata: any
    ):
        self._samples = samples
        self._mel_chunks = mel_chunks
        self._duration = duration
        self._samples_count = samples_count
        self._sample_rate = sample_rate
        self._metadata = metadata

    @property
    def samples(self):
        return self._samples

    @property
    def mel_chunks(self):
        return self._mel_chunks

    @property
    def duration(self):
        return self._duration

    @property
    def metadata(self):
        return self._metadata

    @property
    def samples_count(self):
        return self._samples_count

    @property
    def sample_rate(self):
        return self._sample_rate

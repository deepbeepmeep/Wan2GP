"""CPU encoding and browser rendering for bounded TinyVAE video previews."""
import base64
import io
from dataclasses import dataclass
from fractions import Fraction

from PIL import Image


@dataclass(frozen=True)
class VideoPreview:
    image: Image.Image
    video: bytes

    def to_html(self, modal=False):
        uri = 'data:video/mp4;base64,' + base64.b64encode(self.video).decode('ascii')
        style = 'display:block;max-width:100%;max-height:80vh' if modal else 'display:block;max-width:100%;max-height:200px'
        controls = ' controls' if modal else ''
        playback = '' if modal else ' title="Click to Pause or Resume" onclick="event.stopPropagation();window.__wangpGenerationPreviewPaused = !this.paused;this.paused ? this.play() : this.pause()" onplay="if(window.__wangpGenerationPreviewPaused) this.pause()"'
        if not modal:
            style += ';cursor:pointer'
        return f'<div style="display:flex;justify-content:center"><video src="{uri}" autoplay loop muted playsinline{controls}{playback} style="{style}"></video></div>'


def encode_video(frames, fps, abort_check):
    import av

    buffer = io.BytesIO()
    with av.open(buffer, mode='w', format='mp4', options={'movflags': 'frag_keyframe+empty_moov+default_base_moof'}) as container:
        stream = container.add_stream('libx264', rate=Fraction(fps).limit_denominator(10000))
        stream.width, stream.height = frames[0].size
        stream.pix_fmt = 'yuv420p'
        stream.thread_count = 2
        stream.options = {'preset': 'ultrafast', 'crf': '25', 'tune': 'zerolatency'}
        for index, image in enumerate(frames):
            if abort_check():
                return None
            frame = av.VideoFrame.from_image(image)
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return VideoPreview(frames[0], buffer.getvalue())

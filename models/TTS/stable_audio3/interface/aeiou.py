# Modified from https://github.com/drscotthawley/aeiou/blob/main/aeiou/viz.py under Apache 2.0 License
# License can be found in LICENSES/LICENSE_AEIOU.txt

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

import torch

import torchaudio.transforms as T
from einops import rearrange

import numpy as np

def embeddings_table(tokens):
    from wandb import Table
    from pandas import DataFrame

    "make a table of embeddings for use with wandb"
    features, labels = [], []
    embeddings = rearrange(tokens, 'b d n -> b n d') # each demo sample is n vectors in d-dim space
    for i in range(embeddings.size()[0]):  # nested for's are slow but sure ;-) 
        for j in range(embeddings.size()[1]):
            features.append(embeddings[i,j].detach().cpu().numpy())
            labels.append([f'demo{i}'])    # labels does the grouping / color for each point
    features = np.array(features)
    labels = np.concatenate(labels, axis=0)
    cols = [f"dim_{i}" for i in range(features.shape[1])]
    df   = DataFrame(features, columns=cols)
    df['LABEL'] = labels
    return Table(columns=df.columns.to_list(), data=df.values)

def project_down(tokens,     # batched high-dimensional data with dims (b,d,n)
            proj_dims=3,     # dimensions to project to
            method='pca',    # projection method: 'pca'|'umap'
            n_neighbors=10, # umap parameter for number of neighbors
            min_dist=0.3,    # umap param for minimum distance
            debug=False,     # print more info while running
            **kwargs,        # other params to pass to umap, cf. https://umap-learn.readthedocs.io/en/latest/parameters.html 
            ):
    "this projects to lower dimenions, grabbing the first _`proj_dims`_ dimensions"
    method = method.lower()
    A = rearrange(tokens, 'b d n -> (b n) d') # put all the vectors into the same d-dim space
    if A.shape[-1] > proj_dims: 
        if method=='umap':
            from umap import UMAP
            proj_data = UMAP(n_components=proj_dims, n_neighbors=n_neighbors, min_dist=min_dist,
                            metric='correlation', **kwargs).fit_transform(A.cpu().numpy())
            proj_data = torch.from_numpy(proj_data).to(tokens.device)
        else:  # pca
            (U, S, V) = torch.pca_lowrank(A)
            proj_data = torch.matmul(A, V[:, :proj_dims])  # this is the actual PCA projection step
    else:
        proj_data = A
    if debug: print("proj_data.shape =",proj_data.shape)
    return torch.reshape(proj_data, (tokens.size()[0], -1, proj_dims)) # put it in shape [batch, n, proj_dims]


def proj_pca(tokens, proj_dims=3):
    return project_down(do_proj, method='pca', proj_dims=proj_dims)

def point_cloud(
    tokens,                  # embeddings / latent vectors. shape = (b, d, n)
    method='pca',            # projection method for 3d mapping: 'pca' | 'umap'
    color_scheme='batch',    # 'batch': group by sample; integer n: n groups, sequentially,  otherwise color sequentially by time step
    output_type='wandbobj',  # plotly | points | wandbobj.  NOTE: WandB can do 'plotly' directly!
    mode='markers',    # plotly scatter mode.  'lines+markers' or 'markers'
    size=3,            # size of the dots
    line=dict(color='rgba(10,10,10,0.01)'),  # if mode='lines+markers', plotly line specifier. cf. https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.scatter3d.html#plotly.graph_objects.scatter3d.Line
    ds_preproj=1,         # EXPERIMENTAL: downsampling factor before projecting  (1=no downsampling). Could screw up colors
    ds_preplot=1,         # EXPERIMENTAL: downsampling factor before plotting (1=no downsampling). Could screw up colors
    debug=False,          # print more info
    colormap=None,        # valid color map to use, None=defaults
    darkmode=False,       # dark background, white fonts
    layout_dict=None,      # extra plotly layout options such as camera orientation
    rgb_float = False,   # if True, color_scheme is RGB float values
    **kwargs,             # anything else to pass along
    ):
    "returns a 3D point cloud of the tokens" 
    if ds_preproj != 1: 
        tokens = tokens[torch.randperm(tokens.size()[0])]  # EXPERIMENTAL: to 'correct' for possible weird effects of downsampling
        tokens = tokens[::ds_preproj]
        if debug: print("tokens.shape =",tokens.shape)

    data = project_down(tokens, method=method, debug=debug, **kwargs).cpu().numpy()
    if debug: print("data.shape =",data.shape)
    if data.shape[-1] < 3: # for data less than 3D, embed it in 3D 
        data = np.pad(data, ((0,0),(0,0),(0, 3-data.shape[-1])), mode='constant', constant_values=0)

    bytime = False
    points = []         
    if color_scheme=='batch': # all dots in same batch index same color, each batch-index unique (almost)
        ncolors = data.shape[0]
        cmap, norm = cm.tab20, Normalize(vmin=0, vmax=ncolors)
    elif isinstance(color_scheme, int) or color_scheme.isnumeric():  # n groups, by batch-indices, sequentially
        ncolors = int(color_scheme)
        cmap, norm = cm.tab20, Normalize(vmin=0, vmax=ncolors)
    else:                                                    # time steps match up
        bytime, ncolors = True, data.shape[1]
        cmap, norm = cm.viridis, Normalize(vmin=0, vmax=ncolors)

    cmap = cmap if colormap is None else colormap # overwrite default cmap with user choice if given 

    points = []  
    for bi in range(data.shape[0]):  # batch index
        if color_scheme=='batch': 
            [r, g, b, _] = [int(255*x) for x in cmap(norm(bi+1))] 
        elif isinstance(color_scheme, int) or color_scheme.isnumeric():
            grouplen = data.shape[0]//(ncolors)
            #if debug: print(f"bi, grouplen, bi//grouplen = ",bi, grouplen, bi//grouplen)
            [r, g, b, _] = [int(255*x) for x in cmap(norm(bi//grouplen))] 
            #if debug: print("r,g,b = ",r,g,b)

        if rgb_float: [r, g, b] = [x/255 for x in [r, g, b]]

        for n in range(data.shape[1]):    # across time
            if bytime: [r, g, b, _] = [int(255*x) for x in cmap(norm(n))] 
            points.append([data[bi,n,0], data[bi,n,1], data[bi,n,2], r, g, b])  # include dot colors with point coordinates

    point_cloud = np.array(points)
        
    if output_type == 'points':
        return point_cloud
    elif output_type =='plotly':
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Scatter3d(
            x=point_cloud[::ds_preplot,0], y=point_cloud[::ds_preplot,1], z=point_cloud[::ds_preplot,2], 
            marker=dict(size=size,  color=point_cloud[:,3:6]),
            mode=mode, 
            # show batch index and time index in tooltips: 
            text=[ f'bi: {i*ds_preplot}, ti: {j}' for i in range(data.shape[0]//ds_preplot) for j in range(data.shape[1]) ],
            line=line,
            )])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0)) # tight layout
        if darkmode: 
            fig.layout.template = 'plotly_dark'
            if isinstance(darkmode, str):   # 'rgb(12,15,24)'gradio margins in dark mode
                fig.update_layout( paper_bgcolor=darkmode) 
        if layout_dict:
                fig.update_layout( **layout_dict ) 
            
        if debug: print("point_cloud: fig made. returning")
        return fig
    else:
        from wandb import Object3D
        return Object3D(point_cloud)
    
def pca_point_cloud(
    tokens,                  # embeddings / latent vectors. shape = (b, d, n)
    color_scheme='batch',    # 'batch': group by sample, otherwise color sequentially
    output_type='wandbobj',  # plotly | points | wandbobj.  NOTE: WandB can do 'plotly' directly!
    mode='markers',    # plotly scatter mode.  'lines+markers' or 'markers'
    size=3,            # size of the dots
    line=dict(color='rgba(10,10,10,0.01)'),  # if mode='lines+markers', plotly line specifier. cf. https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.scatter3d.html#plotly.graph_objects.scatter3d.Line
    **kwargs,
    ):
    return point_cloud(tokens, method='pca', color_scheme=color_scheme, output_type=output_type,
        mode=mode, size=size, line=line, **kwargs)

def power_to_db(spec, *, amin = 1e-10):
    magnitude = np.asarray(spec)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, 1))

    log_spec = np.maximum(log_spec, log_spec.max() - 80)

    return log_spec

def mel_spectrogram(waveform, power=2.0, sample_rate=48000, db=False, n_fft=1024, n_mels=128, debug=False):
    "calculates data array for mel spectrogram (in however many channels)"
    win_length = None
    hop_length = n_fft//2 # 512

    mel_spectrogram_op = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, 
        hop_length=hop_length, center=True, pad_mode="reflect", power=power, 
        norm='slaney', onesided=True, n_mels=n_mels, mel_scale="htk")

    melspec = mel_spectrogram_op(waveform.float())
    if db: 
        amp_to_db_op = T.AmplitudeToDB()
        melspec = amp_to_db_op(melspec)
    if debug:
        print_stats(melspec, print=print) 
        print(f"torch.max(melspec) = {torch.max(melspec)}")
        print(f"melspec.shape = {melspec.shape}")
    return melspec

def spectrogram_image(
        spec,
        title=None,
        ylabel='freq_bin',
        aspect='auto',
        xmax=None,
        db_range=[35,120],
        justimage=False,
        figsize=(5, 4), # size of plot (if justimage==False)
        context_mask=None, # 1D inpaint mask: 1 = context/kept (tinted), 0 = inpainted/generated
        context_color=(0.8, 0.2, 0.2, 0.3), # RGBA color for context region overlay (red, visible against viridis)
    ):
    "Modified from PyTorch tutorial https://pytorch.org/tutorials/beginner/audio_feature_extractions_tutorial.html"
    fig = Figure(figsize=figsize, dpi=100) if not justimage else Figure(figsize=(4.145, 4.145), dpi=100, tight_layout=True)
    canvas = FigureCanvasAgg(fig)
    axs = fig.add_subplot()
    spec = spec.squeeze()
    im = axs.imshow(power_to_db(spec), origin='lower', aspect=aspect, vmin=db_range[0], vmax=db_range[1])

    # Overlay tint on context (kept) regions where mask == 1
    if context_mask is not None:
        context_mask = np.asarray(context_mask).flatten()
        num_frames = spec.shape[-1]
        # Resize mask to match spectrogram frames if needed
        if len(context_mask) != num_frames:
            indices = np.linspace(0, len(context_mask) - 1, num_frames).astype(int)
            context_mask = context_mask[indices]
        # Find contiguous context regions (mask > 0.5) and draw axvspan overlays
        in_span = False
        span_start = 0
        for j in range(len(context_mask)):
            if context_mask[j] > 0.5 and not in_span:
                span_start = j
                in_span = True
            elif context_mask[j] <= 0.5 and in_span:
                axs.axvspan(span_start - 0.5, j - 0.5, color=context_color)
                in_span = False
        if in_span:
            axs.axvspan(span_start - 0.5, len(context_mask) - 0.5, color=context_color)

    if xmax:
        axs.set_xlim((0, xmax))
    if justimage:
        import matplotlib.pyplot as plt
        axs.axis('off')
        plt.tight_layout()
    else:
        axs.set_ylabel(ylabel)
        axs.set_xlabel('frame')
        axs.set_title(title or 'Spectrogram (dB)')
        fig.colorbar(im, ax=axs)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    im = Image.fromarray(rgba)
    if justimage: # remove tiny white border
        b = 15 # border size
        im = im.crop((b,b, im.size[0]-b, im.size[1]-b))
        #print(f"im.size = {im.size}")
    return im

def audio_spectrogram_image(waveform, power=2.0, sample_rate=48000, print=print, db=False, db_range=[35,120], justimage=False, log=False, figsize=(5, 4), context_mask=None):
    "Wrapper for calling above two routines at once, does Mel scale; Modified from PyTorch tutorial https://pytorch.org/tutorials/beginner/audio_feature_extractions_tutorial.html"
    melspec = mel_spectrogram(waveform, power=power, db=db, sample_rate=sample_rate, debug=log)
    melspec = melspec[0] # TODO: only left channel for now
    return spectrogram_image(melspec, title="MelSpectrogram", ylabel='mel bins (log freq)', db_range=db_range, justimage=justimage, figsize=figsize, context_mask=context_mask)

from matplotlib.ticker import AutoLocator 
def tokens_spectrogram_image(
        tokens,                # the embeddings themselves (in some diffusion codes these are called 'tokens')
        aspect='auto',         # aspect ratio of plot
        title='Embeddings',    # title to put on top
        ylabel='index',        # label for y axis of plot
        cmap='coolwarm',       # colormap to use. (default used to be 'viridis')
        symmetric=True,        # make color scale symmetric about zero, i.e. +/- same extremes
        figsize=(8, 4),       # matplotlib size of the figure
        dpi=100,               # dpi of figure
        mark_batches=False,     # separate batches with dividing lines
        debug=False,           # print debugging info
    ):
    "for visualizing embeddings in a spectrogram-like way"
    batch_size, dim, samples = tokens.shape
    embeddings = rearrange(tokens, 'b d n -> (b n) d')  # expand batches in time
    vmin, vmax = None, None
    if symmetric:
        vmax = torch.abs(embeddings).max()
        vmin = -vmax
        
    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    if symmetric: 
        subtitle = f'min={embeddings.min():0.4g}, max={embeddings.max():0.4g}'
        ax.set_title(title+'\n')
        ax.text(x=0.435, y=0.9, s=subtitle, fontsize=11, ha="center", transform=fig.transFigure)
    else:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('time frame (samples, in batches)')
    if mark_batches:
        intervals = np.arange(batch_size)*samples
        if debug: print("intervals = ",intervals)
        ax.vlines(intervals, -10, dim+10, color='black', linestyle='dashed', linewidth=1)

    im = ax.imshow(embeddings.cpu().numpy().T, origin='lower', aspect=aspect, interpolation='none', cmap=cmap, vmin=vmin,vmax=vmax) #.T because numpy is x/y 'backwards'
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return Image.fromarray(rgba)



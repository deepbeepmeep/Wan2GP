"""Keep callback layout construction out of Gradio's global startup context."""
from gradio import blocks, context


def install():
    if getattr(context.get_render_context, '_wangp_event_context', False):
        return
    original_get, original_set = context.get_render_context, context.set_render_context

    def get_render_context():
        if context.LocalContext.in_event_listener.get() and context.LocalContext.renderable.get() is None:
            return context.LocalContext.render_block.get()
        return original_get()

    def set_render_context(block):
        if context.LocalContext.in_event_listener.get() and context.LocalContext.renderable.get() is None:
            context.LocalContext.render_block.set(block)
        else:
            original_set(block)

    get_render_context._wangp_event_context = True
    # blocks imports these helpers by value; update both references.
    blocks.get_render_context = context.get_render_context = get_render_context
    blocks.set_render_context = context.set_render_context = set_render_context

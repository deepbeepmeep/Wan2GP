from dataclasses import dataclass
from typing import Any

from .secure_store import SecureStorageError
from .service import send_notification
from .settings import APPRISE_URLS_KEY, NOTIFY_GENERATION_KEY, NOTIFY_QUEUE_COMPLETE_KEY, NOTIFY_QUEUE_INTERRUPTED_KEY, SECURE_STORAGE_KEY, apprise_urls_text, configured_urls


@dataclass(frozen=True)
class NotificationConfigUI:
    urls: Any
    secure_storage: Any
    on_generation: Any
    on_queue_complete: Any
    on_queue_interrupted: Any

    @property
    def save_components(self) -> list[Any]:
        return [self.urls, self.secure_storage, self.on_generation, self.on_queue_complete, self.on_queue_interrupted]


def create_config_ui(gr: Any, config: dict[str, Any]) -> NotificationConfigUI:
    try:
        saved_urls = configured_urls(config)
        storage_error = ""
    except SecureStorageError as error:
        saved_urls, storage_error = [], str(error)
    gr.Markdown("### Remote Notifications\nEnter one or more space-separated [Apprise URLs](https://appriseit.com/services/). Prefer dedicated, revocable tokens or app passwords.")
    with gr.Row(elem_classes=["wangp-bottom-aligned-row"]):
        urls = gr.Textbox(value=apprise_urls_text(saved_urls), type="password", label="Apprise Destinations", info="Masked by default. Supports email, Telegram, Discord, WhatsApp gateways, webhooks, and other Apprise services.", scale=5)
        reveal = gr.Checkbox(value=False, label="Show / edit temporarily", scale=1, min_width=180)
    secure_storage = gr.Checkbox(value=config.get(SECURE_STORAGE_KEY, False), label="Store Destinations in OS Credential Manager", info="Recommended. wgp_config.json stores only an opaque identifier; WanGP retrieves the destinations when needed. No plaintext fallback is used.")
    if storage_error:
        gr.Markdown(f"⚠️ {storage_error} Re-enter the destinations or disable secure storage before saving.")
    with gr.Row():
        on_generation = gr.Checkbox(value=config.get(NOTIFY_GENERATION_KEY, False), label="After Each Generation")
        on_queue_complete = gr.Checkbox(value=config.get(NOTIFY_QUEUE_COMPLETE_KEY, False), label="When Queue Completes")
        on_queue_interrupted = gr.Checkbox(value=config.get(NOTIFY_QUEUE_INTERRUPTED_KEY, False), label="When Queue Is Interrupted")
    test_button = gr.Button("Send Test Notification")
    test_status = gr.Markdown()

    reveal.change(fn=lambda show: gr.Textbox(type="text" if show else "password"), inputs=[reveal], outputs=[urls], show_progress="hidden", api_name=False)

    def test_notification(destination_urls: str) -> str:
        result = send_notification({APPRISE_URLS_KEY: destination_urls, SECURE_STORAGE_KEY: False}, "WanGP test notification", "WanGP remote notifications are configured correctly.")
        if not result["sent"]:
            return f"Notification failed: {result['error']}"
        message = f"Notification sent to {result['destinations']} destination{'s' if result['destinations'] != 1 else ''}."
        return f"{message} {result['warning']}" if result.get("warning") else message

    test_button.click(fn=test_notification, inputs=[urls], outputs=[test_status], show_progress="hidden", api_name=False)
    return NotificationConfigUI(urls, secure_storage, on_generation, on_queue_complete, on_queue_interrupted)

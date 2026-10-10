# Gallery Workspaces

Gallery workspaces help you separate and organize media for different projects. Each workspace remembers which images, videos, and audio files appear in WanGP's galleries, which item is selected, and which gallery tab was open.

Use workspaces when you want to:

- keep unrelated projects out of each other's galleries
- return to a useful media collection after restarting WanGP
- give a Deepy session its own collection of source files and results
- browse more media than the main gallery displays at once
- move, copy, package, or remove groups of gallery items

## What a workspace saves

A workspace saves the state of the **Image / Video Gallery** and **Audio Gallery**, including:

- the ordered list of media in each gallery
- the selected image/video and selected audio
- whether the selection was following the latest result
- the active gallery tab
- the selected video time, when available

The workspace stores references to the original media files. It does not move or duplicate those files. This has several useful consequences:

- the same file can appear in more than one workspace without using additional disk space
- copying an item to another workspace adds another reference, not another file
- deleting a workspace removes the collection but leaves its media files on disk
- moving or deleting a referenced file outside WanGP makes it unavailable when the workspace is restored

This is different from a **Deepy Prime session workspace**, which is a real folder where Prime can create drafts, plans, prompts, and other project files. A dedicated Deepy session may have both: a private file workspace for working documents and a gallery workspace for media.

## Create and switch workspaces

The workspace selector is beside the gallery tabs in Gradio and in the gallery header of the Deepy Web app.

- Select a workspace to replace both galleries with its saved collection.
- Click **+** to create a new, empty workspace.
- Rename an ordinary workspace when its project changes.
- Delete an ordinary workspace when you no longer need the collection. Its media files stay on disk.

If every workspace is removed, WanGP creates a new empty **Default** workspace. Ordinary workspace names must be unique.

Workspace selection and gallery changes are synchronized across open Gradio pages and the Deepy Web app. Generations and imports made on one page appear in the active workspace on the others. Wait for active Deepy work and generation to finish before creating, deleting, or switching a workspace.

## How workspaces connect to galleries

The active workspace owns the gallery lists, order, and selections. New generated or imported media are added to that active collection. Selecting, reordering, ejecting, or importing media updates the workspace automatically; there is no Save Workspace button.

**Configuration > General > Media Visible per Gallery** controls how many of the newest items appear in the normal galleries. It does not remove older items from the workspace. Choose **All** (`0`) to show the complete collection in the main galleries, or use a smaller number to keep large galleries responsive.

When only the latest items are visible:

- older items remain saved in their original order
- ejecting a visible item brings the next older item back into view
- generating new media does not discard earlier results
- the workspace manager can browse and manage the complete collection

If the latest item is selected, a new output becomes the selection automatically. If you intentionally selected an older item, WanGP keeps that selection and viewing position.

## Use the workspace manager

In Gradio, click the **magnifier** beside the workspace selector to open the full-screen workspace manager. The manager is not available in the compact Deepy Web app.

Switch between **Images / Videos** and **Audio**. The grid is ordered from oldest to newest. Items marked **In gallery** are among the newest items currently shown in the normal gallery; every other item is still part of the workspace.

Select an item to view its available generation properties. For video, use the play control to inspect it. The manager loads large collections in pages, so you can organize a workspace without displaying every original media file at once.

### Select several items

- Click a tile to select it.
- Ctrl-click on Windows/Linux or Cmd-click on macOS to toggle individual items.
- Shift-click to select a range on the current page.
- Drag a rectangle from empty space between tiles to select an area.
- Use **Select page** to select every item on the displayed page.

Selections remain active while moving between pages.

### Reorder media

Drag a selected tile to move the entire selection before another item. While dragging, hover over a page arrow to move to another page, then drop the selection at its destination. Use **Move to start** or **Move to end** for large collections.

Order matters because the normal gallery shows the last N items in workspace order. Reordering therefore changes which items receive the **In gallery** badge and appear in the compact gallery.

### Workspace actions

- **Eject** removes the selected entries from the current workspace but leaves their files on disk. Use this to clean up a collection safely.
- **Copy to workspace** adds the selected entries to another workspace. Both workspaces reference the same files; no media is duplicated.
- **ZIP** downloads the selected files together. WanGP packages the originals without re-encoding them.
- **Import** copies selected local uploads into WanGP's configured output folder and adds them to the active workspace. Original filenames are preserved when possible. Reimporting the same file reuses it; a different file with the same name receives a numbered suffix.
- **Delete files** permanently deletes the selected files and removes their references from every workspace. It requires confirmation and waits for Deepy and generation to become idle. Use **Eject** instead unless the files themselves should be erased.

Deleting a file is the only workspace-manager action that removes the underlying media from disk.

## Deepy sessions and workspaces

Deepy's multisession setting controls how saved conversations use gallery workspaces:

- **Multisessions with selectable Workspace** associates the conversation with whichever ordinary workspace you select. Use this when several sessions should share or switch among common media collections.
- **Multisessions with dedicated Workspace** creates a separate gallery workspace for each conversation. Use this when every Deepy project should stay self-contained.

Dedicated workspaces display a robot icon; ordinary workspaces display a folder icon. In dedicated mode, Deepy's Web gallery is locked to its current session workspace. The main Gradio gallery may browse another workspace, but Deepy's next import or request returns its media workflow to the dedicated collection.

A dedicated workspace is owned by its Deepy session:

- rename the session to rename its workspace
- delete the session to delete its workspace record
- resume the session to restore its workspace and media references
- manage the session rather than trying to rename or delete its workspace directly

Deleting a Deepy session or dedicated workspace does not delete the referenced media files. Session media portability is a separate choice: **Keep links to Gallery files** relies on the originals, while **Copy Gallery files into each session** keeps session-owned copies. See [Deepy sessions and workspaces](DEEPY.md#sessions-and-workspaces).

## Synchronization between pages

The Deepy Web app and all open Gradio pages share workspace selection, gallery contents, media order, and selections while connected to the same WanGP process. A change on one page is reflected on the others.

Each Gradio page's unsent generation form remains independent. Changing the model, prompt, or input fields on one page does not rewrite another page's draft. Workspaces synchronize completed generations and gallery state, not unfinished form entries.

In dedicated-session mode, the Deepy Web gallery deliberately continues to show its session-owned workspace even if someone browses another workspace in Gradio.

## Last activity, protection, and automatic archiving

The workspace manager displays **Last Activity**, based on the most recent media addition, removal, or reorder. Merely viewing, selecting, renaming, or protecting a workspace does not make it active again.

Click the **lock** to protect an important workspace from automatic archiving. Click the **broom** to choose how long an unlocked workspace may remain inactive: one to three weeks, one to eleven months, one year, or disabled.

Archiving runs when WanGP starts; it is not a background cleanup timer. When an unlocked workspace is older than the selected period:

- the workspace disappears from active selectors
- its media files remain at their original locations
- a dedicated Deepy session associated with it is also hidden from the active session list

**Recommendation:** leave automatic archiving disabled until you have established a cleanup routine. If you enable it, lock every active or long-term project.

### Restore an archived workspace

There is currently no restore button. To restore one manually:

1. Stop WanGP.
2. Open the configured workspace folder, normally `workspaces/`.
3. Move the workspace JSON file from `workspaces/archives/` back into `workspaces/` without overwriting another file.
4. Set `archive_protected` to `true` in that JSON so it is not archived again at the next startup.
5. For a dedicated workspace, remove `archived_at` from the matching Deepy session's `session.json`.
6. Restart WanGP.

The workspace and session reappear if their referenced media still exist.

## Storage and backups

Workspace definitions are stored in `workspaces/` at the WanGP installation root. Launch WanGP with `--workspaces-dir FOLDER` to use another location.

A backup of this folder preserves workspace names, ordering, selections, and media references, but not the media itself. Back up the referenced output/media folders too when you need a complete, portable archive. For one selected collection, the workspace manager's **ZIP** action is the simpler option.

## Recommended workflows

- **Quick experiments:** use one ordinary workspace and eject rejected results periodically.
- **Separate client or creative projects:** create one ordinary workspace per project and give each a clear name.
- **Long Deepy productions:** use multisessions with dedicated workspaces so conversation context and gallery media stay aligned.
- **Shared source library:** use selectable workspaces and copy references into project workspaces without duplicating files.
- **Small main galleries:** set a modest **Media Visible per Gallery** value and use the manager for older media.
- **Safe cleanup:** eject first; use permanent deletion only after confirming the same files are not needed by another workspace.
- **Long-term retention:** protect the workspace, and back up both its definition and its referenced media.

---

> Applies to: Creating, selecting, synchronizing, managing, archiving, restoring, and backing up WanGP gallery workspaces; their relationship to galleries, media files, Deepy sessions, and the Gradio workspace manager.

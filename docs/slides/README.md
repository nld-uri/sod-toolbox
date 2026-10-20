# Slides

`SOD_toolbox_slides.pdf` is a vector export of the live deck (animations appear as one still each).
`SOD_toolbox.pptx` is the PowerPoint version: slide images, the four animations as MP4 movies, and the speaker notes.

| Path | What it is |
|---|---|
| `deck/project/` | the deck sources (`deck.json`, one HTML file per slide, NLD design tokens) |
| `assets/` | the images the slides use; `blobmap.json` maps the deck's asset ids to these files |
| `lib/export_pdf.py` | renders the deck to PDF with Chromium (`python lib/export_pdf.py deck <abs blobmap> out.pdf`) |
| `lib/export_pptx.py` | builds the PowerPoint from the PDF, the MP4s in `../animations` and the speaker notes |

The PDF export maps each animation to its still (`*_still.png`); the PowerPoint plays the MP4.

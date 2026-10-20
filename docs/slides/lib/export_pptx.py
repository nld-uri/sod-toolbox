"""Build a PowerPoint file from the Slides-type deck: each slide is the rendered page (from the deck PDF), slides with
an animation carry its MP4 over the image area, and the speaker notes are kept.

    python export_pptx.py <deck_dir> <deck.pdf> <anim_dir> <blobmap.json> <out.pptx>

blobmap maps blob ids to file names; an id whose file ends in _deck.gif is played as <anim_dir>/<name>.mp4.
"""
import sys, os, re, json, html, subprocess, tempfile
from pptx import Presentation
from pptx.util import Emu
from PIL import Image

EMU = 12192000 / 1920


def main(deck, pdf, anim, bmap, out):
    order = json.load(open(os.path.join(deck, 'project/deck.json')))['order']
    bmap = json.load(open(bmap))
    tmp = tempfile.mkdtemp()
    subprocess.run(['pdftoppm', '-png', '-scale-to-x', '2560', '-scale-to-y', '1440', pdf, os.path.join(tmp, 's')], check=True)
    pngs = sorted(f for f in os.listdir(tmp) if f.endswith('.png'))
    assert len(pngs) == len(order), (len(pngs), len(order))
    prs = Presentation(); prs.slide_width, prs.slide_height = 12192000, 6858000
    blank = prs.slide_layouts[6]
    for sid, png in zip(order, pngs):
        t = open(os.path.join(deck, 'project/slides', sid + '.html')).read()
        s = prs.slides.add_slide(blank)
        img = os.path.join(tmp, png)
        s.shapes.add_picture(img, 0, 0, prs.slide_width, prs.slide_height)
        for m in re.finditer(r'<img src="/_blob/([0-9a-f]{32})"[^>]*style="([^"]*)"', t):
            f = os.path.basename(bmap.get(m.group(1), ''))
            if not f.endswith('_deck.gif'):
                continue
            st = dict(kv.split(':', 1) for kv in m.group(2).replace(' ', '').split(';') if ':' in kv)
            L, T, W, H = (float(st[k].replace('px', '')) for k in ('left', 'top', 'width', 'height'))
            mp4 = os.path.join(anim, f.replace('_deck.gif', '.mp4'))
            k = 2560 / 1920
            poster = os.path.join(tmp, sid + '_poster.png')
            Image.open(img).crop((round(L * k), round(T * k), round((L + W) * k), round((T + H) * k))).save(poster)
            s.shapes.add_movie(mp4, Emu(round(L * EMU)), Emu(round(T * EMU)), Emu(round(W * EMU)), Emu(round(H * EMU)),
                               poster_frame_image=poster, mime_type='video/mp4')
        n = re.search(r'<aside>(.*?)</aside>', t, re.S)
        if n:
            s.notes_slide.notes_text_frame.text = html.unescape(n.group(1)).strip()
    prs.save(out)
    print(out, len(order), 'slides', os.path.getsize(out) // 1024, 'KB')


if __name__ == '__main__':
    main(*sys.argv[1:6])

"""Render the academic report, embedding Times New Roman when available."""
import argparse
import re
from html import escape
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def register_fonts(directory=None):
    folder = Path(directory) if directory else Path('C:/Windows/Fonts')
    files = ['times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf']
    if all((folder / filename).is_file() for filename in files):
        names = ['ReportTimes', 'ReportTimes-Bold', 'ReportTimes-Italic', 'ReportTimes-BoldItalic']
        for name, filename in zip(names, files):
            pdfmetrics.registerFont(TTFont(name, str(folder / filename)))
        pdfmetrics.registerFontFamily('ReportTimes', normal=names[0], bold=names[1], italic=names[2], boldItalic=names[3])
        return names, 'Times New Roman (embedded)'
    # Fonts are not redistributed. Supply --font-dir when rebuilding on Colab.
    return ['Times-Roman', 'Times-Bold', 'Times-Italic', 'Times-BoldItalic'], 'PDF Times fallback'


def inline(text):
    text = escape(text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#15253b">\1</link>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
    return text.replace('`', '')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font-dir', type=Path, help='Directory containing times.ttf, timesbd.ttf, timesi.ttf and timesbi.ttf')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'Report/Technical_Report.md'
    output = source.with_suffix('.pdf')
    (regular, bold, italic, _), description = register_fonts(args.font_dir)
    styles = {
        'body': ParagraphStyle('Body', fontName=regular, fontSize=12, leading=16, alignment=TA_JUSTIFY, spaceAfter=8, allowWidows=0, allowOrphans=0),
        'title': ParagraphStyle('ReportTitle', fontName=bold, fontSize=19, leading=23, alignment=TA_CENTER, spaceAfter=12, keepWithNext=True),
        'subtitle': ParagraphStyle('Subtitle', fontName=regular, fontSize=11, leading=14, alignment=TA_CENTER, spaceAfter=18, keepWithNext=True),
        'h1': ParagraphStyle('Section', fontName=bold, fontSize=14, leading=18, spaceBefore=14, spaceAfter=7, keepWithNext=True),
        'h2': ParagraphStyle('Subsection', fontName=bold, fontSize=12, leading=16, spaceBefore=10, spaceAfter=5, keepWithNext=True),
        'cell': ParagraphStyle('Cell', fontName=regular, fontSize=9.5, leading=12, spaceAfter=0),
        'header': ParagraphStyle('TableHeader', fontName=bold, fontSize=9.5, leading=12, spaceAfter=0),
        'caption': ParagraphStyle('Caption', fontName=italic, fontSize=10, leading=13, alignment=TA_CENTER, spaceBefore=6, spaceAfter=12),
        'reference': ParagraphStyle('Reference', fontName=regular, fontSize=10.5, leading=14, alignment=TA_LEFT, leftIndent=16, firstLineIndent=-16, spaceAfter=6),
    }
    doc = SimpleDocTemplate(str(output), pagesize=(595.28, 841.89), leftMargin=66, rightMargin=66, topMargin=57, bottomMargin=53,
        title='FastSpeech 2 Speech Synthesis for English and Telugu', subject='Architecture, training and evaluation')
    story = []
    lines = source.read_text(encoding='utf8').splitlines()
    i = 0
    figure_number = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith('|'):
            raw = [line]
            while i < len(lines) and lines[i].strip().startswith('|'):
                raw.append(lines[i].strip())
                i += 1
            rows = []
            for entry in raw:
                cells = [c.strip() for c in entry.strip('|').split('|')]
                if all(set(c) <= set('-: ') for c in cells):
                    continue
                style = styles['header'] if not rows else styles['cell']
                rows.append([Paragraph(inline(c), style) for c in cells])
            n = len(rows[0])
            fractions = [1/n] * n
            if n == 2:
                fractions = [.29, .71]
            elif n == 5 and 'Interpretation' in raw[0]:
                fractions = [.08, .12, .15, .15, .50]
            elif n == 5 and 'Best epoch' in raw[0]:
                fractions = [.16, .28, .16, .20, .20]
            table = Table(rows, colWidths=[doc.width*f for f in fractions], repeatRows=1, hAlign='CENTER')
            table.setStyle(TableStyle([
                ('LINEABOVE', (0,0), (-1,0), .8, colors.black),
                ('LINEBELOW', (0,0), (-1,0), .5, colors.black),
                ('LINEBELOW', (0,-1), (-1,-1), .8, colors.black),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 5), ('RIGHTPADDING', (0,0), (-1,-1), 5),
                ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ]))
            story.extend([table, Spacer(1,10)])
            continue
        match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line)
        if match:
            figure_number += 1
            figure = Image(str((source.parent / match[2]).resolve()))
            ratio = figure.imageHeight / figure.imageWidth
            figure.drawWidth = doc.width
            figure.drawHeight = doc.width * ratio
            caption = Paragraph(inline(f'Figure {figure_number}. {match[1]}'), styles['caption'])
            story.append(KeepTogether([figure, caption]))
            continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), styles['title']))
            continue
        if line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), styles['h1']))
            continue
        if line.startswith('### '):
            story.append(Paragraph(inline(line[4:]), styles['h2']))
            continue
        style = styles['body']
        if line.startswith('*Technical report'):
            style = styles['subtitle']
        elif line.startswith('*Table '):
            style = ParagraphStyle('TableCaption', parent=styles['caption'], keepWithNext=True, spaceAfter=7)
        elif re.match(r'^\[\d+\]', line):
            style = styles['reference']
        story.append(Paragraph(inline(line), style))

    def footer(canvas, document):
        canvas.setStrokeColor(colors.HexColor('#aaaaaa'))
        canvas.setLineWidth(.3)
        canvas.line(doc.leftMargin, 40, 595.28-doc.rightMargin, 40)
        canvas.setFont(regular, 9)
        canvas.setFillColor(colors.HexColor('#444444'))
        canvas.drawString(doc.leftMargin, 27, 'FastSpeech 2: English and Telugu')
        canvas.drawRightString(595.28-doc.rightMargin, 27, str(document.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f'Rendered {output}; typeface: {description}')


if __name__ == '__main__':
    main()

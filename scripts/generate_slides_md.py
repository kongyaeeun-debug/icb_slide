#!/usr/bin/env python3
"""Generate a 20-slide Markdown file for pandoc from the report and plots."""
from pathlib import Path
import re

REPORT = Path('reports/penguins_report.md')
OUT = Path('reports/penguins_slides.md')
PLOTS_DIR = Path('plots')

text = REPORT.read_text(encoding='utf-8')

# Extract headings and paragraphs for plots
# Pattern: ### <caption>\n\n![...](../plots/xx.png)\n\n<insight paragraph>\n\n
plot_pattern = re.compile(r"###\s+(.*?)\n\n!\[.*?\]\((../plots/.*?\.png)\)\n\n(.*?)(?=\n###|\n##|\n---|\Z)", re.S)
plots = plot_pattern.findall(text)
plot_map = {cap.strip(): {'img': img.strip(), 'insight': ins.strip()} for cap, img, ins in plots}

# Extract Cross-tab table (header '## Cross-tab:') and Pivot
crosstab_pattern = re.compile(r"## Cross-tab:.*?\n\n(\|[\s\S]*?)\n\n", re.S)
pivot_pattern = re.compile(r"## Pivot:.*?\n\n(\|[\s\S]*?)\n\n", re.S)
crosstab_md = crosstab_pattern.search(text)
if crosstab_md:
    crosstab_table = crosstab_md.group(1).strip()
else:
    crosstab_table = ''
pivot_md = pivot_pattern.search(text)
if pivot_md:
    pivot_table = pivot_md.group(1).strip()
else:
    pivot_table = ''

# Extract short overview pieces
shape_match = re.search(r"- Shape: \((\d+),\s*(\d+)\)", text)
shape = shape_match.group(0) if shape_match else ''

head_match = re.search(r"### Head\n\n(\|[\s\S]*?)\n\n", text)
head_table = head_match.group(1).strip() if head_match else ''

summary_match = re.search(r"## Summary statistics\n\n(\|[\s\S]*?)\n\n", text)
summary_table = summary_match.group(1).strip() if summary_match else ''

# Plan slides (20): create list of slide dicts
slides = []
# 1 Title
slides.append({'title': '펭귄 데이터셋 분석', 'content': '발표자: 자동 생성 스크립트\n날짜: 자동'} )
# 2 Overview
slides.append({'title': '데이터 개요', 'content': f"{shape}\n\n{head_table}"})
# 3 Missing values & summary
missing_match = re.search(r"## Missing values\n\n(\|[\s\S]*?)\n\n", text)
missing_table = missing_match.group(1).strip() if missing_match else ''
slides.append({'title': '결측치 및 요약 통계', 'content': f"{missing_table}\n\n{summary_table}"})
# 4 Value counts
value_counts_match = re.search(r"## Value counts\n\n([\s\S]*?)\n\n## Grouped summary", text)
value_counts_block = value_counts_match.group(1).strip() if value_counts_match else ''
# replace sub-headers and wrap as code block to avoid pandoc splitting into separate slides
if value_counts_block:
    val_block_clean = value_counts_block.replace('### ', '** ')
    value_counts_block = '```\n' + val_block_clean + '\n```'
slides.append({'title': '범주형 변수 분포', 'content': value_counts_block})
# 5 Grouped summary
grouped_match = re.search(r"## Grouped summary by species\n\n(\|[\s\S]*?)\n\n", text)
grouped_block = grouped_match.group(1).strip() if grouped_match else ''
slides.append({'title': '종별 요약 통계', 'content': grouped_block})
# 6 Cross-tab & Pivot together
cross_and_pivot = ''
if crosstab_table:
    cross_and_pivot += '교차표: species vs sex\n\n' + crosstab_table + '\n\n'
if pivot_table:
    cross_and_pivot += '피봇: species × island 평균\n\n' + pivot_table
# wrap as code block
if cross_and_pivot:
    cross_and_pivot = '```\n' + cross_and_pivot + '\n```'
slides.append({'title': '교차표 및 피봇', 'content': cross_and_pivot})
# Now slides for plots; include all 15 images across slides 7-19, combining some two-per-slide
plot_order = [
    '부리 길이 히스토그램 (mm)',
    '종별 부리 길이 분포 (히스토그램)',
    '종별 지느러미 길이 KDE',
    '종별 부리 길이 박스플롯',
    '종별 부리 깊이 바이올린플롯',
    '종별 개체수',
    '종별 평균 체중 (g)',
    '부리 길이 vs 지느러미 길이 (종/성별)',
    '지느러미 길이와 체중 회귀',
    '수치 변수 페어플롯',
    '수치 변수 상관관계',
    '종별 부리 길이 스웜플롯',
    '섬별 부리 길이 분포',
    '섬별 종 개수 (그룹화 막대)',
    '섬별 종 개수 (누적 막대)'
]

# We'll create slides 7-19 with images, combining some to fit 13 slides for 15 images
# Strategy: combine indices 5&11 (violin & swarm), 6&7 (count & bar) -> done below
i = 6
while i < len(plot_order):
    title = plot_order[i]
    img = plot_map.get(title, {}).get('img')
    ins = plot_map.get(title, {}).get('insight','')
    # Combine certain pairs
    if title == '종별 부리 깊이 바이올린플롯':
        # combine with swarm (which appears later)
        img2 = plot_map.get('종별 부리 길이 스웜플롯', {}).get('img')
        ins2 = plot_map.get('종별 부리 길이 스웜플롯', {}).get('insight','')
        content = ''
        if img:
            content += f"![{title}]({img})\n\n"
        if img2:
            content += f"![종별 부리 길이 스웜플롯]({img2})\n\n"
        content += f"{ins}\n\n{ins2}"
        slides.append({'title': '부리 깊이 분포 및 개별 점 분포', 'content': content})
        # skip ahead past the swarm element which is earlier in order; find its index and skip when encountered
        # remove swarm from list by setting its entry to None
        try:
            swarm_idx = plot_order.index('종별 부리 길이 스웜플롯')
            plot_order[swarm_idx] = None
        except ValueError:
            pass
    elif title == '종별 개체수':
        # combine with 종별 평균 체중
        img2 = plot_map.get('종별 평균 체중 (g)', {}).get('img')
        ins2 = plot_map.get('종별 평균 체중 (g)', {}).get('insight','')
        content = ''
        if img:
            content += f"![{title}]({img})\n\n"
        if img2:
            content += f"![종별 평균 체중 (g)]({img2})\n\n"
        content += f"{ins}\n\n{ins2}"
        slides.append({'title': '표본수 및 평균 체중 비교', 'content': content})
        # mark the paired entry as None to skip later
        try:
            bar_idx = plot_order.index('종별 평균 체중 (g)')
            plot_order[bar_idx] = None
        except ValueError:
            pass
    elif title is None:
        pass
    else:
        # normal single slide
        content = ''
        if img:
            content += f"![{title}]({img})\n\n"
        content += f"{ins}"
        slides.append({'title': title, 'content': content})
    i += 1

# At this point, ensure we included all images; for any remaining (not None) entries we add slides
for title in plot_order:
    if not title:
        continue
    if any(s['title']==title for s in slides):
        continue
    img = plot_map.get(title, {}).get('img')
    ins = plot_map.get(title, {}).get('insight','')
    content = ''
    if img:
        content += f"![{title}]({img})\n\n"
    content += ins
    slides.append({'title': title, 'content': content})

# Finally add final summary slide to make total 20
slides.append({'title': '요약 및 결론', 'content': '주요 인사이트:\n- Gentoo 종은 평균 체중이 가장 큼\n- 지느러미 길이는 체중 예측에 유용함\n- 섬별 분포 차이 관찰됨'})

# Trim or pad slides to 20 exactly
if len(slides) > 20:
    slides = slides[:20]
elif len(slides) < 20:
    # add blank slides if needed
    while len(slides) < 20:
        slides.append({'title': f'부록 {len(slides)+1}', 'content': ''})

# Write markdown with slide separators. Use level-1 headers as slide titles for pandoc pptx
out_lines = []
for s in slides:
    out_lines.append(f"# {s['title']}\n")
    if s['content']:
        # avoid sub-headers that pandoc may split into extra slides; convert '###' and '##' to bold text
        content = s['content'].replace('### ', '** ').replace('## ', '** ')
        out_lines.append(content + '\n')
    out_lines.append('\n')

text_out = '\n'.join(out_lines)
# also replace any remaining header-like occurrences
text_out = text_out.replace('### ', '** ').replace('## ', '** ')
OUT.write_text(text_out, encoding='utf-8')
print(f"Wrote slides to {OUT} ({len(slides)} slides)")

"""把含 Mermaid + LaTeX(KaTeX) + 中文的 Markdown 报告导出为 PDF。

思路：在浏览器里渲染（marked + mermaid + KaTeX），再用 headless Chrome 打印成 PDF。
这样三类内容都按其在 Markdown+KaTeX 预览器里的原生形态呈现，宽表格也能自适应。

用法：
  uv run python scripts/md2pdf.py [reports/中期进展报告.md]

依赖：本机 Chrome/Edge + 联网（marked/mermaid/katex 走 CDN），无需安装额外 Python 包。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  @page { size: A4; margin: 16mm; }
  body { font-family: "Microsoft YaHei","PingFang SC","Segoe UI",sans-serif;
         font-size: 11pt; line-height: 1.55; color: #111; }
  #out { max-width: 900px; margin: 0 auto; }
  h1 { font-size: 19pt; } h2 { font-size: 15pt; margin-top: 1.1em; }
  h3 { font-size: 12.5pt; }
  h2, h3 { page-break-after: avoid; }
  table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 8px 0;
          page-break-inside: avoid; }
  th, td { border: 1px solid #999; padding: 4px 6px; text-align: left; vertical-align: top;
           overflow-wrap: anywhere; }
  th { background: #eef1f5; }
  th:first-child, td:first-child { white-space: nowrap; }  /* 首列不逐字竖排 */
  code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px;
         font-family: Consolas,"Cascadia Mono",monospace; font-size: 9pt; }
  pre { background: #f6f8fa; padding: 9px 11px; border-radius: 5px;
        white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-all;
        font-size: 8.5pt; line-height: 1.4; page-break-inside: avoid; }
  pre code { background: none; padding: 0; white-space: pre-wrap; }
  img { max-width: 100%; height: auto; display: block; margin: 10px auto; }
  .mermaid { text-align: center; margin: 12px 0; page-break-inside: avoid; }
  blockquote { border-left: 4px solid #cbd5e1; margin: 8px 0; padding: 2px 12px;
               color: #444; background: #fafbfc; }
</style>
</head>
<body>
<script type="text/markdown" id="src">__MARKDOWN__</script>
<div id="out"></div>
<script>
(async function () {
  const SEP = String.fromCharCode(1);
  const raw = document.getElementById('src').textContent;
  // 1) 保护数学（先块级 $$..$$，再行内 $..$），避免被 Markdown 解析破坏下划线等
  const math = [];
  let s = raw.replace(/\$\$([\s\S]+?)\$\$/g, function (_, p) {
    math.push({ d: true, t: p }); return SEP + 'M' + (math.length - 1) + SEP;
  });
  s = s.replace(/(^|[^\\])\$([^\$\n]+?)\$/g, function (_, pre, p) {
    math.push({ d: false, t: p }); return pre + SEP + 'M' + (math.length - 1) + SEP;
  });
  // 2) 保护 mermaid 代码块
  const mer = [];
  s = s.replace(/```mermaid\r?\n([\s\S]+?)```/g, function (_, p) {
    mer.push(p.replace(/\s+$/, '')); return '\n' + SEP + 'G' + (mer.length - 1) + SEP + '\n';
  });
  // 3) Markdown -> HTML
  let html = marked.parse(s);
  // 4) 还原 mermaid 占位为容器（源码稍后用 textContent 注入，避免 HTML 解析）
  html = html.replace(new RegExp(SEP + 'G(\\d+)' + SEP, 'g'),
    function (_, i) { return '<div class="mermaid" data-i="' + i + '"></div>'; });
  // 5) 还原数学占位为 KaTeX 定界符文本
  html = html.replace(new RegExp(SEP + 'M(\\d+)' + SEP, 'g'),
    function (_, i) { const m = math[i]; return m.d ? '$$' + m.t + '$$' : '$' + m.t + '$'; });
  document.getElementById('out').innerHTML = html;
  document.querySelectorAll('.mermaid').forEach(function (el) {
    el.textContent = mer[+el.dataset.i];
  });
  // 6) 渲染 mermaid 与数学
  mermaid.initialize({ startOnLoad: false, theme: 'default' });
  try { await mermaid.run({ querySelector: '.mermaid' }); } catch (e) { console.error('mermaid', e); }
  try {
    renderMathInElement(document.getElementById('out'), {
      delimiters: [{ left: '$$', right: '$$', display: true },
                   { left: '$', right: '$', display: false }],
      throwOnError: false,
    });
  } catch (e) { console.error('katex', e); }
  document.title = 'READY';
})();
</script>
</body>
</html>
"""


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    raise SystemExit("未找到 Chrome/Edge，请手动指定可执行路径。")


def main() -> None:
    md_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports" / "中期进展报告.md"
    md_path = md_path if md_path.is_absolute() else ROOT / md_path
    md_text = md_path.read_text(encoding="utf-8")

    html_path = md_path.with_name("_print_" + md_path.stem + ".html")
    html_path.write_text(HTML_TEMPLATE.replace("__MARKDOWN__", md_text), encoding="utf-8")

    pdf_tmp = md_path.with_name("_print_" + md_path.stem + ".pdf")   # ASCII 临时名
    pdf_final = md_path.with_suffix(".pdf")                          # 最终（可含中文）

    chrome = find_chrome()
    url = html_path.resolve().as_uri()
    cmd = [
        chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=25000",
        f"--print-to-pdf={pdf_tmp}", url,
    ]
    print("[md2pdf] 渲染中：", md_path.name)
    subprocess.run(cmd, check=True, timeout=180)

    if not pdf_tmp.exists() or pdf_tmp.stat().st_size < 2000:
        raise SystemExit("PDF 生成失败或过小，请检查 Chrome 输出。")
    pdf_tmp.replace(pdf_final)
    html_path.unlink(missing_ok=True)
    print(f"[md2pdf] 完成：{pdf_final}  ({pdf_final.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

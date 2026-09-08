"""Keep the small generated result tables together and avoid stranded conclusions."""
from pathlib import Path
import re,sys
for filename in sys.argv[1:]:
    p=Path(filename);s=p.read_text()
    pattern=r'\{\\def\\LTcaptype\{none\} % do not increment counter\n\\begin\{longtable\}\[\]\{([^\n]+)\}\n(.*?)\\end\{longtable\}\n\}'
    def compact(m):
        content=m[2]
        header,rest=content.split('\\endhead\n',1)
        _,body=rest.split('\\endlastfoot\n',1)
        return ('\\par\\medskip\\noindent\\begin{minipage}{\\linewidth}\\centering\n'
                '\\begin{tabular}{'+m[1]+'}\n'+header+body+
                '\\bottomrule\n\\end{tabular}\n\\end{minipage}\\par\\medskip')
    s=re.sub(pattern,compact,s,flags=re.S)
    if p.name=='adaptive-deployment-control.tex':
        s=s.replace('\\section{Reproduction and conclusion}', '\\section{Reproduction and conclusion}')
    # Keep figures near their discussion; a barrier before final supplement text
    # prevents the conclusion being stranded after a float-only page.
    s=s.replace('\\usepackage{graphicx}', '\\usepackage{graphicx}\n\\usepackage{placeins}')
    if p.name=='supplementary.tex':
        s=s.replace('\\section{Reproducibility}', '\\FloatBarrier\n\\section{Reproducibility}')
    p.write_text(s)

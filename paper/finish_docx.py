"""Normalize Pandoc tables and theme overrides for reliable Word/LibreOffice layout.
Uses only Python's standard library so the documented build needs no DOCX package.
"""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import xml.etree.ElementTree as E
import sys
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
E.register_namespace('w',W)
E.register_namespace('m','http://schemas.openxmlformats.org/officeDocument/2006/math')
E.register_namespace('r','http://schemas.openxmlformats.org/officeDocument/2006/relationships')
q=lambda name:'{'+W+'}'+name
p=Path(sys.argv[1])
with ZipFile(p) as z: parts={n:z.read(n) for n in z.namelist()}
styles=E.fromstring(parts['word/styles.xml'])
for st in styles.findall(q('style')):
    for b in list(st.iter(q('pBdr'))):
        for parent in st.iter():
            if b in list(parent): parent.remove(b)
    for fonts in st.iter(q('rFonts')):
        for k in list(fonts.attrib):
            if 'theme' in k.lower():del fonts.attrib[k]
    if st.get(q('type'))=='paragraph':
        for color in st.iter(q('color')):
            color.attrib.clear();color.set(q('val'),'000000')
parts['word/styles.xml']=E.tostring(styles,encoding='utf-8',xml_declaration=True)
doc=E.fromstring(parts['word/document.xml'])
# Keep citation text in this paragraph as ordinary runs for Word compatibility.
for paragraph in doc.iter(q('p')):
    if 'The claimed convergence threshold' in ''.join(paragraph.itertext()):
        for link in list(paragraph.findall(q('hyperlink'))):
            index=list(paragraph).index(link)
            for child in reversed(list(link)):paragraph.insert(index,child)
            paragraph.remove(link)
for style in doc.iter(q('pStyle')):
    if style.get(q('val'))=='Compact':style.set(q('val'),'Normal')
for style in doc.iter(q('tblStyle')):
    style.set(q('val'),'TableGrid')
for tbl in doc.iter(q('tbl')):
    cols=tbl.find(q('tblGrid')).findall(q('gridCol'))
    n=len(cols);widths=[int(9500*(.32 if i==0 else .68/(n-1))) for i in range(n)]
    for col,width in zip(cols,widths):col.set(q('w'),str(width))
    pr=tbl.find(q('tblPr'))
    for tag in ['tblpPr','tblW','tblLayout','tblCellMar','tblBorders']:
        for old in pr.findall(q(tag)):pr.remove(old)
    E.SubElement(pr,q('tblW'),{q('type'):'dxa',q('w'):str(sum(widths))})
    E.SubElement(pr,q('tblLayout'),{q('type'):'fixed'})
    margins=E.SubElement(pr,q('tblCellMar'))
    for edge in ['top','bottom','left','right']:E.SubElement(margins,q(edge),{q('w'):'70',q('type'):'dxa'})
    borders=E.SubElement(pr,q('tblBorders'))
    for edge in ['top','bottom','insideH']:E.SubElement(borders,q(edge),{q('val'):'single',q('sz'):'4',q('color'):'DDDDDD'})
    rows=tbl.findall(q('tr'))
    for ri,row in enumerate(rows):
        rowpr=row.find(q('trPr'))
        if rowpr is None:rowpr=E.SubElement(row,q('trPr'))
        E.SubElement(rowpr,q('cantSplit'))
        for i,cell in enumerate(row.findall(q('tc'))):
            props=cell.find(q('tcPr'))
            if props is None:props=E.SubElement(cell,q('tcPr'))
            E.SubElement(props,q('tcW'),{q('type'):'dxa',q('w'):str(widths[i])})
            E.SubElement(props,q('vAlign'),{q('val'):'center'})
            if ri==0:E.SubElement(props,q('shd'),{q('fill'):'EEEEEE'})
            for paragraph in cell.findall(q('p')):
                pp=paragraph.find(q('pPr'))
                if pp is None:pp=E.SubElement(paragraph,q('pPr'))
                for el in pp.findall(q('jc')):pp.remove(el)
                E.SubElement(pp,q('jc'),{q('val'):'left' if i==0 else 'center'})
                if ri < len(rows)-1:E.SubElement(pp,q('keepNext'))
                E.SubElement(pp,q('keepLines'))
for element in doc.iter():
    if element.text=='\u200b':element.text=''
parts['word/document.xml']=E.tostring(doc,encoding='utf-8',xml_declaration=True)
with ZipFile(p,'w',ZIP_DEFLATED) as z:
    for n,data in parts.items():z.writestr(n,data)
